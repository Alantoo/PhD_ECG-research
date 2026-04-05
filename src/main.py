import glob
import os
from pathlib import Path

import wfdb
import wfdb.processing

import dotenv
import numpy as np
import simplejson as json
from apiflask import APIFlask
from flask import jsonify, Response, request
from flask_cors import CORS, cross_origin
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.exceptions import NotFound

from artifacts_config import ArtifactsConfig
from classification_signal import EnsembleSignalClassifier
from gen_sig import to_np_array
from get_config.ecg_config import ECGConfig
from my_helpers.data_preparation import DataPreparation
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.artifact_detection import detect_artifacts, detect_by_zone_intervals, mad_stats
from my_helpers.plot_statistics import PlotStatistics
from my_helpers.read_data.read_data_file import ReadDataFile
from pipeline import butter_bandpass, apply_detrend, \
    defaults_process_ecg_pipeline, defaults_decimate_signal
from preparedSignal import PreparedSignal
from simulation import Simulation

dotenv.load_dotenv()


def refresh_datafiles():
    for db_id, db in databases.items():
        files = [Path(f).stem for f in glob.glob(db['path'] + "/*.dat")]
        file_map = dict()
        for file in files:
            file_map[file] = db['path'] + "/" + file

        db['datafiles'] = file_map


def read_physionet_database_directory(base):
    if base.endswith('/files') is False:
        base += '/files'

    db = dict()
    for f in os.scandir(base):
        if f.is_dir() is False:
            continue

        for child_f in os.scandir(f.path):
            if child_f.is_dir() is False:
                continue
            dbid = f'{f.name}-{child_f.name}'
            db[dbid] = {
                'id': dbid,
                'display_name': f'{f.name} {child_f.name}',
                'path': child_f.path,
                'datafiles': dict(),
            }
    return db


app = APIFlask(__name__)
cors = CORS(app)
# config_block = 'H_P001_PPG_S_S1'

BASE_PHYSIONET_LOCATION = os.environ.get('APP_BASE_PHYSIONET_LOCATION')
if BASE_PHYSIONET_LOCATION is None:
    raise Exception('APP_BASE_PHYSIONET_LOCATION is required')

databases = read_physionet_database_directory(BASE_PHYSIONET_LOCATION)
refresh_datafiles()


# logger.debug("Read config file")
# ecg_config = ECGConfig(config_block)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def new_cfg(database, datafile, sig):
    db = databases[database]
    if db is None:
        raise NotFound()

    fullpath = db['datafiles'][datafile]
    return ECGConfig(None, state={
        'sig_name': int(sig),
        'file_name': fullpath,
        'data_type': 'physionet',
    })


swagger_ui_blueprint = get_swaggerui_blueprint(
    '/swagger',
    "/openapi.json",
    config={
        'app_name': 'Processing API'
    }
)

app.register_blueprint(swagger_ui_blueprint, url_prefix='/swagger')


# app.register_blueprint(blueprint_power)

@app.route('/')
def health():
    return 'OK'


@app.get('/databases')
@cross_origin()
def get_databases():
    return jsonify([{
        'id': db_id,
        'display_name': db['display_name'],
        'files': db['datafiles'],
    } for db_id, db in databases.items()])


v2dbs_cache = dict()


@app.get('/v2/databases')
@cross_origin()
def v2_get_databases():
    key = cache_key('v2_get_databases', 'datafile', 'signal')
    found = v2dbs_cache.get(key)
    if found is not None:
        json_data = json.dumps(found, ignore_nan=True)
        return Response(json_data, mimetype='application/json')
    output = []
    for db_id, db in databases.items():
        fileinfoes = []
        for k in db['datafiles']:
            cfg = new_cfg(db_id, k, 0)
            df = ReadDataFile(cfg)

            def field_or_none(key, idx):
                if len(df.fileds[key]) > idx:
                    return df.fileds[key][idx]

            signals = []
            for sidx, _ in enumerate(df.signals):
                item = {
                    'id': sidx,
                    'name': df.fileds['sig_name'][sidx],
                    'comment': field_or_none('comments', sidx),
                    'units': field_or_none('units', sidx),
                }

                # NOTICE: currently ECG signals supported only
                # Exception: brno-university-of-technology-ecg-signal-database exposes non-ecg-named channels
                if item['name'] != 'ecg' and not db_id.startswith('brno-university-of-technology-ecg-signal-database'):
                    continue

                signals.append(item)
            fileinfoes.append({
                'file_id': k,
                'signals': signals,
            })
        dbinfo = {
            'id': db_id,
            'display_name': db['display_name'],
            'files': fileinfoes,
        }
        output.append(dbinfo)

    v2dbs_cache[key] = output
    return jsonify(output)


@app.get('/databases/<string:database>/data')
def get_database_data_files(database):
    db = databases[database]
    if db is None:
        raise NotFound()

    return [fid for fid in db['datafiles']]


@app.get('/databases/<string:database>/data/<string:datafile>/signals')
def get_database_signals(database, datafile):
    cfg = new_cfg(database, datafile, 0)
    df = ReadDataFile(cfg)

    def field_or_none(key, idx):
        if len(df.fileds[key]) > idx:
            return df.fileds[key][idx]
        return None

    signals = [{
        'id': sidx,
        'name': df.fileds['sig_name'][sidx],
        'comment': field_or_none('comments', sidx),
        'units': field_or_none('units', sidx),
    } for sidx, _ in enumerate(df.signals)]

    return jsonify(signals)


math_cache = dict()


def cache_key(database, datafile, signal):
    return database + "/" + datafile + "/" + str(signal)


@app.get('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>/math-stats')
def get_mat_stats(database, datafile, signal):
    key = cache_key(database, datafile, signal)
    found = math_cache.get(key)
    if found is not None:
        json_data = json.dumps(found, ignore_nan=True)
        return Response(json_data, mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    df = rhythm.get_ecg_dataframe(signal)
    data = DataPreparation(cfg, df)
    statistics = MathematicalStatistics(data.getPreparedData())

    # path = f'{ecg_config.getImgPath()}/All Mean/CSV/Arrhythmia Mathematical Expectation.csv'
    # path = f'{ecg_config.getImgPath()}/All Mean/CSV/Healthy Mathematical Expectation.csv'
    # df = pd.read_csv(path)
    # no_mean = df["Data"]
    # statistics.setNoVariance(no_mean)

    stats = PlotStatistics(statistics, data.getModSamplingRate(), cfg,
                           data.getPreparedData()).get_math_stats_points()

    math_cache[key] = stats
    json_data = json.dumps(stats, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


intervals_cache = dict()


@app.get('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>/intervals')
def get_intervals(database, datafile, signal):
    key = cache_key(database, datafile, signal)
    found = intervals_cache.get(key)
    if found is not None:
        return Response(json.dumps(found, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    data = rhythm.get_ecg_points(signal)

    intervals_cache[key] = data
    return Response(json.dumps(data, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')


@app.get('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>')
def get_signal_data(database, datafile, signal):
    key = cache_key(database, datafile, signal)
    found = intervals_cache.get(key)
    if found is not None:
        return Response(json.dumps(found, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    data = rhythm.get_signal_data(signal)

    intervals_cache[key] = data
    return Response(json.dumps(data, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')


@app.get('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>/raw_values')
def get_signal_raw_data(database, datafile, signal):
    key = cache_key(database, datafile, signal)
    found = intervals_cache.get(key)
    if found is not None:
        return Response(json.dumps(found, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    data = rhythm.get_signal_raw_data(signal)

    intervals_cache[key] = data
    return Response(json.dumps(data, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')


@app.post('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>/modelled')
def get_signal_modelled(database, datafile, signal):
    # key = cache_key(database, datafile, signal)
    # found = intervals_cache.get(key)
    # if found is not None:
    #     return Response(json.dumps(found, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    data = rhythm.get_signal_raw_data(signal)

    count = 0
    exact_placement = False
    segments = list()
    body = request.get_json()
    if body is not None and 'artifacts' in body:
        artifacts = body['artifacts']
        if artifacts is not None:
            if 'count' in artifacts and 'segments' in artifacts:
                count = int(artifacts['count'])
                segments = artifacts['segments']

    cfg = ArtifactsConfig(count, segments)

    s = Simulation()
    generated, meta = s.gen_ecg_from_prototype(data, 500, cfg)

    # intervals_cache[key] = data
    result = {
        "signal": generated,
        "meta": meta,
    }
    json_data = json.dumps(result, ignore_nan=True, cls=NumpyEncoder)
    return Response(json_data, mimetype='application/json')


rhythm_cache = dict()


@app.get('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>/rhythm')
def get_rhythm(database, datafile, signal):
    key = cache_key(database, datafile, signal)
    found = rhythm_cache.get(key)
    if found is not None:
        json_data = json.dumps(found, ignore_nan=True)
        return Response(json_data, mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    data = rhythm.get_rhythm_points(signal)
    rhythm_cache[key] = data
    math_cache.pop(key, None)
    json_data = json.dumps(data, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


def to_np_array_on_demand(array):
    if len(array) == 2:
        return array

    return to_np_array(array)


@app.post('/math-stats')
def generate_mat_stats():
    rawSignal = request.get_json()

    input = rawSignal
    if len(rawSignal) > 2 and len(rawSignal[0]) == 2:
        input = np.transpose(rawSignal).tolist()[1]

    prepared = PreparedSignal(input, 500)
    # cfg = new_cfg(database, datafile, signal)
    # statistics = MathematicalStatistics(input)
    #
    # stats = PlotStatistics(statistics, 500, None,
    #                        input).get_math_stats_points()

    stats = prepared.get_stats()

    json_data = json.dumps(stats, ignore_nan=True)
    return Response(json_data, mimetype='application/json')

@app.post('/classify')
def classify():
    rawSignal = request.get_json()

    input = rawSignal
    if len(rawSignal) > 2 and len(rawSignal[0]) == 2:
        input = np.transpose(rawSignal).tolist()[1]

    prepared = PreparedSignal(input, 500)
    test_data, _ = prepared.get_interpolated_matrix()
    testing = EnsembleSignalClassifier()
    result = testing.classifySignal(
        model_paths=None,
        test_data=test_data
    )
    # cfg = new_cfg(database, datafile, signal)
    # statistics = MathematicalStatistics(input)
    #
    # stats = PlotStatistics(statistics, 500, None,
    #                        input).get_math_stats_points()

    json_data = json.dumps({
        "class": result,
    }, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


@app.post('/rhythm')
def compute_rhythm_endpoint():
    raw_signal = request.get_json()

    input_signal = raw_signal
    if len(raw_signal) > 2 and len(raw_signal[0]) == 2:
        input_signal = np.transpose(raw_signal).tolist()[1]

    sampling_rate = 500
    prepared = PreparedSignal(input_signal, sampling_rate)

    def to_valid(peaks):
        return [float(v) for v in peaks if not np.isnan(float(v))]

    peak_series = [
        to_valid(prepared.ECG_R_Peaks),
        to_valid(prepared.ECG_P_Peaks),
        to_valid(prepared.ECG_T_Peaks),
        to_valid(prepared.ECG_Q_Peaks),
        to_valid(prepared.ECG_S_Peaks),
    ]
    n_intervals = min(len(p) for p in peak_series) - 1

    result = []
    beat_idx = 1
    for i in range(n_intervals):
        for peaks in peak_series:
            result.append([beat_idx, round(peaks[i + 1] - peaks[i], 4)])
            beat_idx += 1

    result = [[round(x / 6, 6), y] for x, y in result]

    json_data = json.dumps(result, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


@app.post('/detect-segments')
def detect_segments_endpoint():
    raw_signal = request.get_json()

    input_signal = raw_signal
    if len(raw_signal) > 2 and len(raw_signal[0]) == 2:
        input_signal = np.transpose(raw_signal).tolist()[1]

    sampling_rate = 500
    prepared = PreparedSignal(input_signal, sampling_rate)

    def to_val(v):
        return None if np.isnan(float(v)) else round(float(v), 4)

    beats = []
    skipped_count = 0
    for i in range(len(prepared.ECG_R_Peaks)):
        p_on  = to_val(prepared.ECG_P_Onsets.iloc[i])
        p_off = to_val(prepared.ECG_P_Offsets.iloc[i])
        q     = to_val(prepared.ECG_Q_Peaks.iloc[i])
        r     = to_val(prepared.ECG_R_Peaks.iloc[i])
        s     = to_val(prepared.ECG_S_Peaks.iloc[i])
        t_on  = to_val(prepared.ECG_T_Onsets.iloc[i])
        t_off = to_val(prepared.ECG_T_Offsets.iloc[i])

        missing = [name for name, val in {"p_on": p_on, "p_off": p_off, "q": q, "r": r, "s": s, "t_on": t_on, "t_off": t_off}.items() if val is None]
        skipped = len(missing) > 0
        if skipped:
            skipped_count += 1

        beats.append({
            "p_on": p_on, "p_off": p_off,
            "q": q, "r": r, "s": s,
            "t_on": t_on, "t_off": t_off,
            "skipped": skipped,
        })

    app.logger.info(f"Segment detection complete — {len(beats)} beats, {skipped_count} skipped")

    result = {
        "beats": beats,
        "r_peaks": [to_val(prepared.ECG_R_Peaks.iloc[i]) for i in range(len(prepared.ECG_R_Peaks))],
    }

    json_data = json.dumps(result, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


@app.get('/databases/<string:database>/data/<string:datafile>/signals/<int:signal>/segments')
def get_signal_segments(database, datafile, signal):
    cfg = new_cfg(database, datafile, signal)
    record_path = cfg.getFileName()
    sampling_rate = 500

    def peaks_to_list(indices, sr):
        return [round(float(i) / sr, 4) for i in indices]

    # Try WFDB annotations first (many PhysioNet records ship with .atr files)
    ann_extensions = ['atr', 'qrs', 'beat']
    for ext in ann_extensions:
        ann_path = record_path + '.' + ext
        if Path(ann_path).exists():
            try:
                ann = wfdb.rdann(record_path, ext)
                r_indices = ann.sample[np.isin(ann.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'])]
                result = {
                    "r_peaks": peaks_to_list(r_indices, ann.fs or sampling_rate),
                    "p_peaks": [],
                    "t_peaks": [],
                    "q_peaks": [],
                    "s_peaks": [],
                    "source": "wfdb_annotation",
                }
                return Response(json.dumps(result, ignore_nan=True), mimetype='application/json')
            except Exception:
                pass

    # Fall back to algorithmic detection on the cleaned signal
    df = ReadDataFile(cfg)
    raw = np.array(df.signals[signal])
    import neurokit2 as nk
    cleaned = nk.ecg_clean(raw, sampling_rate=sampling_rate)
    r_indices = wfdb.processing.xqrs_detect(cleaned, fs=sampling_rate, verbose=False)
    _, waves = nk.ecg_delineate(cleaned, {"ECG_R_Peaks": r_indices}, sampling_rate=sampling_rate)

    def wave_to_list(key):
        arr = np.array(waves.get(key, []), dtype=float)
        return [None if np.isnan(v) else round(float(v) / sampling_rate, 4) for v in arr]

    result = {
        "r_peaks": peaks_to_list(r_indices, sampling_rate),
        "p_peaks": wave_to_list("ECG_P_Peaks"),
        "t_peaks": wave_to_list("ECG_T_Peaks"),
        "q_peaks": wave_to_list("ECG_Q_Peaks"),
        "s_peaks": wave_to_list("ECG_S_Peaks"),
        "source": "xqrs",
    }
    return Response(json.dumps(result, ignore_nan=True), mimetype='application/json')


def _extract_signal(raw):
    if isinstance(raw, dict):
        signal_data = raw.get('signal', raw)
    else:
        signal_data = raw
    if len(signal_data) > 2 and len(signal_data[0]) == 2:
        return np.transpose(signal_data).tolist()[1]
    return signal_data


def _type_result(flags, r_peaks_arr, n_beats):
    time_ranges = []
    for idx in flags:
        start = float(r_peaks_arr[idx - 1]) if idx > 0 else 0.0
        end = float(r_peaks_arr[idx + 1]) if idx < n_beats - 1 else float(r_peaks_arr[-1])
        time_ranges.append([start, end])
    return {
        "flags": flags,
        "artifact_time_ranges": time_ranges,
        "total_beats": n_beats,
        "artifact_ratio": round(len(flags) / n_beats, 4) if n_beats > 0 else 0.0,
    }


@app.post('/detect-artifacts')
def detect_artifacts_endpoint():
    raw_signal = request.get_json()

    input_signal = raw_signal
    if len(raw_signal) > 2 and len(raw_signal[0]) == 2:
        input_signal = np.transpose(raw_signal).tolist()[1]

    sampling_rate = 500
    prepared = PreparedSignal(input_signal, sampling_rate)

    r_peak_indices = prepared.rpeaks["ECG_R_Peaks"]
    r_peaks_sec = (np.array(r_peak_indices) / sampling_rate).tolist()
    r_amplitudes = [float(input_signal[int(idx)]) for idx in r_peak_indices]

    segment_result = detect_artifacts(
        r_peaks_sec,
        r_amplitudes,
        [prepared.matrix_P_wave, prepared.matrix_QRS, prepared.matrix_T_wave],
    )

    zone_per_zone, zone_flagged = detect_by_zone_intervals(prepared.zone_intervals)

    all_flagged = sorted(set(segment_result["artifact_beats"]) | set(zone_flagged))
    n_beats = segment_result["total_beats"]
    r_peaks_arr = np.array(r_peaks_sec)
    time_ranges = []
    for beat_idx in all_flagged:
        start = float(r_peaks_arr[beat_idx - 1]) if beat_idx > 0 else 0.0
        end = float(r_peaks_arr[beat_idx + 1]) if beat_idx < n_beats - 1 else float(r_peaks_arr[-1])
        time_ranges.append([start, end])

    result = {
        "artifact_beats": all_flagged,
        "rhythm_flags": segment_result["rhythm_flags"],
        "segment_flags": segment_result["segment_flags"],
        "amplitude_flags": segment_result["amplitude_flags"],
        "zone_flags": {k: v["flagged"] for k, v in zone_per_zone.items() if v["flagged"]},
        "zone_details": {k: v for k, v in zone_per_zone.items() if v["flagged"]},
        "artifact_time_ranges": time_ranges,
        "total_beats": n_beats,
        "artifact_ratio": round(len(all_flagged) / n_beats, 4) if n_beats > 0 else 0.0,
    }

    json_data = json.dumps(result, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


@app.post('/detect-artifacts/rhythm')
def detect_artifacts_rhythm():
    body = request.get_json()
    sigma = float(body.get('sigma', 3.0)) if isinstance(body, dict) else 3.0
    input_signal = _extract_signal(body)

    sampling_rate = 500
    prepared = PreparedSignal(input_signal, sampling_rate)
    r_peaks_sec = (np.array(prepared.rpeaks["ECG_R_Peaks"]) / sampling_rate).tolist()
    r_arr = np.array(r_peaks_sec)
    n_beats = len(r_peaks_sec)

    rr = np.diff(r_arr)
    z, median_rr, mad_rr = mad_stats(rr)
    flags = [int(i + 1) for i in range(len(z)) if z[i] > sigma]

    scale = 1.4826 * mad_rr + 1e-9
    beat_details = [
        {
            "beat_idx": int(i + 1),
            "rr_interval": round(float(rr[i]), 4),
            "prev_rr": round(float(rr[i - 1]), 4) if i > 0 else None,
            "next_rr": round(float(rr[i + 1]), 4) if i < len(rr) - 1 else None,
            "deviation": round(float(abs(rr[i] - median_rr)), 4),
            "scale": round(float(scale), 4),
            "z_score": round(float(z[i]), 2),
            "median_rr": round(median_rr, 4),
            "reason": "RR interval outlier",
        }
        for i in range(len(z)) if z[i] > sigma
    ]

    # Rhythm anomaly = the interval [i-1, i], so highlight exactly those two peaks.
    rhythm_time_ranges = []
    for idx in flags:
        start = float(r_arr[idx - 1]) if idx > 0 else 0.0
        end = float(r_arr[idx])
        rhythm_time_ranges.append([start, end])
    result = {
        "flags": flags,
        "artifact_time_ranges": rhythm_time_ranges,
        "total_beats": n_beats,
        "artifact_ratio": round(len(flags) / n_beats, 4) if n_beats > 0 else 0.0,
    }
    result["beat_details"] = beat_details
    result["stats"] = {
        "median_rr": round(median_rr, 4),
        "mad": round(mad_rr, 4),
        "scale": round(float(scale), 4),
        "sigma": sigma,
        "n_intervals": int(len(rr)),
    }
    app.logger.info(f"Rhythm artifact detection: {len(flags)}/{n_beats} beats flagged (sigma={sigma})")
    return Response(json.dumps(result, ignore_nan=True), mimetype='application/json')


@app.post('/detect-artifacts/segment')
def detect_artifacts_segment():
    from my_helpers.artifact_detection import detect_by_median_segment
    body = request.get_json()
    sigma = float(body.get('sigma', 2.5)) if isinstance(body, dict) else 2.5
    trim_ratio = float(body.get('trim_ratio', 0.3)) if isinstance(body, dict) else 0.3
    input_signal = _extract_signal(body)

    sampling_rate = 500
    prepared = PreparedSignal(input_signal, sampling_rate)
    r_peaks_sec = (np.array(prepared.rpeaks["ECG_R_Peaks"]) / sampling_rate).tolist()
    r_arr = np.array(r_peaks_sec)
    n_beats = len(r_peaks_sec)

    wave_labels = ["P", "QRS", "T"]
    wave_matrices = [prepared.matrix_P_wave, prepared.matrix_QRS, prepared.matrix_T_wave]

    # per-wave RMSE and z-scores for diagnostics
    wave_z: dict[str, np.ndarray] = {}
    wave_rmse: dict[str, np.ndarray] = {}
    wave_stats: dict[str, dict] = {}
    seg_mask = np.zeros(n_beats, dtype=bool)

    for label, matrix in zip(wave_labels, wave_matrices):
        if len(matrix) == 0:
            continue
        min_len = min(len(row) for row in matrix)
        mat = np.array([row[:min_len] for row in matrix], dtype=float)
        initial_template = np.median(mat, axis=0)
        initial_rmse = np.sqrt(np.mean((mat - initial_template) ** 2, axis=1))
        n_keep = max(1, int(len(mat) * (1 - trim_ratio)))
        keep_idx = np.argsort(initial_rmse)[:n_keep]
        robust_template = np.median(mat[keep_idx], axis=0)
        rmse = np.sqrt(np.mean((mat - robust_template) ** 2, axis=1))
        z, median_rmse, mad_rmse = mad_stats(rmse)

        wave_z[label] = z
        wave_rmse[label] = rmse
        wave_stats[label] = {
            "median_rmse": round(float(median_rmse), 4),
            "mad": round(float(mad_rmse), 4),
            "n_beats": len(matrix),
        }

        m = z > sigma
        length = min(len(m), n_beats)
        seg_mask[:length] |= m[:length]

    flags = np.where(seg_mask)[0].tolist()

    # per-beat details: which waves flagged and their z-scores
    beat_details = []
    for beat_idx in flags:
        waves_flagged = []
        z_scores = {}
        rmse_values = {}
        for label in wave_labels:
            if label not in wave_z or beat_idx >= len(wave_z[label]):
                continue
            if wave_z[label][beat_idx] > sigma:
                waves_flagged.append(label)
                z_scores[label] = round(float(wave_z[label][beat_idx]), 2)
                rmse_values[label] = round(float(wave_rmse[label][beat_idx]), 4)
        beat_details.append({
            "beat_idx": beat_idx,
            "waves_flagged": waves_flagged,
            "z_scores": z_scores,
            "rmse_values": rmse_values,
            "reason": "Beat shape deviates from template",
        })

    result = _type_result(flags, r_arr, n_beats)
    result["beat_details"] = beat_details
    result["stats"] = {
        "sigma": sigma,
        "trim_ratio": trim_ratio,
        "wave_stats": wave_stats,
    }
    app.logger.info(f"Segment artifact detection: {len(flags)}/{n_beats} beats flagged (sigma={sigma}, trim={trim_ratio})")
    return Response(json.dumps(result, ignore_nan=True), mimetype='application/json')


@app.post('/detect-artifacts/amplitude')
def detect_artifacts_amplitude():
    body = request.get_json()
    sigma = float(body.get('sigma', 3.0)) if isinstance(body, dict) else 3.0
    input_signal = _extract_signal(body)

    sampling_rate = 500
    prepared = PreparedSignal(input_signal, sampling_rate)
    r_peak_indices = prepared.rpeaks["ECG_R_Peaks"]
    r_peaks_sec = (np.array(r_peak_indices) / sampling_rate).tolist()
    r_arr = np.array(r_peaks_sec)
    n_beats = len(r_peaks_sec)

    r_amplitudes = np.array([float(input_signal[int(idx)]) for idx in r_peak_indices])
    z, median_amp, mad_amp = mad_stats(r_amplitudes)
    flags = [int(i) for i in range(len(z)) if z[i] > sigma]

    scale_amp = 1.4826 * mad_amp + 1e-9
    beat_details = [
        {
            "beat_idx": int(i),
            "amplitude": round(float(r_amplitudes[i]), 4),
            "deviation": round(float(abs(r_amplitudes[i] - median_amp)), 4),
            "scale": round(float(scale_amp), 4),
            "z_score": round(float(z[i]), 2),
            "median_amplitude": round(median_amp, 4),
            "reason": "R-peak amplitude outlier",
        }
        for i in range(len(z)) if z[i] > sigma
    ]

    result = _type_result(flags, r_arr, n_beats)
    result["beat_details"] = beat_details
    result["stats"] = {
        "median_amplitude": round(median_amp, 4),
        "mad": round(mad_amp, 4),
        "scale": round(float(scale_amp), 4),
        "sigma": sigma,
    }
    app.logger.info(f"Amplitude artifact detection: {len(flags)}/{n_beats} beats flagged (sigma={sigma})")
    return Response(json.dumps(result, ignore_nan=True), mimetype='application/json')


@app.post('/proc/<string:op>')
def apply_processing_fn(op):
    raw_signal = request.get_json()

    src_t = np.transpose(raw_signal).tolist()[0]
    src_x = np.transpose(raw_signal).tolist()[1]

    ops = {
        'detrend': lambda sig: (apply_detrend(sig), src_t),
        'butter_bandpass': lambda sig: (butter_bandpass(sig, 500), src_t),
        'decimate': lambda sig: defaults_decimate_signal(sig),
        'preprocess_pipeline': lambda sig: defaults_process_ecg_pipeline(sig),
    }

    processor = ops[op]

    x, t = processor(src_x)

    result = np.transpose([t, x]).tolist()

    json_data = json.dumps(result, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


@app.post('/modelling/ecg')
def simulate_ecg():
    body = request.get_json()
    variance_data = to_np_array_on_demand(body['variance'])
    mean_data = to_np_array_on_demand(body['mean'])
    rhythm_data = to_np_array_on_demand(body['rhythm'])
    cycles_count = body['count']

    sim = Simulation()
    cycle_data = sim.gen_cycle(rhythm_data, variance_data, mean_data, cycles_count)
    json_data = json.dumps(cycle_data, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


@app.post('/v2/modelling/math_stats')
def simulate_ecg_from_math_stats():
    body = request.get_json()
    count = 0
    segments_count = 5
    segments = list()
    if body is not None and 'artifacts' in body:
        artifacts = body['artifacts']
        if artifacts is not None:
            if 'count' in artifacts:
                count = int(artifacts['count'])
            if 'segments' in artifacts:
                segments = artifacts['segments']

    cfg = ArtifactsConfig(count, segments)

    variance_data = to_np_array_on_demand(body['variance'])
    mean_data = to_np_array_on_demand(body['mean'])
    rhythm_data = to_np_array_on_demand(body['rhythm'])

    sim = Simulation()
    generated, meta = sim.gen_ecg_from_math_stats(segments_count, mean_data, variance_data, rhythm_data, cfg)

    # intervals_cache[key] = data
    result = {
        "signal": generated,
        "meta": meta,
    }
    json_data = json.dumps(result, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


def bandpass(data, fs):
    m = np.mean(data)
    data = (data - m)
    t = 1
    if np.max(data) > 1000:
        t = 1000.0
    res = data / t
    # return data
    return res


if __name__ == '__main__':
    # cfg = new_cfg("pulse-transit-time-ppg.1.1.0", "s1_sit", 0)
    # rhythm = GenerateRhythmFunction(cfg)
    # # ecg_points = rhythm.get_ecg_points(0)
    # dispersion_points = rhythm.get_dispersion_points(0)
    #
    # signals, fileds = wfdb.rdsamp(cfg.getFileName())
    #
    # sampling_rate = fileds['fs']
    # signals = signals.transpose()
    #
    # bandpass_notch_channels = []
    # for i in signals:
    #     bandpass_notch_channels.append(bandpass(i, fs=sampling_rate))
    #
    # signals = bandpass_notch_channels
    # target = signals[0]
    #
    # cleaned = nk.ecg_clean(target, sampling_rate)
    # processed, data = nk.ecg_process(cleaned, sampling_rate)
    # peaksSignals, peaksInfo = nk.ecg_peaks(cleaned, sampling_rate)
    # nk.ecg_plot(processed, data)
    # fig = plt.gcf()
    # fig.set_size_inches(10, 12, forward=True)
    # fig.savefig("ecg.png")
    #
    # disp = np.var(processed, axis=0)
    #
    #
    # # qrs_epochs = nk.ecg_segment(cleaned, rpeaks=None, sampling_rate=sampling_rate, show=True)
    # # first_epoch = qrs_epochs["1"]
    # # points = list()
    # # for ix in range(first_epoch.size):
    # #     points.append(first_epoch.at(ix))
    # fig = plt.gcf()
    # fig.set_size_inches(10, 12, forward=True)
    # fig.savefig("segment.png")

    app.run("0.0.0.0", os.environ.get('APP_PORT', '5100'))
