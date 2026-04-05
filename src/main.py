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
from my_helpers.artifact_detection import detect_artifacts
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

    def peaks_to_list(series):
        return [to_val(v) for v in series]

    def wave_ranges(onsets, offsets):
        return [
            [to_val(on), to_val(off)]
            for on, off in zip(onsets, offsets)
        ]

    result = {
        "r_peaks":   peaks_to_list(prepared.ECG_R_Peaks),
        "p_waves":   wave_ranges(prepared.ECG_P_Onsets, prepared.ECG_P_Offsets),
        "qrs_waves": wave_ranges(prepared.ECG_Q_Peaks,  prepared.ECG_S_Peaks),
        "t_waves":   wave_ranges(prepared.ECG_T_Onsets, prepared.ECG_T_Offsets),
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

    result = detect_artifacts(
        r_peaks_sec,
        r_amplitudes,
        [prepared.matrix_P_wave, prepared.matrix_QRS, prepared.matrix_T_wave],
    )

    json_data = json.dumps(result, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


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
