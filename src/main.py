import glob
import os
from pathlib import Path

import numpy as np
import simplejson as json
from flask import Flask, jsonify, Response, request
from flask_cors import CORS, cross_origin
from werkzeug.exceptions import NotFound

from gen_sig import to_np_array
from get_config.ecg_config import ECGConfig
from my_helpers.data_preparation import DataPreparation
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.plot_statistics import PlotStatistics
from my_helpers.read_data.read_data_file import ReadDataFile
from simulation import Simulation
import dotenv
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

app = Flask(__name__)
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
                if item['name'] != 'ecg':
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


@app.get('/databases/<database>/data')
def get_database_data_files(database):
    db = databases[database]
    if db is None:
        raise NotFound()

    return [fid for fid in db['datafiles']]


@app.get('/databases/<database>/data/<datafile>/signals')
def get_database_signals(database, datafile):
    cfg = new_cfg(database, datafile, 0)
    df = ReadDataFile(cfg)

    def field_or_none(key, idx):
        if len(df.fileds[key]) > idx:
            return df.fileds[key][idx]

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


@app.get('/databases/<database>/data/<datafile>/signals/<int:signal>/math-stats')
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


@app.get('/databases/<database>/data/<datafile>/signals/<int:signal>/intervals')
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

@app.get('/databases/<database>/data/<datafile>/signals/<int:signal>')
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

@app.get('/databases/<database>/data/<datafile>/signals/<int:signal>/raw_values')
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

@app.get('/databases/<database>/data/<datafile>/signals/<int:signal>/modelled')
def get_signal_modelled(database, datafile, signal):
    # key = cache_key(database, datafile, signal)
    # found = intervals_cache.get(key)
    # if found is not None:
    #     return Response(json.dumps(found, ignore_nan=True, cls=NumpyEncoder), mimetype='application/json')

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    data = rhythm.get_signal_raw_data(signal)
    s = Simulation()
    generated, meta = s.gen_ecg_from_prototype(data, 500)

    # intervals_cache[key] = data
    result = {
        "signal": generated,
        "meta": meta,
    }
    json_data = json.dumps(result, ignore_nan=True, cls=NumpyEncoder)
    return Response(json_data, mimetype='application/json')


rhythm_cache = dict()


@app.get('/databases/<database>/data/<datafile>/signals/<int:signal>/rhythm')
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
    json_data = json.dumps(data, ignore_nan=True)
    return Response(json_data, mimetype='application/json')


def to_np_array_on_demand(array):
    if len(array) == 2:
        return array

    return to_np_array(array)

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

    app.run("0.0.0.0", os.environ.get('APP_PORT', '5000'))
