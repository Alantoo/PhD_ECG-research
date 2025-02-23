import glob
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


def refresh_datafiles():
    for db_id, db in databases.items():
        files = [Path(f).stem for f in glob.glob(db['path'] + "/*.dat")]
        file_map = dict()
        for file in files:
            file_map[file] = db['path'] + "/" + file

        db['datafiles'] = file_map


app = Flask(__name__)
cors = CORS(app)
# config_block = 'H_P001_PPG_S_S1'

databases = {
    'pulse-transit-time-ppg.1.1.0': {
        'id': 'pulse-transit-time-ppg.1.1.0',
        'display_name': 'pulse-transit-time-ppg 1.1.0',
        'path': '/Users/alantoo/Workspace/Edu/ecg_database/physionet.org/files/pulse-transit-time-ppg/1.1.0',
        'datafiles': dict()
    }
}
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
    } for db_id, db in databases.items()])


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

    app.run()

# from authentication.authentication import Authentication
# from authentication.ft_authentication import Authentication as FTAuthentication
# from authentication.hurst_index import HurstIndex
# from classifiers_test.test import Test
# from classifiers_test.test_src import Test as Test_src
# from loguru import logger
# import pandas as pd
# from get_config.ecg_config import ECGConfig
# from my_helpers.generate_rhythm_function import GenerateRhythmFunction
# from my_helpers.plot_statistics import PlotStatistics
# from my_helpers.data_preparation import DataPreparation
# from my_helpers.fourier_series import FourierSeries
# from my_helpers.mathematical_statistics import MathematicalStatistics
# from my_helpers.classifiers import Classifiers
# from my_helpers.test_classifiers import TestClassifiers
# from my_helpers.plot_classifier import PlotClassifier
# from my_helpers.test_diff_fr import TestDiffFr
# from napolitano.napolitano import Napolitano
# import sys
# import argparse
# import os
# import time
#
# from napolitano.plot_classifiers import PlotClassifiers
# from classifiers_test.plot_classifiers import PlotClassifiers as PlotClassifiers2
# from no_classifires_test.no_classifires import NoClassidire
#
# if __name__ == '__main__':
#
#     logger.info("ECG Research")
#     # time.sleep(10)
#     parser = argparse.ArgumentParser()
#     parser.add_argument('action', choices=('qq-plot','get-hurst-index', 'plot-n', 'plot-n-ft', 'authentication-diff', 'ft-authentication-diff', 'ft-authentication-test', 'authentication-test', 'no-all-test','no-all-mean', 'no-all-sigma', 'test-classifier-src', 'plot-classifier2', 'test-classifier2', 'plot-classifier', 'napolitano-test','napolitano-plot-red', 'show-intervals', 'test-classifier', 'diff-test', 'plot-ekg', 'get-intervals', 'gen-intervals', 'gen-fr', 'plot-fr', 'main', 'fourier-series-demonstration', 'fourier-series-test-1', 'train_classifier', 'plot_classifier', 'plot-statistics', 'plot-fourier-statistics', 'plot-autocorrelation', 'plot-autocovariation'))
#     parser.add_argument('-c', type=str, required=True)
#     parser.add_argument('-d', type=str)
#     parser.add_argument('-s', type=str)
#     a = parser.parse_args()
#     config_block = a.c
#     logger.debug("Read config file")
#     ecg_config = ECGConfig(config_block)
#     logger.debug(ecg_config)
#     if a.action == "get-hurst-index":
#         HurstIndex(ecg_config)
#     if a.action == "ft-authentication-test":
#         FTAuthentication(ecg_config).Classifiers()
#     if a.action == "authentication-test":
#         Authentication(ecg_config).Classifiers()
#     if a.action == "plot-n-ft":
#         FTAuthentication(ecg_config).Plot_n()
#     if a.action == "plot-n":
#         Authentication(ecg_config).Plot_n()
#     if a.action == "qq-plot":
#         Authentication(ecg_config).QQplot()
#     if a.action == "authentication-diff":
#         Authentication(ecg_config).Diff()
#     if a.action == "ft-authentication-diff":
#         FTAuthentication(ecg_config).Diff()
#     if a.action == "no-all-test":
#         NoClassidire(ecg_config).NoTest()
#     if a.action == "no-all-sigma":
#         NoClassidire(ecg_config).NoAllSigma()
#     if a.action == "no-all-mean":
#         NoClassidire(ecg_config).NoAllMean()
#     if a.action == "test-classifier-src":
#         Test_src(ecg_config).TestRed()
#     if a.action == "plot-classifier2":
#         PlotClassifiers2(ecg_config)
#     if a.action == "test-classifier2":
#         # for i in range(1, 31):
#         #     Test(ecg_config).TestRed(i)
#         Test(ecg_config).TestRed(15)
#     if a.action == "plot-classifier":
#         PlotClassifiers(ecg_config)
#     if a.action == "napolitano-test":
#         # for i in range(1, 201):
#         #     Napolitano(ecg_config).TestRed(i)
#         Napolitano(ecg_config).TestRed(1)
#     if a.action == "napolitano-plot-red":
#         Napolitano(ecg_config).PlotDataRed()
#     if a.action == "diff-test":
#         TestDiffFr(ecg_config).getTestData()
#     if a.action == "show-intervals":
#         GenerateRhythmFunction(ecg_config).showIntervals()
#     if a.action == "gen-intervals":
#         GenerateRhythmFunction(ecg_config).genIntervals()
#     if a.action == "get-intervals":
#         GenerateRhythmFunction(ecg_config).getIntervals()
#     if a.action == "gen-fr":
#         GenerateRhythmFunction(ecg_config).genFr()
#     if a.action == "plot-fr":
#         GenerateRhythmFunction(ecg_config).plotFr()
#     if a.action == "plot-ekg":
#         GenerateRhythmFunction(ecg_config).plotECG()
#     if a.action == "fourier-series-demonstration":
#         data = DataPreparation(ecg_config)
#         FourierSeries(ecg_config, data).getFourierSeriesDemonstration(int(a.s or 0))
#     if a.action == "fourier-series-test-1":
#         data = DataPreparation(ecg_config)
#         # data.plotAllCycles()
#         # FourierSeries(ecg_config, data).getFourierSpectrum([1, 3, 10])
#     if a.action == "plot-statistics":
#         data = DataPreparation(ecg_config)
#         statistics = MathematicalStatistics(data.getPreparedData())
#
#         # path = f'{ecg_config.getImgPath()}/All Mean/CSV/Arrhythmia Mathematical Expectation.csv'
#         # path = f'{ecg_config.getImgPath()}/All Mean/CSV/Healthy Mathematical Expectation.csv'
#         # df = pd.read_csv(path)
#         # no_mean = df["Data"]
#         # statistics.setNoVariance(no_mean)
#
#         PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAllStatistics()
#
#     if a.action == "plot-fourier-statistics":
#         data = DataPreparation(ecg_config)
#         statistics = MathematicalStatistics(data.getPreparedData())
#         PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAllFourierStatistics()
#
#     if a.action == "plot-autocorrelation":
#         data = DataPreparation(ecg_config)
#         statistics = MathematicalStatistics(data.getPreparedData())
#         PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAutocorrelation()
#
#     if a.action == "plot-autocovariation":
#         data = DataPreparation(ecg_config)
#         statistics = MathematicalStatistics(data.getPreparedData())
#         PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAutocovariation()
#
#     if a.action == "plot_classifier":
#         data = DataPreparation(ecg_config)
#         PlotClassifier(ecg_config, data)
#
#     if a.action == "test-classifier":
#         TestClassifiers(ecg_config)
#
#     if a.action == "train_classifier":
#         config_block_2 = a.d or config_block
#         logger.debug("Read config file 2")
#         ecg_config_2 = ECGConfig(config_block_2)
#         logger.debug(ecg_config_2)
#         data = DataPreparation(ecg_config)
#         data_2 = DataPreparation(ecg_config_2)
#         Classifiers(ecg_config, data, ecg_config_2, data_2)
#
#     if a.action == "main":
#         config_block_2 = a.d or config_block
#         logger.debug("Read config file 2")
#         ecg_config_2 = ECGConfig(config_block_2)
#         logger.debug(ecg_config_2)
#         data = DataPreparation(ecg_config)
#         data_2 = DataPreparation(ecg_config_2)
#         statistics = MathematicalStatistics(data.getPreparedData())
#         statistics_2 = MathematicalStatistics(data_2.getPreparedData())
#         statistics.setSamplingRate(data.getModSamplingRate())
#         statistics_2.setSamplingRate(data.getModSamplingRate())
#         statistics_mean = statistics.getMean(statistics_2)
#         statistics_fourier_mean = statistics.getFourierMean(statistics_2)
#
#         print("The average value of the moment function")
#         print("Mathematical expectation")
#         print("%.5f" % statistics_mean.getMathematicalExpectation())
#         print("Initial moments of the second order")
#         print("%.5f" % statistics_mean.getInitialMomentsSecondOrder())
#         print("Initial moments of the third order")
#         print("%.5f" % statistics_mean.getInitialMomentsThirdOrder())
#         print("Initial moments of the fourth order")
#         print("%.5f" % statistics_mean.getInitialMomentsFourthOrder())
#         print("Variance")
#         print("%.5f" % statistics_mean.getVariance())
#         print("Central moment functions of the fourth order")
#         print("%.5f" % statistics_mean.getCentralMomentFunctionsFourthOrder())
#         print()
#         print("The average value of the spectral components of the moment function")
#         print("Mathematical expectation")
#         print("%.5f" % statistics_fourier_mean.getMathematicalExpectation())
#         print("Initial moments of the second order")
#         print("%.5f" % statistics_fourier_mean.getInitialMomentsSecondOrder())
#         print("Initial moments of the third order")
#         print("%.5f" % statistics_fourier_mean.getInitialMomentsThirdOrder())
#         print("Initial moments of the fourth order")
#         print("%.5f" % statistics_fourier_mean.getInitialMomentsFourthOrder())
#         print("Variance")
#         print("%.5f" % statistics_fourier_mean.getVariance())
#         print("Central moment functions of the fourth order")
#         print("%.5f" % statistics_fourier_mean.getCentralMomentFunctionsFourthOrder())
#
#
#
#
#
