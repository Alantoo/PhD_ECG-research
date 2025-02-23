import sys
from pathlib import Path

from werkzeug.exceptions import NotFound

from flask import Flask, jsonify, Response, abort
from get_config.ecg_config import ECGConfig
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.plot_statistics import PlotStatistics
from my_helpers.data_preparation import DataPreparation
from my_helpers.fourier_series import FourierSeries
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.classifiers import Classifiers
from my_helpers.read_data.read_data_file import ReadDataFile
from my_helpers.test_classifiers import TestClassifiers
from my_helpers.plot_classifier import PlotClassifier
from my_helpers.test_diff_fr import TestDiffFr
import glob
from flask_cors import CORS, cross_origin
import simplejson as json
import wfdb
import neurokit2 as nk

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import sawtooth
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
import pywt
from scipy.signal import find_peaks, savgol_filter
import emd
import statsmodels.api as sm

from simulation import Simulation


def refresh_datafiles():
    for db_id, db in databases.items():
        files = [Path(f).stem for f in glob.glob(db['path'] + "/*.dat")]
        file_map = dict()
        for file in files:
            file_map[file] = db['path'] + "/" + file

        db['datafiles'] = file_map

databases = {
    'pulse-transit-time-ppg.1.1.0': {
        'id': 'pulse-transit-time-ppg.1.1.0',
        'display_name': 'pulse-transit-time-ppg 1.1.0',
        'path': '/Users/alantoo/Workspace/Edu/ecg_database/physionet.org/files/pulse-transit-time-ppg/1.1.0',
        'datafiles': dict()
    }
}

refresh_datafiles()

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


def replace_nan(val):
    if np.isnan(val) or val is None:
        return 0
    return val

def to_np_array(points):
    time = list()
    values = list()
    min_val = sys.maxsize
    for point in points:
        if point[1] is None or np.isnan(point[1]):
            point[1] = min_val

        time.append(replace_nan(point[0]))
        val = replace_nan(point[1])
        min_val = min(min_val, val)
        values.append(val)
    return np.array([time, values])


def show_plot(title, t_data, sig_data):
    # Візуалізація
    plt.plot(t_data, sig_data, label="Змодельований сигнал")
    plt.xlabel("Час")
    plt.ylabel("Сигнал")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':

    database = 'pulse-transit-time-ppg.1.1.0'
    datafile = 's3_sit'
    signal = 0

    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    df = rhythm.get_ecg_dataframe(signal)
    data = DataPreparation(cfg, df)
    statistics = MathematicalStatistics(data.getPreparedData())
    rhythmData = rhythm.get_rhythm_points(signal)
    # rhythmData = rhythm.to_data_points(statistics.rhythm[0]) # rhythm.get_rhythm_points(signal)

    stats = PlotStatistics(statistics, data.getModSamplingRate(), cfg,
                           data.getPreparedData()).get_math_stats_points()

    # Вхідні двовимірні масиви (X - значення, Y - час)
    rhythm_data = to_np_array(rhythmData) # np.array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]])  # Функція ритму
    mean_data = to_np_array(stats["mathematical_expectation"])  # np.array([[0, 1, 2, 3, 4], [1, 1, 0, -1, -1]])  # Математичне сподівання
    variance_data = to_np_array(stats["variance"])  # np.array([[0, 1, 2, 3, 4], [0.1, 0.2, 0.3, 0.2, 0.1]])  # Дисперсія

    sim = Simulation()
    cycle = sim.gen_cycle(rhythm_data, variance_data, mean_data, 3)
    generated = to_np_array(cycle)
    #
    # # Інтерполяція для приведення до спільного часу
    # t = np.linspace(0, 1, 500)  # Спільний часовий інтервал
    #
    #
    # # peaks, _ = find_peaks(data.getPreparedData())
    # # rr_intervals = np.diff(peaks) / data.getModSamplingRate()  # Часові інтервали (секунди)
    #
    #
    #
    # rhythm_interp = interp1d(rhythm_data[0], rhythm_data[1], kind='cubic', fill_value="extrapolate")
    # mean_interp = interp1d(mean_data[0], mean_data[1], kind='cubic', fill_value="extrapolate")
    # variance_interp = interp1d(variance_data[0], variance_data[1], kind='cubic', fill_value="extrapolate")
    #
    # # Отримуємо значення на всьому часовому інтервалі
    # rhythm = rhythm_interp(t)
    # mean = mean_interp(t)
    # variance = variance_interp(t)
    # # rhythm = rhythm_data[1]
    # # mean = mean_data[1]
    # # variance = variance_data[1]
    # # def moving_average(s, window_size=10):
    # #     return np.convolve(s, np.ones(window_size)/window_size, mode='same')
    #
    # def wavelet_denoising(s):
    #     coeffs = pywt.wavedec(s, 'db4', level=4)
    #     coeffs[1:] = [pywt.threshold(c, np.std(c) / 2, mode='soft') for c in coeffs[1:]]
    #     return pywt.waverec(coeffs, 'db4')
    #
    # # def emd_denoising(s):
    # #     imf = emd.sift.sift(s)  # Розкладання на моди
    # #     return np.sum(imf[:-1], axis=0)  # Видаляємо найшумнішу моду
    # # def lowess_denoising(s):
    # #     frac = 0.05  # Відсоток точки, який використовується для локального згладжування
    # #     return sm.nonparametric.lowess(s, t, frac=frac, it=3)[:, 1]
    #
    # # Генерація циклічного сигналу
    # ecg_signal = mean + np.sin(2 * np.pi * rhythm * t) + np.random.normal(0, np.sqrt(variance), len(t))
    # # ecg_signal = mean + sawtooth(2 * np.pi * rhythm * t, 0.3) + np.random.normal(0, np.sqrt(variance), len(t))
    #
    #
    # # # Застосування фільтру Savitzky-Golay для згладжування сигналу
    # # smoothed_signal = savgol_filter(ecg_signal, window_length=51, polyorder=3)
    # #
    # # # Нормалізація результату, щоб він залишався в межах функцій математичного сподівання
    # # smoothed_signal = (smoothed_signal - np.min(smoothed_signal)) / (np.max(smoothed_signal) - np.min(smoothed_signal))  # Нормалізація
    # # smoothed_signal = smoothed_signal * (np.max(mean) - np.min(mean)) + np.min(mean)  # Повертаємо до масштабу mean
    #



    show_plot("Rhythm", rhythm_data[0], rhythm_data[1])
    # show_plot("Rhythm 2", t, rhythm)
    show_plot("Mean", mean_data[0], mean_data[1])
    # show_plot("Mean 2", t, mean)
    show_plot("Variance", variance_data[0], variance_data[1])
    # show_plot("Variance 2", t, variance)
    # show_plot("Циклічний сигнал з урахуванням ритму, сподівання та дисперсії", t, ecg_signal)
    # show_plot("savgol_filter", t, savgol_filter(ecg_signal, window_length=13, polyorder=3))
    show_plot("generated", generated[0], generated[1])
    # show_plot("moving_average", t, moving_average(ecg_signal, window_size=5))
    # show_plot("gaussian_filter1d", t, gaussian_filter1d(ecg_signal, sigma=1))
    # show_plot("wavelet_denoising", t, wavelet_denoising(ecg_signal))
    # show_plot("medfilt", t, medfilt(ecg_signal))
    # show_plot("lowess_denoising", t, lowess_denoising(ecg_signal))
    # show_plot("savgol_filter(medfilt(", t, savgol_filter(medfilt(ecg_signal, 5), 21, 3))
