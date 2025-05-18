import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.interpolate as interp
import wfdb
from loguru import logger
from werkzeug.exceptions import NotFound

from get_config.ecg_config import ECGConfig
from my_helpers.data_preparation import DataPreparation
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.plot_statistics import PlotStatistics
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

def outdated():
    database = 'pulse-transit-time-ppg.1.1.0'
    datafile = 's3_run'
    signal = 0
    cfg = new_cfg(database, datafile, signal)
    rhythm = GenerateRhythmFunction(cfg)
    df = rhythm.get_ecg_dataframe(signal)
    data = DataPreparation(cfg, df)
    prepared_data = data.getPreparedData()

    # complexes_segmented = list()
    # # for segment in prepared_data:
    # sampling_rate = 100
    # _, rpeaks = nk.ecg_peaks(prepared_data, sampling_rate=sampling_rate)
    # _, waves = nk.ecg_delineate(prepared_data, rpeaks, sampling_rate=sampling_rate)
    # ECG_P_Peaks = list(np.round(np.array(waves["ECG_P_Peaks"]) / sampling_rate, 4))
    # ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / sampling_rate, 4))
    # ECG_R_Peaks = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / sampling_rate, 4))
    # ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / sampling_rate, 4))
    # ECG_T_Peaks = list(np.round(np.array(waves["ECG_T_Peaks"]) / sampling_rate, 4))
    #
    # ecg_fr = pd.DataFrame({"ECG_P_Peaks": ECG_P_Peaks, "ECG_Q_Peaks": ECG_Q_Peaks, "ECG_R_Peaks": ECG_R_Peaks,
    #                        "ECG_S_Peaks": ECG_S_Peaks, "ECG_T_Peaks": ECG_T_Peaks})
    # complexes_segmented.append(rpeaks)

    self_data = prepared_data
    # Mathematical expectation

    statistics = MathematicalStatistics(prepared_data)
    rhythmData = rhythm.get_rhythm_points(signal)
    # rhythmData = rhythm.to_data_points(statistics.rhythm[0]) # rhythm.get_rhythm_points(signal)

    stats = PlotStatistics(statistics, data.getModSamplingRate(), cfg,
                           prepared_data).get_math_stats_points()

    # Вхідні двовимірні масиви (X - значення, Y - час)
    rhythm_data = to_np_array(rhythmData)  # np.array([[0, 1, 2, 3, 4], [0, 2, 4, 6, 8]])  # Функція ритму
    mean_data = to_np_array(
        stats["mathematical_expectation"])  # np.array([[0, 1, 2, 3, 4], [1, 1, 0, -1, -1]])  # Математичне сподівання
    variance_data = to_np_array(
        stats["variance"])  # np.array([[0, 1, 2, 3, 4], [0.1, 0.2, 0.3, 0.2, 0.1]])  # Дисперсія

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

def show_plot(title, t_data, sig_data):
    # Візуалізація
    plt.plot(t_data, sig_data, label="Змодельований сигнал")
    plt.xlabel("Час")
    plt.ylabel("Сигнал")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def bandpass(data, fs):
    m = np.mean(data)
    data = (data - m)
    t = 1
    if np.max(data) > 1000:
        t = 1000.0
    res = data / t
    # return data
    return res

def process_physionet_file():
    def interpolate_matrix(matrix, size=None):
        if size is None:
            size = getNewMatrixSize(matrix, multiplier)
        out = []
        for i in range(len(matrix)):
            arr = np.array(matrix[i])
            arr_interp = interp.interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, size))
            out.append(arr_stretch)

        return out

    data_path = f'{cfg.getFileName()}'
    logger.info("Read physionet file")
    logger.info(data_path)

    signals, fileds = wfdb.rdsamp(data_path)

    sampling_rate = fileds['fs']
    signals = signals.transpose()

    bandpass_notch_channels = []
    for i in signals:
        bandpass_notch_channels.append(bandpass(i, fs=sampling_rate))

    signals = bandpass_notch_channels

    logger.info(f'Fileds: {fileds["sig_name"]}')
    logger.info(f'Sampling rate: {sampling_rate}')

    signal_data = signals[signal]

    _, rpeaks = nk.ecg_peaks(signal_data, sampling_rate=sampling_rate)
    _, waves = nk.ecg_delineate(signal_data, rpeaks, sampling_rate=sampling_rate)
    ECG_P_Peaks = list(np.round(np.array(waves["ECG_P_Peaks"]) / sampling_rate, 4))
    ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / sampling_rate, 4))
    ECG_R_Peaks = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / sampling_rate, 4))
    ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / sampling_rate, 4))
    ECG_T_Peaks = list(np.round(np.array(waves["ECG_T_Peaks"]) / sampling_rate, 4))

    ecg_fr = pd.DataFrame({"ECG_P_Peaks": ECG_P_Peaks, "ECG_Q_Peaks": ECG_Q_Peaks, "ECG_R_Peaks": ECG_R_Peaks,
                           "ECG_S_Peaks": ECG_S_Peaks, "ECG_T_Peaks": ECG_T_Peaks})
    Q_S_exist = ("ECG_Q_Peaks" in ecg_fr and "ECG_S_Peaks" in ecg_fr)

    input_peaks = [ECG_P_Peaks, ECG_Q_Peaks, ECG_S_Peaks, ECG_T_Peaks]
    output_matrix = [list() for i in range(len(input_peaks))]

    time_signal_rhythm = []
    time_peaks_rhythm = [list() for i in range(len(input_peaks))]
    for i in range(len(ECG_P_Peaks) - 1):
        # curr_signal = signal_data[int(ECG_P_Peaks[i] * sampling_rate):int(ECG_P_Peaks[i + 1] * sampling_rate)]
        # show_plot("Signal", [i for i in range(len(curr_signal))], curr_signal)
        size = 0
        for peak_idx in range(len(input_peaks)):
            def replaceNaN(val):
                if np.isnan(val):
                    return 0
                return val


            curr_complex = input_peaks[peak_idx]
            nextCmpIdx = peak_idx + 1
            curr_peak_idx = i
            next_peak_idx = i
            if nextCmpIdx == len(input_peaks):
                nextCmpIdx = 0
                next_peak_idx = i + 1

            next_complex = input_peaks[nextCmpIdx]
            start_peak = curr_complex[curr_peak_idx]
            start = int(replaceNaN(start_peak) * sampling_rate)
            next_peak = next_complex[next_peak_idx]
            if np.isnan(next_peak):
                next_peak = input_peaks[0][i + 1]

            end = int(replaceNaN(next_peak) * sampling_rate)
            complex_slice = signal_data[start:end]
            curr_duration = len(complex_slice)
            size += curr_duration

            output_matrix[peak_idx].append(complex_slice)
            time_peaks_rhythm[peak_idx].append(curr_duration)
            # time_ratio = len(complex_slice) / sampling_rate
            # show_plot("Complex", [i*time_ratio for i in range(len(complex_slice))], complex_slice)
        time_signal_rhythm.append(size)

    rhythm_segments = []
    interpolated_matrix = []
    mean_matrix = []
    variance_matrix = []
    for curr_slice in output_matrix:
        interp_slice = interpolate_matrix(curr_slice)
        rhythm_segments.append(np.diff(interp_slice)) # this is incorrectly calculated rhythm. it should be calculated by time somehow
        mean_matrix.append( [np.mean(i) for i in interp_slice])
        variance_matrix.append([np.var(i) for i in interp_slice])
        interpolated_matrix.append(interp_slice)

    mean_time_rhythm = [np.mean(i) for i in time_peaks_rhythm]
    time_rhythm = []
    for i in range(len(time_peaks_rhythm[0])):
        for pidx in range(len(time_peaks_rhythm)):
            rhythm_v = float(time_peaks_rhythm[pidx][i])
            mean_rhythm = float(mean_time_rhythm[pidx])
            time_rhythm.append(rhythm_v / mean_rhythm)


    show_plot("Time rhythm", [i for i in range(len(time_rhythm))], time_rhythm)

    mod_sampling_rate = int(sampling_rate * multiplier)
    concatenated = np.concatenate(interpolated_matrix, axis=1)
    interpolated_signal = interpolate_matrix(concatenated, mod_sampling_rate)
    # time_ratio = len(complex_slice) / sampling_rate
    # show_plot("Complex", [i*time_ratio for i in range(len(complex_slice))], complex_slice)
    show_plot("Interpolated signal", [i for i in range(len(interpolated_signal[0]))], interpolated_signal[0])
    matrix_P_R = []
    matrix_R_T = []
    matrix_T_P = []

    for i in range(len(ECG_P_Peaks) - 1):
        def replaceNaN(val):
            if np.isnan(val):
                return 0
            return val


        def appendIfNotNaN(segA, segB, lst):

            start = int(replaceNaN(segA[i]) * sampling_rate)
            end = int(replaceNaN(segB[i]) * sampling_rate)

            sig = signals[signal]
            item = sig[start:end]
            lst.append(item)


        # appendIfNotNaN(ECG_P_Peaks, ECG_R_Peaks, matrix_P_R)
        # appendIfNotNaN(ECG_R_Peaks, ECG_T_Peaks, matrix_R_T)
        # appendIfNotNaN(ECG_T_Peaks, ECG_P_Peaks, matrix_T_P)
        start = int(replaceNaN(ECG_P_Peaks[i]) * sampling_rate)
        end = int(replaceNaN(ECG_R_Peaks[i]) * sampling_rate)
        matrix_P_R.append(signal_data[start:end])
        start = int(replaceNaN(ECG_R_Peaks[i]) * sampling_rate)
        end = int(replaceNaN(ECG_T_Peaks[i]) * sampling_rate)
        matrix_R_T.append(signal_data[start:end])
        start = int(replaceNaN(ECG_T_Peaks[i]) * sampling_rate)
        end = int(replaceNaN(ECG_P_Peaks[i + 1]) * sampling_rate)
        matrix_T_P.append(signal_data[start:end])

    matrix_T_P = matrix_T_P
    matrix_P_R = matrix_P_R
    matrix_R_T = matrix_R_T

    if Q_S_exist:
        ECG_Q_Peaks = ecg_fr["ECG_Q_Peaks"]
        ECG_S_Peaks = ecg_fr["ECG_S_Peaks"]

    matrix_T_P_size = getNewMatrixSize(matrix_T_P, multiplier)
    matrix_P_R_size = getNewMatrixSize(matrix_P_R, multiplier)
    matrix_R_T_size = getNewMatrixSize(matrix_R_T, multiplier)


    # print(matrix_T_P_size)
    # print(matrix_P_R_size)
    # print(matrix_R_T_size)

    # matrix_T_P_size = 145
    # matrix_P_R_size = 48
    # matrix_R_T_size = 83


    interp_matrix_T_P = interpolate_matrix(matrix_T_P)
    interp_matrix_P_R = interpolate_matrix(matrix_P_R)
    interp_matrix_R_T = interpolate_matrix(matrix_R_T)

    concat_interp_matrix_all = np.concatenate((interp_matrix_P_R, interp_matrix_R_T, interp_matrix_T_P), axis=1)
    interp_matrix_all = interpolate_matrix(concat_interp_matrix_all, mod_sampling_rate)

    m_m = np.mean(interp_matrix_all, 1)

    prepared_data = interp_matrix_all - m_m[:, None]
    self_m_ = [np.mean(i) for i in prepared_data]
    self_variance__ = [np.var(i) for i in prepared_data]
    self_rhythm = np.diff(prepared_data)
    # Initial moments of the second order
    self_m_2_ = [np.sum(np.array(i) ** 2) / len(i) for i in prepared_data]

    _ = self_m_2_

def getNewMatrixSize(matrix, multiplier):
    n = 0
    for i in range(len(matrix)):
        n = n + len(matrix[i])
    n = int((n / len(matrix)) * multiplier)
    n = int(len(matrix[0]) * multiplier)
    return n


if __name__ == '__main__':
    # outdated()
    database = 'pulse-transit-time-ppg.1.1.0'
    datafile = 's3_run'
    signal = 0
    cfg = new_cfg(database, datafile, signal)
    multiplier = cfg.getMultiplier()
    s = Simulation()

    # data_path = f'{cfg.getFileName()}'
    # logger.info("Read physionet file")
    # logger.info(data_path)
    #
    # signals, fileds = wfdb.rdsamp(data_path)
    #
    # sampling_rate = fileds['fs']
    # signals = signals.transpose()
    #
    # bandpass_notch_channels = []
    # for i in signals:
    #     bandpass_notch_channels.append(bandpass(i, fs=sampling_rate))
    #
    # signals = bandpass_notch_channels
    #
    # logger.info(f'Fileds: {fileds["sig_name"]}')
    # logger.info(f'Sampling rate: {sampling_rate}')
    #
    # signal_data = signals[signal]
    # generated = s.gen_ecg_from_prototype(signal_data, 500)
    # gen_values = []
    # src_values = []
    # time = []
    # for i in range(len(generated)):
    #     time.append(generated[i][0])
    #     gen_values.append(generated[i][1])
    #     src_values.append(signal_data[i])
    #     if i > 600:
    #         break
    # show_plot("Source", time, src_values)
    # show_plot("Generated", time, gen_values)
    # _ = generated

    with open("/Users/alantoo/Desktop/demo_ecg_signal_raw_data.json") as f:
        data = json.load(f)
        generated = s.gen_ecg_from_prototype(data, 500)
        gen_values = []
        src_values = []
        time = []
        offset_data = data[350:]
        for i in range(len(generated)):
            time.append(generated[i][0])
            gen_values.append(generated[i][1])
            src_values.append(offset_data[i])
            if i > 600:
                break
        show_plot("Source", time, src_values)
        show_plot("Generated", time, gen_values)
        _ = generated

