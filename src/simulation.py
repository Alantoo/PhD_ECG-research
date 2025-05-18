import random

from scipy.signal import find_peaks, savgol_filter
import numpy as np
from scipy.interpolate import interp1d
import scipy.interpolate as interp
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample

def show_plot(title, t_data, sig_data):
    # Візуалізація
    plt.plot(t_data, sig_data, label="Змодельований сигнал")
    plt.xlabel("Час")
    plt.ylabel("Сигнал")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def getNewMatrixSize(matrix, multiplier):
    n = 0
    for i in range(len(matrix)):
        n = n + len(matrix[i])
    n = int((n / len(matrix)) * multiplier)
    n = int(len(matrix[0]) * multiplier)
    return n

def interpolate_matrix(matrix, size=None):
    if size is None:
        size = getNewMatrixSize(matrix, 1)
    out = []
    for i in range(len(matrix)):
        arr = np.array(matrix[i])
        arr_interp = interp.interp1d(np.arange(arr.size), arr)
        arr_stretch = arr_interp(np.linspace(0, arr.size - 1, size))
        out.append(arr_stretch)

    return out

class Simulation:
    def __init__(self):
        pass


    def gen_cycle(self, rhythm_data, variance_data, mean_data, count):
        last_time = 0
        points = list()
        for iter in range(count):
            # Інтерполяція для приведення до спільного часу
            t = np.linspace(0, 1, 500)  # Спільний часовий інтервал

            rhythm_interp = interp1d(rhythm_data[0], rhythm_data[1], kind='cubic', fill_value="extrapolate")
            mean_interp = interp1d(mean_data[0], mean_data[1], kind='cubic', fill_value="extrapolate")
            variance_interp = interp1d(variance_data[0], variance_data[1], kind='cubic', fill_value="extrapolate")

            # Отримуємо значення на всьому часовому інтервалі
            rhythm = rhythm_interp(t)
            mean = mean_interp(t)
            variance = variance_interp(t)

            # Генерація циклічного сигналу
            ecg_signal = mean + np.sin(2 * np.pi * rhythm * t) + np.random.normal(0, np.sqrt(variance), len(t))
            show_plot(f"Mean", t, mean)
            show_plot(f"Variance", t, variance)
            show_plot(f"Generated", t, ecg_signal)

            postprocessed = savgol_filter(ecg_signal, window_length=13, polyorder=3)
            last_tval = 0
            for i in range(len(postprocessed)):
                time = t[i] + last_time
                value = postprocessed[i]
                last_tval = time
                points.append([time, value])

            last_time = last_tval

        return points

    def gen_ecg_from_prototype(self, signal_data, sampling_rate):
        _, rpeaks = nk.ecg_peaks(signal_data, sampling_rate=sampling_rate)
        _, waves = nk.ecg_delineate(signal_data, rpeaks, sampling_rate=sampling_rate)
        ECG_P_Peaks = list(np.round(np.array(waves["ECG_P_Peaks"]) / sampling_rate, 4))
        ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / sampling_rate, 4))
        # ECG_R_Peaks = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / sampling_rate, 4))
        ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / sampling_rate, 4))
        ECG_T_Peaks = list(np.round(np.array(waves["ECG_T_Peaks"]) / sampling_rate, 4))

        # ecg_fr = pd.DataFrame({"ECG_P_Peaks": ECG_P_Peaks, "ECG_Q_Peaks": ECG_Q_Peaks, "ECG_R_Peaks": ECG_R_Peaks,
        #                        "ECG_S_Peaks": ECG_S_Peaks, "ECG_T_Peaks": ECG_T_Peaks})

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

        return self.gen_ecg_from_matrix(output_matrix, time_peaks_rhythm)

    def gen_ecg_from_matrix(self, ecg_matrix, rhythm_matrix):
        interpolated_matrix = []
        mean_matrix = []
        variance_matrix = []
        for curr_slice in ecg_matrix:
            interp_slice = interpolate_matrix(curr_slice)
            transposed_slice = np.transpose(interp_slice)
            mean_slice = [np.mean(i) for i in transposed_slice]
            mean_matrix.append(mean_slice)
            variance_slice = [np.var(i) for i in transposed_slice]
            variance_matrix.append(variance_slice)
            interpolated_matrix.append(interp_slice)

        # for i in range(len(mean_matrix)):
        #     curr_mean = mean_matrix[i]
        #     curr_variance = variance_matrix[i]
        #     mean_time = [i for i in range(len(curr_mean))]
        #     variance_time = [i for i in range(len(curr_variance))]
        #     show_plot(f"Mean {i}", mean_time, curr_mean)
        #     show_plot(f"Variance {i}", variance_time, curr_variance)

        # demo_mean = []
        # demo_variance = []
        # for i in range(len(variance_matrix)):
        #     demo_mean.extend(mean_matrix[i])
        #     demo_variance.extend(variance_matrix[i])

        # demo_mean_time = [i for i in range(len(demo_mean))]
        # demo_variance_time = [i for i in range(len(demo_variance))]
        # show_plot(f"Mean", demo_mean_time, demo_mean)
        # show_plot(f"Variance ", demo_variance_time, demo_variance)

        mean_time_rhythm = [np.mean(i) for i in rhythm_matrix]
        time_rhythm = []
        for i in range(len(rhythm_matrix[0])):
            for pidx in range(len(rhythm_matrix)):
                rhythm_v = float(rhythm_matrix[pidx][i])
                mean_rhythm = float(mean_time_rhythm[pidx])
                time_rhythm.append(rhythm_v / mean_rhythm)

        cycles_count = len(rhythm_matrix[0])
        segments_count = len(interpolated_matrix)
        points = list()
        last_time = 0
        for cycle_ix in range(cycles_count):
            for segment_ix in range(segments_count):
                raw_data = interpolated_matrix[segment_ix][cycle_ix]

                raw_rhythm = rhythm_matrix[segment_ix]
                mean_rhythm = float(mean_time_rhythm[segment_ix])
                rhythm_v = float(raw_rhythm[cycle_ix])
                rhythm_ratio = rhythm_v / mean_rhythm

                mean_data = mean_matrix[segment_ix]
                mean_time = [i for i in range(len(mean_data))]
                variance_data = variance_matrix[segment_ix]
                variance_time = [i for i in range(len(variance_data))]

                # Інтерполяція для приведення до спільного часу
                segment_duration = len(raw_data)
                new_duration = int(segment_duration * rhythm_ratio)
                t = np.linspace(0, new_duration, new_duration)  # Спільний часовий інтервал

                mean_interp = interp1d(mean_time, mean_data, kind='cubic', fill_value="extrapolate")
                variance_interp = interp1d(variance_time, variance_data, kind='cubic', fill_value="extrapolate")

                # Отримуємо значення на всьому часовому інтервалі
                # mean = resample(mean_data, new_duration)
                # variance = resample(variance_data, new_duration)
                mean = mean_interp(t)
                variance = variance_interp(t)

                # Генерація циклічного сигналу
                direction = 1
                if bool(random.getrandbits(1)):
                    direction = -1

                ecg_signal = mean + direction * np.sqrt(variance)
                # rhythm = 1
                # ecg_signal = mean + np.sin(2 * np.pi * rhythm * t) + np.random.normal(0, np.sqrt(variance), len(t))

                # show_plot(f"Mean {segment_ix}", mean_time, mean_data)
                # show_plot(f"Mean {segment_ix} 2", t, mean)
                # show_plot(f"Variance {segment_ix}", variance_time, variance_data)
                # show_plot(f"Variance {segment_ix} 2", t, variance)
                # show_plot(f"Generated {segment_ix}", t, ecg_signal)

                # postprocessed = savgol_filter(ecg_signal, window_length=min(13, len(ecg_signal)), polyorder=3)
                postprocessed = ecg_signal
                # show_plot(f"Generated {segment_ix}", t, postprocessed)
                last_tval = 0
                for i in range(len(postprocessed)):
                    time = t[i] + last_time
                    value = postprocessed[i]
                    last_tval = time
                    points.append([time, value])

                last_time = last_tval

        return points