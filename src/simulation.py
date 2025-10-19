import random

from scipy.signal import find_peaks, savgol_filter
import numpy as np
from scipy.interpolate import interp1d
import scipy.interpolate as interp
import neurokit2 as nk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample

from artifacts_config import ArtifactsConfig
from segment_artifacts_config import SegmentArtifactsConfig


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


def pick_random_unique_n(size: int, n: int):
    seq = np.linspace(0, size - 1, size)
    seq_copy = seq.copy().tolist()
    np.random.shuffle(seq_copy)
    return seq_copy[:n]


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

    def gen_ecg_from_prototype(self, signal_data, sampling_rate, cfg: ArtifactsConfig):
        _, rpeaks = nk.ecg_peaks(signal_data, sampling_rate=sampling_rate)
        _, waves = nk.ecg_delineate(signal_data, rpeaks, sampling_rate=sampling_rate)
        ECG_P_Peaks = list(np.round(np.array(waves["ECG_P_Peaks"]) / sampling_rate, 4))
        ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / sampling_rate, 4))
        ECG_R_Peaks = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / sampling_rate, 4))
        ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / sampling_rate, 4))
        ECG_T_Peaks = list(np.round(np.array(waves["ECG_T_Peaks"]) / sampling_rate, 4))

        # ecg_fr = pd.DataFrame({"ECG_P_Peaks": ECG_P_Peaks, "ECG_Q_Peaks": ECG_Q_Peaks, "ECG_R_Peaks": ECG_R_Peaks,
        #                        "ECG_S_Peaks": ECG_S_Peaks, "ECG_T_Peaks": ECG_T_Peaks})

        input_peaks = [ECG_P_Peaks, ECG_Q_Peaks, ECG_R_Peaks, ECG_S_Peaks, ECG_T_Peaks]
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

        return self.gen_ecg_from_matrix(output_matrix, time_peaks_rhythm, cfg)

    def gen_ecg_from_matrix(self, ecg_matrix, rhythm_matrix, cfg: ArtifactsConfig):
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

        cycles_count = min(len(rhythm_matrix[0]), cfg.cycles_count if cfg.cycles_count > 0 else 10)

        segments_count = len(interpolated_matrix)
        artifacts_idx = dict()
        for c in cfg.segment_cfg:
            insert_places = pick_random_unique_n(cycles_count, c.count_or_pos) if not c.exact_placement else [
                c.count_or_pos - 1]
            artifacts_idx[c.index] = {
                "places": insert_places,
                "cfg": c,
            }

        # points = list()
        # last_time = 0
        final_seq = {
            'last_time': 0,
            'points': list(),
        }

        def append_seg_point(t: list[float], v: list[float]):
            last_tval = 0
            for i in range(len(v)):
                time = t[i] + final_seq['last_time']
                value = v[i]
                last_tval = time
                final_seq['points'].append([time, value])

            final_seq['last_time'] = last_tval

        for cycle_ix in range(cycles_count):
            for segment_ix in range(segments_count):
                raw_rhythm = rhythm_matrix[segment_ix]
                mean_rhythm = float(mean_time_rhythm[segment_ix])
                rhythm_v = float(raw_rhythm[cycle_ix])

                artifact_data = artifacts_idx.get(segment_ix)
                if artifact_data is not None and cycle_ix in artifact_data['places']:
                    segment_cfg: SegmentArtifactsConfig = artifact_data['cfg']
                    max_ts = segment_cfg.points[-1][0]
                    rhythm_ratio = mean_rhythm / max_ts
                    scaled_points = [[p[0] * rhythm_ratio, p[1]] for p in segment_cfg.points]
                    t, v = zip(*scaled_points)
                    append_seg_point(t, v)
                    continue

                raw_data = interpolated_matrix[segment_ix][cycle_ix]

                mean_data = mean_matrix[segment_ix]
                mean_time = [i for i in range(len(mean_data))]
                variance_data = variance_matrix[segment_ix]
                variance_time = [i for i in range(len(variance_data))]

                # Інтерполяція для приведення до спільного часу
                segment_duration = len(raw_data)
                rhythm_ratio = rhythm_v / mean_rhythm
                new_duration = int(segment_duration * rhythm_ratio)
                t = np.linspace(0, new_duration, segment_duration)  # Спільний часовий інтервал

                # mean_interp = interp1d(mean_time, mean_data, kind='nearest', fill_value="extrapolate")
                # variance_interp = interp1d(variance_time, variance_data, kind='nearest', fill_value="extrapolate")

                # Отримуємо значення на всьому часовому інтервалі
                # mean = resample(mean_data, new_duration)
                # variance = resample(variance_data, new_duration)
                # mean = mean_interp(t)
                # variance = variance_interp(t)

                # Генерація циклічного сигналу
                direction = 1
                if bool(random.getrandbits(1)):
                    direction = -1

                ecg_signal = mean_data + direction * np.sqrt(np.abs(variance_data))
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
                append_seg_point(t, postprocessed)
                # last_tval = 0
                # for i in range(len(postprocessed)):
                #     time = t[i] + last_time
                #     value = postprocessed[i]
                #     last_tval = time
                #     points.append([time, value])
                #
                # last_time = last_tval

        def stats_matrix_to_points(matrix):
            data = []
            for row in matrix:
                data.extend(row)
            data_time = np.linspace(0, len(data), len(data))
            output_points = []
            for i in range(len(data)):
                output_points.append([data_time[i], data[i]])

            return output_points

        meta = {
            "mean": stats_matrix_to_points(mean_matrix),
            "variance": stats_matrix_to_points(variance_matrix),
            "rhythm": stats_matrix_to_points(rhythm_matrix),
            "mean_rhythm": mean_time_rhythm
        }

        return final_seq['points'], meta

    def gen_ecg_from_math_stats(self, segments_count, mean, variance, rhythm, cfg: ArtifactsConfig):
        def to_matrix(time_series):
            values = time_series[1]
            matrix = list()
            for ix in range(segments_count):
                rsix = int(ix * (len(values) / segments_count))
                reix = int((rsix+(len(values) / segments_count)))
                matrix.append(values[rsix:reix])
            return matrix

        mean_matrix, variance_matrix, rhythm_matrix = to_matrix(mean), to_matrix(variance), to_matrix(rhythm)
        mean_time_rhythm = [np.mean(i) for i in rhythm_matrix]
        time_rhythm = []
        for i in range(len(rhythm_matrix[0])):
            for pidx in range(len(rhythm_matrix)):
                rhythm_v = float(rhythm_matrix[pidx][i])
                mean_rhythm = float(mean_time_rhythm[pidx])
                time_rhythm.append(rhythm_v / mean_rhythm)

        cycles_count = min(len(rhythm_matrix[0]), cfg.cycles_count if cfg.cycles_count > 0 else 10)

        artifacts_idx = dict()
        for c in cfg.segment_cfg:
            insert_places = pick_random_unique_n(cycles_count, c.count_or_pos) if not c.exact_placement else [
                c.count_or_pos - 1]
            artifacts_idx[c.index] = {
                "places": insert_places,
                "cfg": c,
            }

        # points = list()
        # last_time = 0
        final_seq = {
            'last_time': 0,
            'points': list(),
        }

        def append_seg_point(t: list[float], v: list[float]):
            last_tval = 0
            for i in range(len(v)):
                time = t[i] + final_seq['last_time']
                value = v[i]
                last_tval = time
                final_seq['points'].append([time, value])

            final_seq['last_time'] = last_tval

        for cycle_ix in range(cycles_count):
            for segment_ix in range(segments_count):
                raw_rhythm = rhythm_matrix[segment_ix]
                mean_rhythm = float(mean_time_rhythm[segment_ix])
                rhythm_v = float(raw_rhythm[cycle_ix])

                artifact_data = artifacts_idx.get(segment_ix)
                if artifact_data is not None and cycle_ix in artifact_data['places']:
                    segment_cfg: SegmentArtifactsConfig = artifact_data['cfg']
                    max_ts = segment_cfg.points[-1][0]
                    rhythm_ratio = mean_rhythm / max_ts
                    scaled_points = [[p[0] * rhythm_ratio, p[1]] for p in segment_cfg.points]
                    t, v = zip(*scaled_points)
                    append_seg_point(t, v)
                    continue

                mean_data = mean_matrix[segment_ix]
                variance_data = variance_matrix[segment_ix]

                # Інтерполяція для приведення до спільного часу
                segment_duration = len(mean_data)
                rhythm_ratio = rhythm_v / mean_rhythm
                new_duration = int(segment_duration * rhythm_ratio)
                t = np.linspace(0, new_duration, segment_duration)  # Спільний часовий інтервал

                # Генерація циклічного сигналу
                direction = 1
                if bool(random.getrandbits(1)):
                    direction = -1

                ecg_signal = mean_data + direction * np.sqrt(np.abs(variance_data))

                postprocessed = ecg_signal
                append_seg_point(t, postprocessed)

        def stats_matrix_to_points(matrix):
            data = []
            for row in matrix:
                data.extend(row)
            data_time = np.linspace(0, len(data), len(data))
            output_points = []
            for i in range(len(data)):
                output_points.append([data_time[i], data[i]])

            return output_points

        meta = {
            "mean": stats_matrix_to_points(mean_matrix),
            "variance": stats_matrix_to_points(variance_matrix),
            "rhythm": stats_matrix_to_points(rhythm_matrix),
            "mean_rhythm": mean_time_rhythm
        }

        return final_seq['points'], meta
