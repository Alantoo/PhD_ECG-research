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
from physio_artifact_config import PhysioArtifactConfig


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

    def gen_ecg_from_prepared(self, prepared, cfg: ArtifactsConfig):
        """Generate ECG from a PreparedSignal using 6 anatomical zones.

        Zone indices match PreparedSignal.get_zone_matrices():
            0 — P-wave        (P_on  → P_off)
            1 — PQ segment    (P_off → Q)
            2 — QRS complex   (Q     → S)
            3 — ST segment    (S     → T_on)
            4 — T-wave        (T_on  → T_off)
            5 — TP baseline   (T_off → P_on of next beat)  ← trailing
        """
        matrices, rhythm_matrix = prepared.get_zone_matrices()
        if not matrices[0]:
            raise ValueError(
                "PreparedSignal has no valid zone cycles — check delineation quality"
            )
        return self.gen_ecg_from_matrix(matrices, rhythm_matrix, cfg, prepared.sampling_rate)

    def gen_ecg_from_matrix(self, ecg_matrix, rhythm_matrix, cfg: ArtifactsConfig, sampling_rate: int = 500):
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

        available_rhythm = len(rhythm_matrix[0])
        cycles_count = cfg.cycles_count if cfg.cycles_count > 0 else 10

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
            for i in range(len(v)):
                time = t[i] + final_seq['last_time']
                final_seq['points'].append([time, v[i]])

            final_seq['last_time'] += len(v) / sampling_rate

        for cycle_ix in range(cycles_count):
            for segment_ix in range(segments_count):
                raw_rhythm = rhythm_matrix[segment_ix]
                mean_rhythm = float(mean_time_rhythm[segment_ix])
                rhythm_v = float(raw_rhythm[cycle_ix % available_rhythm])

                artifact_data = artifacts_idx.get(segment_ix)
                if artifact_data is not None and cycle_ix in artifact_data['places']:
                    segment_cfg: SegmentArtifactsConfig = artifact_data['cfg']
                    # Normalize X to start from 0: points may come from a project signal
                    # with absolute timestamps (e.g. [2.5, 2.502, ...]) rather than relative
                    # ones. Without normalization the first ~N ms of the zone is a straight-line
                    # gap and the custom shape only occupies the tail end of the slot.
                    first_x = segment_cfg.points[0][0] if segment_cfg.points else 0
                    rel_points = [[p[0] - first_x, p[1]] for p in segment_cfg.points]
                    max_ts = rel_points[-1][0] if rel_points else 0

                    # X-axis: explicit duration overrides rhythm; otherwise follow per-cycle rhythm_v.
                    # rhythm_v is in samples; artifact points X-axis is in seconds — convert.
                    rhythm_duration_sec = rhythm_v / sampling_rate
                    if segment_cfg.duration is not None and segment_cfg.duration > 0:
                        target_duration_sec = segment_cfg.duration
                        rhythm_ratio = target_duration_sec / max_ts if max_ts > 0 else 1.0
                    else:
                        rhythm_ratio = rhythm_duration_sec / max_ts if max_ts > 0 else 1.0

                    scaled_points = [[p[0] * rhythm_ratio, p[1]] for p in rel_points]

                    # Y-axis: remap amplitude to [min_height, max_height]
                    if segment_cfg.min_height is not None and segment_cfg.max_height is not None:
                        y_vals = [p[1] for p in scaled_points]
                        y_min, y_max = min(y_vals), max(y_vals)
                        y_range = y_max - y_min
                        h_range = segment_cfg.max_height - segment_cfg.min_height
                        if y_range > 0:
                            scaled_points = [
                                [p[0], segment_cfg.min_height + (p[1] - y_min) / y_range * h_range]
                                for p in scaled_points
                            ]
                        else:
                            mid = (segment_cfg.min_height + segment_cfg.max_height) / 2
                            scaled_points = [[p[0], mid] for p in scaled_points]

                    t, v = zip(*scaled_points)
                    prev_last_time = final_seq['last_time']
                    append_seg_point(t, v)
                    # Override: advance last_time by actual scaled duration (t[-1] + one sample),
                    # not by len(v)/sampling_rate (the unscaled point count). Without this,
                    # when rhythm_ratio ≠ 1 the next zone starts at the wrong time, making
                    # the time axis non-monotonic and creating overlapping or gapped zones.
                    final_seq['last_time'] = prev_last_time + t[-1] + 1.0 / sampling_rate
                    continue

                raw_data = interpolated_matrix[segment_ix][cycle_ix]

                mean_data = mean_matrix[segment_ix]
                mean_time = [i for i in range(len(mean_data))]
                variance_data = variance_matrix[segment_ix]
                variance_time = [i for i in range(len(variance_data))]

                segment_duration = len(raw_data)
                rhythm_ratio = rhythm_v / mean_rhythm
                new_duration = max(1, int(segment_duration * rhythm_ratio))
                t = np.arange(new_duration, dtype=float) / sampling_rate

                direction = 1
                if bool(random.getrandbits(1)):
                    direction = -1

                ecg_signal = mean_data + direction * np.sqrt(np.abs(variance_data))
                postprocessed = resample(ecg_signal, new_duration)
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

    def gen_ecg_from_math_stats(self, segments_count, mean, variance, rhythm, cfg: ArtifactsConfig, sampling_rate: int = 500,
                                mean_7zones=None, var_7zones=None, variance_scale: float = 0.3,
                                physio_artifacts: list[PhysioArtifactConfig] | None = None):
        def to_matrix(time_series):
            values = time_series[1]
            matrix = list()
            for ix in range(segments_count):
                rsix = int(ix * (len(values) / segments_count))
                reix = int((rsix + (len(values) / segments_count)))
                matrix.append(values[rsix:reix])
            return matrix

        def to_rhythm_matrix(time_series):
            # Rhythm is interleaved: [R1→R2, P1→P2, T1→T2, Q1→Q2, S1→S2, R2→R3, ...]
            # Stride by segments_count to recover per-peak-type interval lists
            values = time_series[1]
            return [list(values[ix::segments_count]) for ix in range(segments_count)]

        def to_matrix_zone(time_series, mean_durations):
            """Fallback: split flat mean beat into per-zone arrays proportionally.

            Used when mean_7zones is not supplied. All zones including trailing TP
            are proportional slices of the full mean beat waveform, sized according
            to each zone's mean duration.
            """
            values = time_series[1]
            n = len(values)
            total = sum(mean_durations)

            if total <= 0:
                return to_matrix(time_series)

            matrix = []
            offset = 0
            for z, dur in enumerate(mean_durations):
                if z < len(mean_durations) - 1:
                    size = max(4, int(round(dur / total * n)))
                    matrix.append(np.array(values[offset:offset + size]))
                    offset += size
                else:
                    matrix.append(np.array(values[offset:]))   # last zone: remainder

            return matrix

        def to_variance_zone(time_series, mean_durations):
            """Fallback: same proportional split as to_matrix_zone for variance.

            All zones including trailing TP get real variance from the flat variance
            array — no fabricated near-zero values.
            """
            return to_matrix_zone(time_series, mean_durations)

        def to_split_from_7zone(time_series, mean_durations):
            """Split pre-computed per-zone concatenated data into a zone matrix.

            The concatenated array was produced by get_zone_stats(), which writes
            zones 0-5 sequentially, each sized according to the actual zone arrays.
            We recover the sizes from mean_durations (same ordering as rhythm_7zones).
            """
            values = time_series[1]
            matrix = []
            offset = 0
            for d in mean_durations:
                size = max(4, int(round(d)))
                chunk = np.array(values[offset:offset + size], dtype=float)
                if len(chunk) == 0:
                    chunk = np.zeros(size)
                matrix.append(chunk)
                offset += size
            return matrix

        rhythm_matrix = to_rhythm_matrix(rhythm)
        mean_time_rhythm = [np.mean(i) for i in rhythm_matrix]

        if segments_count == 6 and mean_7zones is not None and var_7zones is not None:
            mean_matrix     = to_split_from_7zone(mean_7zones,   mean_time_rhythm)
            variance_matrix = to_split_from_7zone(var_7zones,    mean_time_rhythm)
        elif segments_count == 6:
            mean_matrix     = to_matrix_zone(mean,     mean_time_rhythm)
            variance_matrix = to_variance_zone(variance, mean_time_rhythm)
        else:
            mean_matrix     = to_matrix(mean)
            variance_matrix = to_matrix(variance)

        available_rhythm = len(rhythm_matrix[0])
        cycles_count = cfg.cycles_count if cfg.cycles_count > 0 else 10

        artifacts_idx = dict()
        for c in cfg.segment_cfg:
            insert_places = pick_random_unique_n(cycles_count, c.count_or_pos) if not c.exact_placement else [
                c.count_or_pos - 1]
            artifacts_idx[c.index] = {
                "places": insert_places,
                "cfg": c,
            }

        # Physio artifact index: cycle_ix → PhysioArtifactConfig
        physio_idx: dict[int, PhysioArtifactConfig] = {}
        for pa in (physio_artifacts or []):
            places = pick_random_unique_n(cycles_count, pa.count_or_pos) if not pa.exact_placement else [pa.count_or_pos - 1]
            for place in [int(p) for p in places]:
                if 0 <= place < cycles_count:
                    physio_idx[place] = pa

        # points = list()
        # last_time = 0
        final_seq = {
            'last_time': 0,
            'points': list(),
        }

        def append_seg_point(t: list[float], v: list[float]):
            for i in range(len(v)):
                time = t[i] + final_seq['last_time']
                final_seq['points'].append([time, v[i]])

            final_seq['last_time'] += len(v) / sampling_rate

        for cycle_ix in range(cycles_count):
            physio_art = physio_idx.get(cycle_ix)
            for segment_ix in range(segments_count):
                raw_rhythm = rhythm_matrix[segment_ix]
                mean_rhythm = float(mean_time_rhythm[segment_ix])
                rhythm_v = float(raw_rhythm[cycle_ix % available_rhythm])

                if physio_art and physio_art.artifact_type == 'rhythm':
                    if segments_count == 6:
                        # Only stretch the trailing TP zone so that exactly one RR
                        # interval is affected. Scaling all zones would bleed into
                        # both the preceding and following RR intervals.
                        if segment_ix == segments_count - 1:
                            rhythm_v *= physio_art.rr_scale
                    else:
                        rhythm_v *= physio_art.rr_scale

                artifact_data = artifacts_idx.get(segment_ix)
                if artifact_data is not None and cycle_ix in artifact_data['places']:
                    segment_cfg: SegmentArtifactsConfig = artifact_data['cfg']
                    # Normalize X to start from 0 (same reason as gen_ecg_from_matrix).
                    first_x = segment_cfg.points[0][0] if segment_cfg.points else 0
                    rel_points = [[p[0] - first_x, p[1]] for p in segment_cfg.points]
                    max_ts = rel_points[-1][0] if rel_points else 0

                    # X-axis: explicit duration overrides rhythm; otherwise follow per-cycle rhythm_v.
                    # rhythm_v is in samples; artifact points X-axis is in seconds — convert.
                    rhythm_duration_sec = rhythm_v / sampling_rate
                    if segment_cfg.duration is not None and segment_cfg.duration > 0:
                        target_duration_sec = segment_cfg.duration
                        rhythm_ratio = target_duration_sec / max_ts if max_ts > 0 else 1.0
                    else:
                        rhythm_ratio = rhythm_duration_sec / max_ts if max_ts > 0 else 1.0

                    scaled_points = [[p[0] * rhythm_ratio, p[1]] for p in rel_points]

                    if segment_cfg.min_height is not None and segment_cfg.max_height is not None:
                        y_vals = [p[1] for p in scaled_points]
                        y_min, y_max = min(y_vals), max(y_vals)
                        y_range = y_max - y_min
                        h_range = segment_cfg.max_height - segment_cfg.min_height
                        if y_range > 0:
                            scaled_points = [
                                [p[0], segment_cfg.min_height + (p[1] - y_min) / y_range * h_range]
                                for p in scaled_points
                            ]
                        else:
                            mid = (segment_cfg.min_height + segment_cfg.max_height) / 2
                            scaled_points = [[p[0], mid] for p in scaled_points]

                    t, v = zip(*scaled_points)
                    prev_last_time = final_seq['last_time']
                    append_seg_point(t, v)
                    final_seq['last_time'] = prev_last_time + t[-1] + 1.0 / sampling_rate
                    continue

                mean_data = mean_matrix[segment_ix]
                variance_data = variance_matrix[segment_ix]

                segment_duration = len(mean_data)
                rhythm_ratio = rhythm_v / mean_rhythm
                new_duration = max(1, int(segment_duration * rhythm_ratio))
                t = np.arange(new_duration, dtype=float) / sampling_rate

                ZONE_NOISE_FACTORS = [0.6, 0.1, 1.0, 0.05, 0.6, 0.6]  # P, PQ, QRS, ST, T, TP
                zone_factor = ZONE_NOISE_FACTORS[segment_ix] if segments_count == 6 and segment_ix < len(ZONE_NOISE_FACTORS) else 1.0

                # Shape artifact: boost noise scale for this cycle
                effective_variance_scale = variance_scale
                if physio_art and physio_art.artifact_type == 'shape':
                    effective_variance_scale *= physio_art.noise_scale

                n = len(mean_data)
                if n < 20:
                    ecg_signal = mean_data
                else:
                    std = effective_variance_scale * zone_factor * np.sqrt(np.abs(variance_data))
                    noise = np.random.normal(0, std)
                    wl = min(max(7, (n // 2) | 1), 101)
                    if wl % 2 == 0:
                        wl -= 1
                    noise = savgol_filter(noise, window_length=wl, polyorder=2)
                    ecg_signal = mean_data + noise

                # Amplitude artifact: scale the whole zone signal
                if physio_art and physio_art.artifact_type == 'amplitude':
                    ecg_signal = ecg_signal * physio_art.amplitude_scale

                if new_duration != len(ecg_signal):
                    x_old = np.linspace(0, 1, len(ecg_signal))
                    x_new = np.linspace(0, 1, new_duration)
                    postprocessed = np.interp(x_new, x_old, ecg_signal)
                else:
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
