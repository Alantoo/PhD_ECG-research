import logging
import numpy as np
import neurokit2 as nk
import pandas as pd
import scipy.interpolate as interp
import wfdb.processing

logger = logging.getLogger(__name__)


class PreparedSignal:
    def __init__(self, signal, sampling_rate):
        self.sampling_rate = sampling_rate
        self.signal = signal
        self.multiplier = 1

        cleaned = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)
        r_indices = wfdb.processing.xqrs_detect(np.array(cleaned, dtype=float), fs=self.sampling_rate, verbose=False)
        rpeaks = {"ECG_R_Peaks": r_indices}
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=self.sampling_rate)
        def to_sec(key):
            return list(np.round(np.array(waves[key]) / self.sampling_rate, 4))

        ECG_P_Peaks    = to_sec("ECG_P_Peaks")
        ECG_Q_Peaks    = to_sec("ECG_Q_Peaks")
        ECG_R_Peaks    = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate, 4))
        ECG_S_Peaks    = to_sec("ECG_S_Peaks")
        ECG_T_Peaks    = to_sec("ECG_T_Peaks")
        ECG_P_Onsets   = to_sec("ECG_P_Onsets")
        ECG_P_Offsets  = to_sec("ECG_P_Offsets")
        ECG_T_Onsets   = to_sec("ECG_T_Onsets")
        ECG_T_Offsets  = to_sec("ECG_T_Offsets")

        ecg_fr = pd.DataFrame({
            "ECG_P_Peaks": ECG_P_Peaks, "ECG_Q_Peaks": ECG_Q_Peaks,
            "ECG_R_Peaks": ECG_R_Peaks, "ECG_S_Peaks": ECG_S_Peaks,
            "ECG_T_Peaks": ECG_T_Peaks,
            "ECG_P_Onsets": ECG_P_Onsets, "ECG_P_Offsets": ECG_P_Offsets,
            "ECG_T_Onsets": ECG_T_Onsets, "ECG_T_Offsets": ECG_T_Offsets,
        })
        self.rpeaks = rpeaks
        self.waves = waves
        self.ecg_fr = ecg_fr
        self.ECG_P_Peaks   = ecg_fr["ECG_P_Peaks"]
        self.ECG_Q_Peaks   = ecg_fr["ECG_Q_Peaks"]
        self.ECG_R_Peaks   = ecg_fr["ECG_R_Peaks"]
        self.ECG_S_Peaks   = ecg_fr["ECG_S_Peaks"]
        self.ECG_T_Peaks   = ecg_fr["ECG_T_Peaks"]
        self.ECG_P_Onsets  = ecg_fr["ECG_P_Onsets"]
        self.ECG_P_Offsets = ecg_fr["ECG_P_Offsets"]
        self.ECG_T_Onsets  = ecg_fr["ECG_T_Onsets"]
        self.ECG_T_Offsets = ecg_fr["ECG_T_Offsets"]

        self.Q_S_exist = ("ECG_Q_Peaks" in ecg_fr and "ECG_S_Peaks" in ecg_fr)

        if self.Q_S_exist:
            self.ECG_Q_Peaks = self.ecg_fr["ECG_Q_Peaks"]
            self.ECG_S_Peaks = self.ecg_fr["ECG_S_Peaks"]

        def make_lookup(series):
            return {i: float(v) if not np.isnan(float(v)) else None for i, v in enumerate(series)}

        zone_keys = ["ECG_R_Peaks", "ECG_P_Peaks", "ECG_T_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks"]
        zone_raw   = [ECG_R_Peaks, ECG_P_Peaks, ECG_T_Peaks, ECG_Q_Peaks, ECG_S_Peaks]
        zone_lookups = [make_lookup(s) for s in zone_raw]
        n_beats = len(ECG_R_Peaks)

        # rhythm_points: interleaved per-zone intervals, only where BOTH consecutive
        # beats have all 5 peaks valid (keeps alignment for modelling/stats)
        rhythm_points = []
        beat_idx = 1
        for i in range(n_beats - 1):
            vals_i  = [lk.get(i)     for lk in zone_lookups]
            vals_i1 = [lk.get(i + 1) for lk in zone_lookups]
            if any(v is None for v in vals_i + vals_i1):
                continue
            for v0, v1 in zip(vals_i, vals_i1):
                rhythm_points.append([beat_idx, round(v1 - v0, 4)])
                beat_idx += 1
        self.rhythm_points = [[round(x / 6, 6), y] for x, y in rhythm_points]

        # zone_intervals: per-zone independent intervals, consecutive beats only,
        # NaN in one zone does NOT affect other zones
        zone_intervals: dict[str, list[tuple[int, float]]] = {}
        for key, lk in zip(zone_keys, zone_lookups):
            intervals = []
            for i in range(n_beats - 1):
                t0 = lk.get(i)
                t1 = lk.get(i + 1)
                if t0 is not None and t1 is not None:
                    intervals.append((i + 1, round(t1 - t0, 4)))
            zone_intervals[key] = intervals
        self.zone_intervals = zone_intervals

        sig_arr = np.array(self.signal)

        def extract(on_sec, off_sec):
            s = int(float(on_sec) * self.sampling_rate)
            e = int(float(off_sec) * self.sampling_rate)
            return sig_arr[s:e] if e > s else None

        matrix_P_wave, matrix_QRS, matrix_T_wave, matrix_beat = [], [], [], []
        skipped_nan = 0
        skipped_too_short = 0
        for i in range(len(self.ECG_P_Onsets)):
            p_on  = float(self.ECG_P_Onsets.iloc[i])
            p_off = float(self.ECG_P_Offsets.iloc[i])
            q     = float(self.ECG_Q_Peaks.iloc[i])
            s     = float(self.ECG_S_Peaks.iloc[i])
            t_on  = float(self.ECG_T_Onsets.iloc[i])
            t_off = float(self.ECG_T_Offsets.iloc[i])
            named = {'p_on': p_on, 'p_off': p_off, 'q': q, 's': s, 't_on': t_on, 't_off': t_off}
            missing = [name for name, v in named.items() if np.isnan(v)]
            if missing:
                skipped_nan += 1
                logger.warning(
                    "Beat #%d skipped (NaN fiducial point(s): %s) | %s",
                    i, ', '.join(missing),
                    '  '.join(f"{k}={v:.4f}" if not np.isnan(v) else f"{k}=NaN" for k, v in named.items()),
                )
                continue
            pw   = extract(p_on, p_off)
            qrs  = extract(q, s)
            tw   = extract(t_on, t_off)
            beat = extract(p_on, t_off)
            if any(x is None for x in [pw, qrs, tw, beat]):
                skipped_too_short += 1
                logger.warning("Beat #%d skipped (empty segment slice) | p_on=%.4f p_off=%.4f q=%.4f s=%.4f t_on=%.4f t_off=%.4f", i, p_on, p_off, q, s, t_on, t_off)
                continue
            if len(pw) < 2 or len(qrs) < 2 or len(tw) < 2 or len(beat) < 4:
                skipped_too_short += 1
                logger.warning("Beat #%d skipped (segment too short: pw=%d qrs=%d tw=%d beat=%d samples) | p_on=%.4f t_off=%.4f", i, len(pw), len(qrs), len(tw), len(beat), p_on, t_off)
                continue
            matrix_P_wave.append(pw)
            matrix_QRS.append(qrs)
            matrix_T_wave.append(tw)
            matrix_beat.append(beat)

        self.matrix_P_wave = matrix_P_wave
        self.matrix_QRS    = matrix_QRS
        self.matrix_T_wave = matrix_T_wave
        self.matrix_beat   = matrix_beat

        total = len(self.ECG_P_Onsets)
        kept  = len(matrix_beat)
        logger.info(
            "PreparedSignal: %d/%d beats kept  (skipped: %d NaN fiducials, %d too-short segments)",
            kept, total, skipped_nan, skipped_too_short,
        )

    def get_interpolated_matrix(self):
        mod_sampling_rate = int(self.sampling_rate * self.multiplier)
        interp_matrix = []
        for beat in self.matrix_beat:
            arr = np.array(beat, dtype=float)
            f = interp.interp1d(np.arange(arr.size), arr)
            interp_matrix.append(f(np.linspace(0, arr.size - 1, mod_sampling_rate)).tolist())
        return interp_matrix, mod_sampling_rate

    def get_stats(self):
        interp_matrix_all, _ = self.get_interpolated_matrix()

        m_m = np.mean(interp_matrix_all, 1)

        data = interp_matrix_all - m_m[:, None]

        data = np.transpose(data)
        # Mathematical expectation
        mathematicalExpectation = [np.mean(i) for i in data]
        # self.variance__ = [np.var(i) for i in data]
        rhythm = np.diff(data)
        # Initial moments of the second order
        initialMomentsSecondOrder = [np.sum(np.array(i) ** 2) / len(i) for i in data]

        # fs = data.data.shape[0]
        # peaks, _ = find_peaks(data, height=0.5, distance=fs * 0.6)  # Мінімальна відстань ~600 мс
        #
        # # Обчислення RR-інтервалів
        # rr_intervals = np.diff(peaks) / fs  # Інтервали між піками (у секундах)
        #
        # # Обчислення миттєвої частоти серцевих скорочень (ЧСС)
        # heart_rate = 60 / rr_intervals  # Перетворення в удари за хвилину (BPM)

        # Initial moments of the third order
        initialMomentsThirdOrder = [np.sum(np.array(i) ** 3) / len(i) for i in data]
        # Initial moments of the fourth order
        initialMomentsFourthOrder = [np.sum(np.array(i) ** 4) / len(i) for i in data]
        # Variance
        variance = [sum((data[i] - mathematicalExpectation[i]) ** 2) / len(data[i]) for i in
                    range(len(mathematicalExpectation))]
        # Central moment functions of the fourth order
        centralMomentFunctionsFourthOrder = [sum((data[i] - mathematicalExpectation[i]) ** 4) / len(data[i]) for i in
                                             range(len(mathematicalExpectation))]

        def to_data_points(raw_list):
            time = np.arange(0, len(raw_list), 1) / self.sampling_rate
            points = list()
            for i in range(len(raw_list)):
                points.append([time[i], raw_list[i]])
            return points

        def stats_matrix_to_points(matrix):
            data = []
            for row in matrix:
                data.extend(row)
            data_time = np.linspace(0, len(data), len(data))
            output_points = []
            for i in range(len(data)):
                output_points.append([data_time[i], data[i]])

            return output_points

        return {
            "mathematical_expectation": to_data_points(mathematicalExpectation),
            "initial_moments_second_order": to_data_points(initialMomentsSecondOrder),
            "initial_moments_third_order": to_data_points(initialMomentsThirdOrder),
            "initial_moments_fourth_order": to_data_points(initialMomentsFourthOrder),
            "central_moment_functions_fourth_order": to_data_points(centralMomentFunctionsFourthOrder),
            "variance": to_data_points(variance),
            "rhythm": self.rhythm_points,
        }