import logging
import numpy as np
import neurokit2 as nk
import pandas as pd
import scipy.interpolate as interp
import wfdb.processing

logger = logging.getLogger(__name__)


class PreparedSignal:
    def __init__(self, signal, sampling_rate, delineation_method='dwt', r_peak_method='xqrs'):
        self.sampling_rate = sampling_rate
        self.signal = signal
        self.multiplier = 1

        cleaned = nk.ecg_clean(signal, sampling_rate=self.sampling_rate)
        if r_peak_method == 'xqrs':
            r_indices = wfdb.processing.xqrs_detect(np.array(cleaned, dtype=float), fs=self.sampling_rate, verbose=False)
        else:
            _, info = nk.ecg_peaks(cleaned, sampling_rate=self.sampling_rate, method=r_peak_method)
            r_indices = info['ECG_R_Peaks']
        rpeaks = {"ECG_R_Peaks": r_indices}
        n_beats = len(rpeaks["ECG_R_Peaks"])

        def _delineate(method):
            _, w = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=self.sampling_rate, method=method)
            return w

        def _align(waves, key):
            raw = waves.get(key, [])
            arr = np.array(raw, dtype=float)
            if len(arr) < n_beats:
                arr = np.concatenate([arr, np.full(n_beats - len(arr), np.nan)])
            elif len(arr) > n_beats:
                arr = arr[:n_beats]
            return arr

        def _merge_waves(waves_a, waves_b, keys):
            """For each key, merge two arrays: use non-NaN value; if both valid, average."""
            merged = {}
            for key in keys:
                a = _align(waves_a, key)
                b = _align(waves_b, key)
                both = ~np.isnan(a) & ~np.isnan(b)
                only_a = ~np.isnan(a) & np.isnan(b)
                only_b = np.isnan(a) & ~np.isnan(b)
                result = np.full(n_beats, np.nan)
                result[both]  = np.round((a[both] + b[both]) / 2)
                result[only_a] = a[only_a]
                result[only_b] = b[only_b]
                merged[key] = result
            return merged

        wave_keys = ["ECG_P_Peaks", "ECG_Q_Peaks", "ECG_S_Peaks", "ECG_T_Peaks",
                     "ECG_P_Onsets", "ECG_P_Offsets", "ECG_T_Onsets", "ECG_T_Offsets"]

        if delineation_method == 'merged':
            waves_dwt = _delineate('dwt')
            waves_cwt = _delineate('cwt')
            merged = _merge_waves(waves_dwt, waves_cwt, wave_keys)
            waves = {k: merged[k].tolist() for k in wave_keys}
            # Stored for consensus computation in the caller
            self.waves_dwt = waves_dwt
            self.waves_cwt = waves_cwt
        else:
            _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=self.sampling_rate, method=delineation_method)
            self.waves_dwt = None
            self.waves_cwt = None

        def to_sec(key):
            raw = waves.get(key, [])
            arr = np.array(raw, dtype=float)
            if len(arr) < n_beats:
                arr = np.concatenate([arr, np.full(n_beats - len(arr), np.nan)])
            elif len(arr) > n_beats:
                arr = arr[:n_beats]
            return list(np.where(np.isnan(arr), np.nan, np.round(arr / self.sampling_rate, 4)))

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
        self.zone_peak_times: dict[str, dict[int, float]] = {
            key: {i: t for i, t in lk.items() if t is not None}
            for key, lk in zip(zone_keys, zone_lookups)
        }

        sig_arr = np.array(self.signal)
        sig_len = len(sig_arr)

        def extract(on_sec, off_sec):
            s = int(float(on_sec) * self.sampling_rate)
            e = int(float(off_sec) * self.sampling_rate)
            if s < 0 or e > sig_len or e <= s:
                return None
            return sig_arr[s:e]

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

    def get_6zone_matrices(self):
        """
        Returns (matrices, rhythm_matrix) for 6 anatomical zones, aligned beat-by-beat.

        Zone indices:
            0 — TP baseline (T_off[i-1] → P_on[i])
            1 — P-wave      (P_on        → P_off)
            2 — PQ segment  (P_off       → Q)
            3 — QRS complex (Q           → S)
            4 — ST segment  (S           → T_on)
            5 — T-wave      (T_on        → T_off)

        Each matrices[z] is a list of numpy arrays (one per valid cycle).
        rhythm_matrix[z] is a list of ints (duration in samples per cycle).
        """
        sig_arr = np.array(self.signal)
        sig_len = len(sig_arr)

        def extract(on_sec, off_sec):
            s = int(float(on_sec) * self.sampling_rate)
            e = int(float(off_sec) * self.sampling_rate)
            if s < 0 or e > sig_len or e <= s:
                return None
            return sig_arr[s:e]

        n         = len(self.ECG_P_Onsets)
        p_on_arr  = np.array(self.ECG_P_Onsets,  dtype=float)
        p_off_arr = np.array(self.ECG_P_Offsets, dtype=float)
        q_arr     = np.array(self.ECG_Q_Peaks,   dtype=float)
        s_arr     = np.array(self.ECG_S_Peaks,   dtype=float)
        t_on_arr  = np.array(self.ECG_T_Onsets,  dtype=float)
        t_off_arr = np.array(self.ECG_T_Offsets, dtype=float)

        matrices      = [[] for _ in range(6)]
        rhythm_matrix = [[] for _ in range(6)]
        skipped = 0

        for i in range(1, n):
            t_off_prev = t_off_arr[i - 1]
            p_on  = p_on_arr[i]
            p_off = p_off_arr[i]
            q     = q_arr[i]
            s     = s_arr[i]
            t_on  = t_on_arr[i]
            t_off = t_off_arr[i]

            fiducials = [t_off_prev, p_on, p_off, q, s, t_on, t_off]
            if any(np.isnan(float(v)) for v in fiducials):
                skipped += 1
                continue

            if not (t_off_prev < p_on < p_off < q < s < t_on < t_off):
                skipped += 1
                continue

            segs = [
                extract(t_off_prev, p_on),  # 0 TP baseline
                extract(p_on,  p_off),       # 1 P-wave
                extract(p_off, q),           # 2 PQ segment
                extract(q,     s),           # 3 QRS complex
                extract(s,     t_on),        # 4 ST segment
                extract(t_on,  t_off),       # 5 T-wave
            ]

            if any(x is None or len(x) < 2 for x in segs):
                skipped += 1
                continue

            for z, seg in enumerate(segs):
                matrices[z].append(seg)
                rhythm_matrix[z].append(len(seg))

        logger.info(
            "get_6zone_matrices: %d valid cycles, %d skipped",
            len(matrices[0]), skipped,
        )
        return matrices, rhythm_matrix

    def get_6zone_stats(self):
        """Per-zone mean and variance for 6-zone modelling.

        Returns (mean_points, variance_points) where each is a list of [index, value]
        pairs with zones 0-5 concatenated in order.  The split boundaries are
        recoverable from rhythm_6zones mean durations (same ordering as get_6zone_matrices).

        Returns (None, None) if the signal has no valid 6-zone cycles.
        """
        matrices, _ = self.get_6zone_matrices()
        if not matrices[0]:
            return None, None

        mean_concat = []
        var_concat  = []

        for z in range(6):
            zone_arrays = matrices[z]
            if not zone_arrays:
                continue
            target_size = len(zone_arrays[0])
            normalized  = []
            for arr in zone_arrays:
                arr = np.array(arr, dtype=float)
                if len(arr) == target_size:
                    normalized.append(arr)
                elif len(arr) >= 2:
                    f = interp.interp1d(np.arange(len(arr)), arr)
                    normalized.append(f(np.linspace(0, len(arr) - 1, target_size)))
                else:
                    normalized.append(np.full(target_size, arr[0] if len(arr) else 0.0))

            mat       = np.array(normalized)          # (n_cycles, target_size)
            mean_concat.extend(np.mean(mat, axis=0).tolist())
            var_concat.extend( np.var( mat, axis=0).tolist())

        n = len(mean_concat)
        return (
            [[i, mean_concat[i]] for i in range(n)],
            [[i, var_concat[i]]  for i in range(n)],
        )

    def get_6zone_rhythm_points(self):
        """
        Returns rhythm as interleaved 6-zone durations (samples) for use with
        /v2/modelling/math_stats (segments_count=6).

        Decoded by: rhythm_values[z::6] gives durations for zone z across all cycles.
        """
        _, rhythm_matrix = self.get_6zone_matrices()
        if not rhythm_matrix[0]:
            return []
        n_cycles = min(len(rm) for rm in rhythm_matrix)
        points   = []
        beat_idx = 1
        for i in range(n_cycles):
            for rm in rhythm_matrix:
                points.append([round(beat_idx / 7, 6), rm[i]])
                beat_idx += 1
        return points

    def get_interpolated_matrix(self):
        mod_sampling_rate = int(self.sampling_rate * self.multiplier)
        interp_matrix = []
        for beat in self.matrix_beat:
            arr = np.array(beat, dtype=float)
            f = interp.interp1d(np.arange(arr.size), arr)
            interp_matrix.append(f(np.linspace(0, arr.size - 1, mod_sampling_rate)).tolist())
        return interp_matrix, mod_sampling_rate

    def get_stats(self):
        # Build full-cycle beats as P → PQ → QRS → ST → T → TP so that the
        # mean waveform includes the trailing TP baseline.
        # Zone 0 of beat i is T_off[i-1]→P_on[i], so the TP that follows beat
        # i is zone 0 of beat i+1 — hence we pair zones 1-5 of beat i with
        # zone 0 of beat i+1, yielding n_cycles-1 complete cycles.
        # Fall back to the old matrix_beat path if 6-zone extraction fails.
        matrices, _ = self.get_6zone_matrices()
        mod_sampling_rate = int(self.sampling_rate * self.multiplier)
        if matrices[0]:
            n_cycles = min(len(m) for m in matrices) - 1  # need next beat's zone 0
            interp_matrix_all = []
            for i in range(n_cycles):
                beat = np.concatenate(
                    [matrices[z][i] for z in range(1, 6)] + [matrices[0][i + 1]]
                )
                f = interp.interp1d(np.arange(beat.size), beat)
                interp_matrix_all.append(
                    f(np.linspace(0, beat.size - 1, mod_sampling_rate)).tolist()
                )
        else:
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
            "rhythm_6zones": self.get_6zone_rhythm_points(),
        }