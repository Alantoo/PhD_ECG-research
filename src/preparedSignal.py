import numpy as np
import neurokit2 as nk
import pandas as pd
import scipy.interpolate as interp


class PreparedSignal:
    def __init__(self, signal, sampling_rate):
        self.sampling_rate = sampling_rate
        multiplier = 1

        def getNewMatrixSize(matrix):
            n = 0
            for i in range(len(matrix)):
                n = n + len(matrix[i])
            n = int((n / len(matrix)) * multiplier)
            n = int(len(matrix[0]) * multiplier)
            return n

        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=self.sampling_rate)
        _, waves = nk.ecg_delineate(signal, rpeaks, sampling_rate=self.sampling_rate)
        ECG_P_Peaks = list(np.round(np.array(waves["ECG_P_Peaks"]) / self.sampling_rate, 4))
        ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / self.sampling_rate, 4))
        ECG_R_Peaks = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate, 4))
        ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / self.sampling_rate, 4))
        ECG_T_Peaks = list(np.round(np.array(waves["ECG_T_Peaks"]) / self.sampling_rate, 4))

        ecg_fr = pd.DataFrame({"ECG_P_Peaks": ECG_P_Peaks, "ECG_Q_Peaks": ECG_Q_Peaks, "ECG_R_Peaks": ECG_R_Peaks,
                               "ECG_S_Peaks": ECG_S_Peaks, "ECG_T_Peaks": ECG_T_Peaks})
        self.ECG_T_Peaks = ecg_fr["ECG_T_Peaks"]
        self.ECG_P_Peaks = ecg_fr["ECG_P_Peaks"]
        self.ECG_R_Peaks = ecg_fr["ECG_R_Peaks"]

        self.Q_S_exist = ("ECG_Q_Peaks" in ecg_fr and "ECG_S_Peaks" in ecg_fr)

        # Line block
        # T1_ECG_T_Peaks = []
        # T1_ECG_P_Peaks = []
        # T1_ECG_R_Peaks = []
        # T1_Y = []
        # for i in range(len(self.ECG_T_Peaks)-1):
        #     T1_ECG_T_Peaks.append(round(self.ECG_T_Peaks[i+1] - self.ECG_T_Peaks[i], 2))

        # for i in range(len(self.ECG_P_Peaks)-1):
        #     T1_ECG_P_Peaks.append(round(self.ECG_P_Peaks[i+1] - self.ECG_P_Peaks[i], 2))

        # for i in range(len(self.ECG_R_Peaks)-1):
        #     T1_ECG_R_Peaks.append(round(self.ECG_R_Peaks[i+1] - self.ECG_R_Peaks[i], 2))

        # for i in range(len(T1_ECG_P_Peaks)):
        #     T1_Y.append(T1_ECG_P_Peaks[i])
        #     T1_Y.append(T1_ECG_R_Peaks[i])
        #     T1_Y.append(T1_ECG_T_Peaks[i])

        # m = np.mean(T1_Y)

        # self.ECG_T_Peaks = np.arange(self.ECG_T_Peaks.iloc[0], self.ECG_T_Peaks.iloc[-1] - 1, m)
        # self.ECG_R_Peaks = np.arange(self.ECG_R_Peaks.iloc[0], self.ECG_R_Peaks.iloc[-1] - 1, m)
        # self.ECG_P_Peaks = np.arange(self.ECG_P_Peaks.iloc[0], self.ECG_P_Peaks.iloc[-1] - 1, m)

        matrix_P_R = []
        matrix_R_T = []
        matrix_T_P = []


        input_peaks = [ECG_P_Peaks, ECG_Q_Peaks, ECG_R_Peaks, ECG_S_Peaks, ECG_T_Peaks]
        time_signal_rhythm = []
        time_peaks_rhythm = [list() for i in range(len(input_peaks))]
        for i in range(len(ECG_P_Peaks) - 1):
            # curr_signal = signal_data[int(ECG_P_Peaks[i] * sampling_rate):int(ECG_P_Peaks[i + 1] * sampling_rate)]
            # show_plot("Signal", [i for i in range(len(curr_signal))], curr_signal)
            size = 0
            for peak_idx in range(len(input_peaks)):
                def replace_nan(val):
                    if np.isnan(val):
                        return 0
                    return val

                curr_complex = input_peaks[peak_idx]
                next_cmp_idx = peak_idx + 1
                curr_peak_idx = i
                next_peak_idx = i
                if next_cmp_idx == len(input_peaks):
                    next_cmp_idx = 0
                    next_peak_idx = i + 1

                next_complex = input_peaks[next_cmp_idx]
                start_peak = curr_complex[curr_peak_idx]
                start = int(replace_nan(start_peak) * sampling_rate)
                next_peak = next_complex[next_peak_idx]
                if np.isnan(next_peak):
                    next_peak = input_peaks[0][i + 1]

                end = int(replace_nan(next_peak) * sampling_rate)
                complex_slice = signal[start:end]
                curr_duration = len(complex_slice)
                size += curr_duration

                time_peaks_rhythm[peak_idx].append(curr_duration)
                # time_ratio = len(complex_slice) / sampling_rate
                # show_plot("Complex", [i*time_ratio for i in range(len(complex_slice))], complex_slice)
            time_signal_rhythm.append(size)

        for i in range(len(self.ECG_P_Peaks) - 1):
            def replaceNaN(val):
                if np.isnan(val):
                    return 0
                return val

            def appendIfNotNaN(segA, segB, lst):

                start = int(replace_nan(segA[i]) * self.sampling_rate)
                end = int(replace_nan(segB[i]) * self.sampling_rate)

                sig_name = self.ecg_config.getSigName()
                sig = self.signals[sig_name]
                item = sig[start:end]
                lst.append(item)

            def slice(matrixLeft, matrixRight, rate, i):
                start = int(replaceNaN(matrixLeft[i]) * rate)

                end = int(replaceNaN(matrixRight[i]) * rate)
                return signal[start:end]

            # appendIfNotNaN(self.ECG_P_Peaks, self.ECG_R_Peaks, matrix_P_R)
            # appendIfNotNaN(self.ECG_R_Peaks, self.ECG_T_Peaks, matrix_R_T)
            # appendIfNotNaN(self.ECG_T_Peaks, self.ECG_P_Peaks, matrix_T_P)
            start = int(replaceNaN(self.ECG_P_Peaks[i]) * self.sampling_rate)
            end = int(replaceNaN(self.ECG_R_Peaks[i]) * self.sampling_rate)

            pr =  signal[start:end] # slice(self.ECG_P_Peaks, self.ECG_R_Peaks, self.sampling_rate)
            start = int(replaceNaN(self.ECG_R_Peaks[i]) * self.sampling_rate)
            end = int(replaceNaN(self.ECG_T_Peaks[i]) * self.sampling_rate)
            rt = signal[start:end]
            start = int(replaceNaN(self.ECG_T_Peaks[i]) * self.sampling_rate)
            end = int(replaceNaN(self.ECG_P_Peaks[i + 1]) * self.sampling_rate)
            tp = signal[start:end]

            if len(pr) == 0 or len(rt) == 0 or len(tp) == 0:
                continue

            matrix_P_R.append(pr)
            matrix_R_T.append(rt)
            matrix_T_P.append(tp)

        self.matrix_T_P = matrix_T_P
        self.matrix_P_R = matrix_P_R
        self.matrix_R_T = matrix_R_T

        if self.Q_S_exist:
            self.ECG_Q_Peaks = ecg_fr["ECG_Q_Peaks"]
            self.ECG_S_Peaks = ecg_fr["ECG_S_Peaks"]

        self.mod_sampling_rate = int(self.sampling_rate * multiplier)

        matrix_T_P_size = getNewMatrixSize(self.matrix_T_P)
        matrix_P_R_size = getNewMatrixSize(self.matrix_P_R)
        matrix_R_T_size = getNewMatrixSize(self.matrix_R_T)

        # print(matrix_T_P_size)
        # print(matrix_P_R_size)
        # print(matrix_R_T_size)

        matrix_T_P_size = 145
        matrix_P_R_size = 48
        matrix_R_T_size = 83

        interp_matrix_T_P = []
        interp_matrix_P_R = []
        interp_matrix_R_T = []
        self.interp_matrix_all = []

        for i in range(len(self.matrix_T_P)):
            arr = np.array(self.matrix_T_P[i])
            arr_interp = interp.interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_T_P_size))
            interp_matrix_T_P.append(arr_stretch)

        for i in range(len(self.matrix_P_R)):
            arr = np.array(self.matrix_P_R[i])
            arr_interp = interp.interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_P_R_size))
            interp_matrix_P_R.append(arr_stretch)

        for i in range(len(self.matrix_R_T)):
            arr = np.array(self.matrix_R_T[i])
            arr_interp = interp.interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_R_T_size))
            interp_matrix_R_T.append(arr_stretch)

        interp_matrix_all = np.concatenate((interp_matrix_P_R, interp_matrix_R_T, interp_matrix_T_P), axis=1)

        # self.interp_matrix_all = interp_matrix_all

        for i in range(len(interp_matrix_all)):
            arr = np.array(interp_matrix_all[i])
            arr_interp = interp.interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, self.mod_sampling_rate))
            self.interp_matrix_all.append(arr_stretch)

        m_m = np.mean(self.interp_matrix_all, 1)

        data = self.interp_matrix_all - m_m[:, None]

        data = np.transpose(data)
        self.data = data
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

        self.math_stats = {
            "mathematical_expectation": to_data_points(mathematicalExpectation),
            "initial_moments_second_order": to_data_points(initialMomentsSecondOrder),
            "initial_moments_third_order": to_data_points(initialMomentsThirdOrder),
            "initial_moments_fourth_order": to_data_points(initialMomentsFourthOrder),
            "central_moment_functions_fourth_order": to_data_points(centralMomentFunctionsFourthOrder),
            "variance": to_data_points(variance),
            "rhythm": stats_matrix_to_points(time_peaks_rhythm),
        }
