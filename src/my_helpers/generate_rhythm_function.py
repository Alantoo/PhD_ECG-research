from loguru import logger
from my_helpers.read_data.read_data_file import ReadDataFile
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class GenerateRhythmFunction(ReadDataFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)

    def genFr(self):
        logger.info("Get ECG Peaks")
        i = self.ecg_config.getSigName()

        _, rpeaks = nk.ecg_peaks(self.signals[i], sampling_rate=self.sampling_rate)
        _, waves = nk.ecg_delineate(self.signals[i], rpeaks, sampling_rate=self.sampling_rate, method="cwt")
        ECG_P_Peaks = list(np.array(waves["ECG_P_Peaks"]) / self.sampling_rate)
        ECG_Q_Peaks = list(np.array(waves["ECG_Q_Peaks"]) / self.sampling_rate)
        ECG_R_Peaks = list(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate)
        ECG_S_Peaks = list(np.array(waves["ECG_S_Peaks"]) / self.sampling_rate)
        ECG_T_Peaks = list(np.array(waves["ECG_T_Peaks"]) / self.sampling_rate)        

        ecg_fr = pd.DataFrame({"ECG_P_Peaks" : ECG_P_Peaks, "ECG_Q_Peaks" : ECG_Q_Peaks, "ECG_R_Peaks" : ECG_R_Peaks, "ECG_S_Peaks" : ECG_S_Peaks, "ECG_T_Peaks" : ECG_T_Peaks})
        nk.write_csv(ecg_fr, f'{self.ecg_config.getFrPath()}/{self.ecg_config.getConfigBlock()}.csv')

    def genIntervals(self):
        logger.info("Gen All ECG Intervals")
        i = self.ecg_config.getSigName()

        _, rpeaks = nk.ecg_peaks(self.signals[i], sampling_rate=self.sampling_rate)
        _, waves = nk.ecg_delineate(self.signals[i], rpeaks, sampling_rate=self.sampling_rate)
        
        ECG_P_Onsets = list(np.array(waves["ECG_P_Onsets"]) / self.sampling_rate)
        ECG_P_Peaks = list(np.array(waves["ECG_P_Peaks"]) / self.sampling_rate)
        ECG_P_Offsets = list(np.array(waves["ECG_P_Offsets"]) / self.sampling_rate)
        # ECG_R_Onsets = list(np.array(waves["ECG_R_Onsets"]) / self.sampling_rate)
        ECG_Q_Peaks = list(np.array(waves["ECG_Q_Peaks"]) / self.sampling_rate)
        ECG_R_Peaks = list(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate)
        ECG_S_Peaks = list(np.array(waves["ECG_S_Peaks"]) / self.sampling_rate)
        # ECG_R_Offsets = list(np.array(waves["ECG_R_Offsets"]) / self.sampling_rate)
        ECG_T_Onsets = list(np.array(waves["ECG_T_Onsets"]) / self.sampling_rate)
        ECG_T_Peaks = list(np.array(waves["ECG_T_Peaks"]) / self.sampling_rate)
        ECG_T_Offsets = list(np.array(waves["ECG_T_Offsets"]) / self.sampling_rate)

        ecg_fr = pd.DataFrame({"ECG_P_Onsets" : ECG_P_Onsets, "ECG_P_Peaks" : ECG_P_Peaks, "ECG_P_Offsets" : ECG_P_Offsets, 
                               "ECG_Q_Peaks" : ECG_Q_Peaks,
                               "ECG_R_Peaks" : ECG_R_Peaks,
                               "ECG_S_Peaks" : ECG_S_Peaks,
                               "ECG_T_Onsets" : ECG_T_Onsets, "ECG_T_Peaks" : ECG_T_Peaks, "ECG_T_Offsets" : ECG_T_Offsets})
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Intervals'
        Path(path).mkdir(parents=True, exist_ok=True)
        nk.write_csv(ecg_fr, f'{path}/All.csv')

    def getIntervals(self):
        logger.info("To file ECG Intervals") 
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Intervals'
        intervals_path = f'{path}/All.csv'
        if not Path(intervals_path).is_file():
            e = 'The rhythm function file %s does not exist' % intervals_path
            logger.error(e)
            raise FileNotFoundError(e)
        
        intervals_fr = pd.read_csv(intervals_path)
        ECG_P_Onsets = intervals_fr["ECG_P_Onsets"]
        ECG_P_Offsets = intervals_fr["ECG_P_Offsets"]
        ECG_Q_Peaks = intervals_fr["ECG_Q_Peaks"]
        ECG_S_Peaks = intervals_fr["ECG_S_Peaks"]
        ECG_T_Onsets = intervals_fr["ECG_T_Onsets"]
        ECG_T_Offsets = intervals_fr["ECG_T_Offsets"]

        P = ECG_P_Offsets - ECG_P_Onsets
        P_df = pd.DataFrame({"Y" : round(P, 4)})
        P_df.index = np.arange(1, len(P_df) + 1)
        P_df.index.name = "X"
        nk.write_csv(P_df, f'{path}/P.csv')

        PQ = ECG_Q_Peaks - ECG_P_Offsets
        PQ_df = pd.DataFrame({"Y" : round(PQ, 4)})
        PQ_df.index = np.arange(1, len(PQ_df) + 1)
        PQ_df.index.name = "X"
        nk.write_csv(PQ_df, f'{path}/pq.csv')

        QRS = ECG_S_Peaks - ECG_Q_Peaks
        QRS_df = pd.DataFrame({"Y" : round(QRS, 4)})
        QRS_df.index = np.arange(1, len(QRS_df) + 1)
        QRS_df.index.name = "X"
        nk.write_csv(QRS_df, f'{path}/qRs.csv')

        T = ECG_T_Offsets - ECG_T_Onsets
        T_df = pd.DataFrame({"Y" : round(T, 4)})
        T_df.index = np.arange(1, len(T_df) + 1)
        T_df.index.name = "X"
        nk.write_csv(T_df, f'{path}/T.csv')

        ECG_P_Onsets_arr = ECG_P_Onsets.to_numpy()
        ECG_T_Offsets_arr = ECG_T_Offsets.to_numpy()

        TP = ECG_P_Onsets_arr[1:] - ECG_T_Offsets_arr[:-1]
        TP_df = pd.DataFrame({"Y" : np.round(TP, 4)})
        TP_df.index = np.arange(1, len(TP_df) + 1)
        TP_df.index.name = "X"
        nk.write_csv(TP_df, f'{path}/tp.csv')


    def plotECG(self):
        logger.info("Plot ECG")
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Intervals'
        Path(path).mkdir(parents=True, exist_ok=True)

        start = 0
        end = 520
        ECG_data = np.round(self.signals[self.ecg_config.getSigName()][start:end], 4)
        time = np.arange(0, len(ECG_data), 1) / self.sampling_rate

        data = pd.DataFrame({"ECG_data" : ECG_data})
        data.index = time
        data.index.name = "Time"
        nk.write_csv(data, f'{path}/ECG_data.csv')

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.plot(time, ECG_data, linewidth=3)
        axis.set_xlabel("$t, s$", loc = 'right')
        # axis.legend(['$T(t, 1), s$'])
        # axis.axis(ymin = 0, ymax = 1.2)
        # axis.axis(xmin = 0)
        plt.savefig(f'{path}/ECG_data.png', dpi=300)


    def plotFr(self, debug = False):
        logger.info("Plot a rhythm function")

        self.getData()

        plot_path = f'{self.ecg_config.getFrImgPath()}/{self.ecg_config.getConfigBlock()}'
        Path(plot_path).mkdir(parents=True, exist_ok=True)



        B_i = len(self.ECG_P_Peaks) - 2

        T1_ECG_T_Peaks = []
        T1_ECG_P_Peaks = []
        T1_ECG_R_Peaks = []
        T1_ECG_S_Peaks = []
        T1_ECG_Q_Peaks = []
        T1_X = []
        T1_Y = []

        for i in range(len(self.ECG_T_Peaks)-1):
            T1_ECG_T_Peaks.append(round(self.ECG_T_Peaks[i+1] - self.ECG_T_Peaks[i], 2))

        for i in range(len(self.ECG_P_Peaks)-1):
            T1_ECG_P_Peaks.append(round(self.ECG_P_Peaks[i+1] - self.ECG_P_Peaks[i], 2))

        for i in range(len(self.ECG_R_Peaks)-1):
            T1_ECG_R_Peaks.append(round(self.ECG_R_Peaks[i+1] - self.ECG_R_Peaks[i], 2))

        if self.Q_S_exist:
            for i in range(len(self.ECG_S_Peaks)-1):
                T1_ECG_S_Peaks.append(round(self.ECG_S_Peaks[i+1] - self.ECG_S_Peaks[i], 2))

            for i in range(len(self.ECG_Q_Peaks)-1):
                T1_ECG_Q_Peaks.append(round(self.ECG_Q_Peaks[i+1] - self.ECG_Q_Peaks[i], 2))

        for i in range(len(self.ECG_P_Peaks) - 1):
            T1_X.append(self.ECG_P_Peaks[i])
            # if i == B_i :
            #     break
            if self.Q_S_exist:
                T1_X.append(self.ECG_Q_Peaks[i])
            T1_X.append(self.ECG_R_Peaks[i])
            if self.Q_S_exist:
                T1_X.append(self.ECG_S_Peaks[i])
            T1_X.append(self.ECG_T_Peaks[i])

        for i in range(len(T1_ECG_P_Peaks)):
            T1_Y.append(T1_ECG_P_Peaks[i])
            # if i == B_i :
            #     break
            if self.Q_S_exist:
                T1_Y.append(T1_ECG_Q_Peaks[i])
            T1_Y.append(T1_ECG_R_Peaks[i])
            if self.Q_S_exist:
                T1_Y.append(T1_ECG_S_Peaks[i])
            T1_Y.append(T1_ECG_T_Peaks[i])

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(10, 6)
        axis.grid(True)
        axis.plot(T1_X, T1_Y, linewidth=1)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.legend(['$T(t, 1), s$'])
        axis.axis(ymin = 0, ymax = 1.2)
        axis.axis(xmin = 0)
        plt.savefig(f'{plot_path}/FR.png', dpi=300)

        if self.Q_S_exist:
            fr = pd.DataFrame({"FR_ECG_P" : T1_ECG_P_Peaks, "FR_ECG_Q" : T1_ECG_Q_Peaks, "FR_ECG_R" : T1_ECG_R_Peaks, "FR_ECG_S" : T1_ECG_S_Peaks, "FR_ECG_T" : T1_ECG_T_Peaks})
            nk.write_csv(fr, f'{plot_path}/FR.csv')