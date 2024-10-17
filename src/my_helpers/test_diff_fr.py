from loguru import logger
from my_helpers.read_data.read_data_file import ReadDataFile
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class TestDiffFr(ReadDataFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)

    def getTestData(self):
        logger.info("Get test data")
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Intervals'
        
        # self.testDataToFile(path, "Generated")
        # self.testDataToFile(path, "Reference")
        # g_P_Onsets, g_Q_Peaks, g_R_Peaks, g_S_Peaks, g_T_Onsets = self.testDataFromFile(path, "Generated_data")
        # r_P_Onsets, r_Q_Peaks, r_R_Peaks, r_S_Peaks, r_T_Onsets = self.testDataFromFile(path, "Reference_data")

        # self.testPlot(path, "1_P_Onsets", g_P_Onsets, r_P_Onsets)
        # self.testPlot(path, "2_Q_Peaks", g_Q_Peaks, r_Q_Peaks)
        # self.testPlot(path, "3_R_Peaks", g_R_Peaks, r_R_Peaks)
        # self.testPlot(path, "4_S_Peaks", g_S_Peaks, r_S_Peaks)
        # self.testPlot(path, "5_T_Onsets", g_T_Onsets, r_T_Onsets)

        #FR
        g_P_Onsets, g_Q_Peaks, g_R_Peaks, g_S_Peaks, g_T_Onsets = self.testDataFromFile(path, "Generated_time")
        r_P_Onsets, r_Q_Peaks, r_R_Peaks, r_S_Peaks, r_T_Onsets = self.testDataFromFile(path, "Reference_time")

        Tg_P_Onsets = []
        Tg_Q_Peaks = []
        Tg_R_Peaks = []
        Tg_S_Peaks = []
        Tg_T_Onsets = []
        Tr_P_Onsets = []
        Tr_Q_Peaks = []
        Tr_R_Peaks = []
        Tr_S_Peaks = []
        Tr_T_Onsets = []

        for i in range(len(g_P_Onsets)-1):
            Tg_P_Onsets.append(round(g_P_Onsets[i+1] - g_P_Onsets[i], 3))

        for i in range(len(g_Q_Peaks)-1):
            Tg_Q_Peaks.append(round(g_Q_Peaks[i+1] - g_Q_Peaks[i], 3))

        for i in range(len(g_R_Peaks)-1):
            Tg_R_Peaks.append(round(g_R_Peaks[i+1] - g_R_Peaks[i], 3))

        for i in range(len(g_S_Peaks)-1):
            Tg_S_Peaks.append(round(g_S_Peaks[i+1] - g_S_Peaks[i], 3))

        for i in range(len(g_T_Onsets)-1):
            Tg_T_Onsets.append(round(g_T_Onsets[i+1] - g_T_Onsets[i], 3))

        T1_X = []
        T1_Y = []

        for i in range(len(r_P_Onsets)-1):
            Tr_P_Onsets.append(round(r_P_Onsets[i+1] - r_P_Onsets[i], 3))

        for i in range(len(r_Q_Peaks)-1):
            Tr_Q_Peaks.append(round(r_Q_Peaks[i+1] - r_Q_Peaks[i], 3))

        for i in range(len(r_R_Peaks)-1):
            Tr_R_Peaks.append(round(r_R_Peaks[i+1] - r_R_Peaks[i], 3))

        for i in range(len(r_S_Peaks)-1):
            Tr_S_Peaks.append(round(r_S_Peaks[i+1] - r_S_Peaks[i], 3))

        for i in range(len(r_T_Onsets)-1):
            Tr_T_Onsets.append(round(r_T_Onsets[i+1] - r_T_Onsets[i], 3))

        T2_X = []
        T2_Y = []

        for i in range(len(g_P_Onsets) - 1):
            T1_X.append(g_P_Onsets[i])
            T1_X.append(g_Q_Peaks[i])
            T1_X.append(g_R_Peaks[i])
            T1_X.append(g_S_Peaks[i])
            T1_X.append(g_T_Onsets[i])

        for i in range(len(Tg_P_Onsets)):
            T1_Y.append(Tg_P_Onsets[i])
            T1_Y.append(Tg_Q_Peaks[i])
            T1_Y.append(Tg_R_Peaks[i])
            T1_Y.append(Tg_S_Peaks[i])
            T1_Y.append(Tg_T_Onsets[i])

        for i in range(len(r_P_Onsets) - 1):
            T2_X.append(r_P_Onsets[i])
            T2_X.append(r_Q_Peaks[i])
            T2_X.append(r_R_Peaks[i])
            T2_X.append(r_S_Peaks[i])
            T2_X.append(r_T_Onsets[i])

        for i in range(len(Tr_P_Onsets)):
            T2_Y.append(Tr_P_Onsets[i])
            T2_Y.append(Tr_Q_Peaks[i])
            T2_Y.append(Tr_R_Peaks[i])
            T2_Y.append(Tr_S_Peaks[i])
            T2_Y.append(Tr_T_Onsets[i])

        res = np.mean((np.array(T1_Y) - np.array(T2_Y))**2)
        res2 = np.mean(np.array(T2_Y)**2)
        res3 = (res/res2)*100

        print(res3)

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)

        axis.plot(T1_X, T1_Y, "o", linewidth=3, label="Generated")
        axis.plot(T2_X, T2_Y, "o", linewidth=3, label="Reference")
        axis.legend(loc='best',prop={'size':16})
        axis.axis(ymin = 0, ymax = 1.2)
        plt.savefig(f'{path}/Test.png', dpi=300)

        data_fr = pd.DataFrame({"X" : np.round(T1_X, 4), "Y" : T1_Y})
        nk.write_csv(data_fr, f'{path}/Rhythm function Generated.csv')

        data_fr = pd.DataFrame({"X" : np.round(T2_X, 4), "Y" : T2_Y})
        nk.write_csv(data_fr, f'{path}/Rhythm function Reference.csv')



    def testPlot(self, path, filename, g_set, r_set):
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)

        axis.plot(g_set, "o", linewidth=3, label="Generated")
        axis.plot(r_set, "o", linewidth=3, label="Reference")
        axis.legend(loc='best',prop={'size':16})
        axis.axis(ymin = 0, ymax = 2)
        plt.savefig(f'{path}/{filename}.png', dpi=300)


    def testDataFromFile(self, path, tag):
        logger.info(f'Get {tag} test data')
        test_path = f'{path}/{tag}.csv'
        if not Path(test_path).is_file():
            e = 'The %s rhythm function file %s does not exist' % (tag, test_path)
            logger.error(e)
            raise FileNotFoundError(e)
        test_fr = pd.read_csv(test_path)
        P_Onsets = test_fr["ECG_P_Onsets"]
        Q_Peaks = test_fr["ECG_Q_Peaks"]
        R_Peaks = test_fr["ECG_R_Peaks"]
        S_Peaks = test_fr["ECG_S_Peaks"]
        T_Onsets = test_fr["ECG_T_Onsets"]
        return P_Onsets, Q_Peaks, R_Peaks, S_Peaks, T_Onsets

    def testDataToFile(self, path, tag):
        logger.info(f'Get {tag} test data')
        test_path = f'{path}/{tag}_time.csv'
        if not Path(test_path).is_file():
            e = 'The %s rhythm function file %s does not exist' % (tag, test_path)
            logger.error(e)
            raise FileNotFoundError(e)
        
        test_fr = pd.read_csv(test_path)
        ECG_P_Onsets = test_fr["ECG_P_Onsets"]
        ECG_Q_Peaks = test_fr["ECG_Q_Peaks"]
        ECG_R_Peaks = test_fr["ECG_R_Peaks"]
        ECG_S_Peaks = test_fr["ECG_S_Peaks"]
        ECG_T_Onsets = test_fr["ECG_T_Onsets"]

        n_ECG_P_Onsets = []
        n_ECG_Q_Peaks = []
        n_ECG_R_Peaks = []
        n_ECG_S_Peaks = []
        n_ECG_T_Onsets = []

        for P_Onset, Q_Peak, R_Peak, S_Peak, T_Onset in zip(ECG_P_Onsets, ECG_Q_Peaks, ECG_R_Peaks, ECG_S_Peaks, ECG_T_Onsets):
            n_ECG_P_Onsets.append(self.signals[self.ecg_config.getSigName()][int(P_Onset * self.sampling_rate)])
            n_ECG_Q_Peaks.append(self.signals[self.ecg_config.getSigName()][int(Q_Peak * self.sampling_rate)])
            n_ECG_R_Peaks.append(self.signals[self.ecg_config.getSigName()][int(R_Peak * self.sampling_rate)])
            n_ECG_S_Peaks.append(self.signals[self.ecg_config.getSigName()][int(S_Peak * self.sampling_rate)])
            n_ECG_T_Onsets.append(self.signals[self.ecg_config.getSigName()][int(T_Onset * self.sampling_rate)])

        data_fr = pd.DataFrame({"ECG_P_Onsets" : n_ECG_P_Onsets, "ECG_Q_Peaks" : n_ECG_Q_Peaks, "ECG_R_Peaks" : n_ECG_R_Peaks,
                               "ECG_S_Peaks" : n_ECG_S_Peaks, "ECG_T_Onsets" : n_ECG_T_Onsets})
        nk.write_csv(data_fr, f'{path}/{tag}_data.csv')