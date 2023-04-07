from loguru import logger
from my_helpers.read_physionet_file import ReadPhysionetFile
import pandas as pd
import neurokit2 as nk
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class GenerateRhythmFunction(ReadPhysionetFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)

    def genFr(self):
        logger.info("Get ECG Peaks")

        for i in range(2):
            _, rpeaks = nk.ecg_peaks(self.signals[i], sampling_rate=self.sampling_rate)
            _, waves = nk.ecg_delineate(self.signals[i], rpeaks, sampling_rate=self.sampling_rate)
            ECG_T_Peaks = list(np.array(waves["ECG_T_Peaks"]) / self.sampling_rate)
            ECG_P_Peaks = list(np.array(waves["ECG_P_Peaks"]) / self.sampling_rate)
            ECG_R_Peaks = list(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate)
            del ECG_P_Peaks[0]
            del ECG_R_Peaks[0]
            ECG_P_Peaks.append(None)
            ECG_R_Peaks.append(None)
            ecg_fr = pd.DataFrame({"D_c" : ECG_T_Peaks, "D_z_1" : ECG_P_Peaks, "D_z_2" : ECG_R_Peaks})
            nk.write_csv(ecg_fr, f'{self.ecg_config.getFrPath()}/{self.ecg_config.getFileName()}_{self.fileds["sig_name"][i]}.csv')

    def plotFr(self, debug = False):
        logger.info("Plot a rhythm function")

        self.getData()

        plot_path = f'{self.ecg_config.getFrImgPath()}/{self.ecg_config.getFileName()}_{self.fileds["sig_name"][self.ecg_config.getSigName()]}'
        Path(plot_path).mkdir(parents=True, exist_ok=True)

        if debug :
            plt.clf()
            f, axis = plt.subplots(1)
            f.tight_layout()
            f.set_size_inches(17, 4)
            axis.grid(True)
            for i in self.matrix_T_P:
                axis.plot(i, linewidth=2)
            plt.savefig(f'{plot_path}/1_matrix_T_P.png', dpi=300)

            plt.clf()
            f, axis = plt.subplots(1)
            f.tight_layout()
            f.set_size_inches(17, 4)
            axis.grid(True)
            for i in self.matrix_P_R:
                axis.plot(i, linewidth=2)
            plt.savefig(f'{plot_path}/2_matrix_P_R.png', dpi=300)

            plt.clf()
            f, axis = plt.subplots(1)
            f.tight_layout()
            f.set_size_inches(17, 4)
            axis.grid(True)
            for i in self.matrix_R_T:
                axis.plot(i, linewidth=2)
            plt.savefig(f'{plot_path}/3_matrix_R_T.png', dpi=300)

        B_i = len(self.D_z_1) -1

        T1_D_c = []
        T1_D_z_1 = []
        T1_D_z_2 = []
        T1_X = []
        T1_Y = []

        for i in range(len(self.D_c)-1):
            T1_D_c.append(round(self.D_c[i+1] - self.D_c[i], 2))

        for i in range(len(self.D_z_1)-1):
            T1_D_z_1.append(round(self.D_z_1[i+1] - self.D_z_1[i], 2))

        for i in range(len(self.D_z_2)-1):
            T1_D_z_2.append(round(self.D_z_2[i+1] - self.D_z_2[i], 2))

        for i in range(len(self.D_c)):
            T1_X.append(self.D_c[i])
            if i == B_i :
                break
            T1_X.append(self.D_z_1[i])
            T1_X.append(self.D_z_2[i])

        for i in range(len(T1_D_c)):
            T1_Y.append(T1_D_c[i])
            if i == B_i :
                break
            T1_Y.append(T1_D_z_1[i])
            T1_Y.append(T1_D_z_2[i])

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(10, 6)
        axis.grid(True)
        axis.plot(T1_X, T1_Y, linewidth=3)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.legend(['$T(t, 1), s$'])
        axis.axis(ymin = -0.2, ymax = 1.2)
        axis.axis(xmin = 0)
        plt.savefig(f'{plot_path}/FR.png', dpi=300)