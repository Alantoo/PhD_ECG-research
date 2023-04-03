from loguru import logger
from my_helpers.read_physionet_file import ReadPhysionetFile
import numpy as np
import scipy.interpolate as interp
from pathlib import Path
import matplotlib.pyplot as plt

class DataPreparation(ReadPhysionetFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)
        self.getData()

        self.mod_sampling_rate = self.sampling_rate * self.ecg_config.getMultiplier()

        matrix_T_P_size = self.getNewMatrixSize(self.matrix_T_P)
        matrix_P_R_size = self.getNewMatrixSize(self.matrix_P_R)
        matrix_R_T_size = self.getNewMatrixSize(self.matrix_R_T)

        interp_matrix_T_P = []
        interp_matrix_P_R = []
        interp_matrix_R_T = []

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

        self.interp_matrix_all = np.concatenate((interp_matrix_P_R[:-1], interp_matrix_R_T[:-1], interp_matrix_T_P[1:]), axis=1)

    def plotAllCycles(self):
        plot_path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getFileName()}_{self.fileds["sig_name"][self.ecg_config.getSigName()]}'

        Path(plot_path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(17, 4)
        axis.grid(True)
        for i in self.interp_matrix_all:
            axis.plot(i, linewidth=2)
        plt.savefig(f'{plot_path}/1_All_Cycles.png', dpi=300)


    def getNewMatrixSize(self, matrix):
        n = 0
        for i in range(len(matrix)):
            n = n + len(matrix[i])
        n = int((n / len(matrix)) * self.ecg_config.getMultiplier())
        return n