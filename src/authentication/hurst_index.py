from get_config.ecg_config import ECGConfig
from loguru import logger
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from my_helpers.data_preparation import DataPreparation
import numpy as np
import pandas as pd
from scipy.stats import linregress


class HurstIndex():

    def __init__(self, ecg_config):
        logger.debug("Hurst Index")
        self.ecg_config = ecg_config
        self.a_path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Hurst Index'
        Path(self.a_path).mkdir(parents=True, exist_ok=True)

        self.ecg_data = np.array(DataPreparation(ecg_config).getPreparedData())

        self.getHurstIndex()


    def getHurstIndex(self):
        logger.info("Get Hurst Index")
        # ecg_data = self.ecg_data
        ecg_data = np.mean(self.ecg_data, axis=0)
        ecg_data = np.power(ecg_data, 3)

        N = 360
        
        mean_value = np.mean(ecg_data)
        time_series = ecg_data
        Z = np.cumsum(time_series - mean_value)
        R = max(Z) - min(Z)
        S = np.sqrt(np.mean((time_series - mean_value)**2))

        for a in [0.5, 1.0, 1.5]:
            H = np.log(R / S) / (np.log(a * N))
            print(np.round(H, 4))
        
        
        # data_matrix = np.transpose(self.ecg_data)
        # # self.hurst_exponent(data_matrix[0], 0.5)

        # print(len(np.mean(self.ecg_data, axis=1)))
        # print(np.mean(self.ecg_data[1]))

        # for a in [0.5, 1.0, 1.5]:
        #     H = self.hurst_exponent(np.mean(self.ecg_data, axis=1), a)
        #     print(np.round(H, 4))

        # for a in [0.5, 1.0, 1.5]:
        #     H = [self.hurst_exponent(data, a) for data in data_matrix]
        #     self.plot_to_png(self.a_path, H, f"_Hurst channel - {self.ecg_config.getSigName() + 1} a - {str(a).replace('.', '_')} ")

        # self.plot_to_png(self.a_path, self.ecg_data[3], f"__Hurst channel - {self.ecg_config.getSigName() + 1}")

    def hurst_exponent(self, time_series, a):
        N = len(time_series)
        mean_value = np.mean(time_series)
        Z = np.cumsum(time_series - mean_value)
        R = max(Z) - min(Z)
        S = np.sqrt(np.mean((time_series - mean_value)**2))
        H = np.log(R / S) / (np.log(a * N))
        return H

    def plot_to_png(self, path, plot, name, xlim = None, ylim = None, ytext="", xtext="", size=(12, 6)):
        logger.info(f"Plot {name}.png")
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        time = np.arange(0, len(plot), 1) / 360
        axis.plot(time, plot, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.set_title(ytext, loc = 'left', fontsize=14, position=(-0.06, 0))
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)