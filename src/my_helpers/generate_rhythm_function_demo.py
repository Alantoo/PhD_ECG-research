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

        # path_all Шлях де буде збережено функцію ритмк
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        path_all = f'{path}/FR-{self.ecg_config.getConfigBlock()}.csv'
        Path(path).mkdir(parents=True, exist_ok=True)
        if Path(path_all).is_file():
            logger.warning(f'File {path_all} is exist. Create a backup and continue')
            return
        
        # Номер відведення
        i = self.ecg_config.getSigName()

        # Використовуємо бубліотеку NeuroKit2 для пошуку зупців
        _, rpeaks = nk.ecg_peaks(self.signals[i], sampling_rate=self.sampling_rate)
        _, waves = nk.ecg_delineate(self.signals[i], rpeaks, sampling_rate=self.sampling_rate)
        ECG_P_Peaks = list(np.round(np.array(waves["ECG_P_Peaks"]) / self.sampling_rate, 4))
        ECG_Q_Peaks = list(np.round(np.array(waves["ECG_Q_Peaks"]) / self.sampling_rate, 4))
        ECG_R_Peaks = list(np.round(np.array(rpeaks["ECG_R_Peaks"]) / self.sampling_rate, 4))
        ECG_S_Peaks = list(np.round(np.array(waves["ECG_S_Peaks"]) / self.sampling_rate, 4))
        ECG_T_Peaks = list(np.round(np.array(waves["ECG_T_Peaks"]) / self.sampling_rate, 4))        

        ecg_fr = pd.DataFrame({"ECG_P_Peaks" : ECG_P_Peaks, "ECG_Q_Peaks" : ECG_Q_Peaks, "ECG_R_Peaks" : ECG_R_Peaks, "ECG_S_Peaks" : ECG_S_Peaks, "ECG_T_Peaks" : ECG_T_Peaks})
        nk.write_csv(ecg_fr, path_all)
