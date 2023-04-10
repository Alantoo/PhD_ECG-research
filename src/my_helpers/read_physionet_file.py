from loguru import logger
import wfdb
import pandas as pd
from pathlib import Path

class ReadPhysionetFile:

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        data_path = f'{self.ecg_config.getDataPath()}/{self.ecg_config.getFileName()}'
        logger.info("Read physionet file")
        logger.info(data_path)

        signals, self.fileds = wfdb.rdsamp(data_path)
        self.signals = signals.transpose()

        self.sampling_rate = self.fileds['fs']
        logger.info(f'Fileds: {self.fileds["sig_name"]}')
        logger.info(f'Sampling rate: {self.sampling_rate}')

    def getData(self):
        fr_path = f'{self.ecg_config.getFrPath()}/{self.getSigNameDir()}.csv'
        if not Path(fr_path).is_file():
            e = 'The rhythm function file %s does not exist' % fr_path
            logger.error(e)
            raise FileNotFoundError(e)
        
        ecg_fr = pd.read_csv(fr_path)
        self.D_c = ecg_fr["D_c"]
        self.D_z_1 = ecg_fr["D_z_1"][:-1]
        self.D_z_2 = ecg_fr["D_z_2"][:-1]

        matrix_T_P = []
        matrix_P_R = []
        matrix_R_T = []

        for i in range(len(self.D_z_1)):
            start = int((self.D_c[i]) * self.sampling_rate)
            end = int((self.D_z_1[i]) * self.sampling_rate)
            matrix_T_P.append(self.signals[self.ecg_config.getSigName()][start:end])
            start = int((self.D_z_1[i]) * self.sampling_rate)
            end = int((self.D_z_2[i]) * self.sampling_rate)
            matrix_P_R.append(self.signals[self.ecg_config.getSigName()][start:end])
            start = int((self.D_z_2[i]) * self.sampling_rate)
            end = int((self.D_c[i + 1]) * self.sampling_rate)
            matrix_R_T.append(self.signals[self.ecg_config.getSigName()][start:end])

        self.matrix_T_P = matrix_T_P
        self.matrix_P_R = matrix_P_R
        self.matrix_R_T = matrix_R_T

    def getSigNameDir(self):
        return f'{self.ecg_config.getConfigBlock()}'