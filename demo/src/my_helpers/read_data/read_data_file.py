from loguru import logger
import pandas as pd
from pathlib import Path
from my_helpers.read_data.read_mat_file import ReadMatFile
from my_helpers.read_data.read_physionet_file import ReadPhysionetFile
from my_helpers.read_data.read_xls_file import ReadXLSFile

class ReadDataFile:

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        data_type = ecg_config.getDataType()
        if data_type == 'xlsx':
            self.instance = ReadXLSFile(ecg_config)
        elif data_type == 'physionet':
            self.instance = ReadPhysionetFile(ecg_config)
        elif data_type == 'mat':
            self.instance = ReadMatFile(ecg_config)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)

    def getData(self):
        # fr_path = f'{self.ecg_config.getFrPath()}/{self.getSigNameDir()}.csv'
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        fr_path = f'{path}/FR-{self.ecg_config.getConfigBlock()}.csv'
        if not Path(fr_path).is_file():
            e = 'The rhythm function file %s does not exist' % fr_path
            logger.error(e)
            raise FileNotFoundError(e)
        
        ecg_fr = pd.read_csv(fr_path)
        self.ECG_T_Peaks = ecg_fr["ECG_T_Peaks"]
        self.ECG_P_Peaks = ecg_fr["ECG_P_Peaks"]
        self.ECG_R_Peaks = ecg_fr["ECG_R_Peaks"]

        self.Q_S_exist = ("ECG_Q_Peaks" in ecg_fr and "ECG_S_Peaks" in ecg_fr)

        matrix_P_R = []
        matrix_R_T = []
        matrix_T_P = []

        for i in range(len(self.ECG_P_Peaks) - 1):
            start = int((self.ECG_P_Peaks[i]) * self.sampling_rate)
            end = int((self.ECG_R_Peaks[i]) * self.sampling_rate)
            matrix_P_R.append(self.signals[self.ecg_config.getSigName()][start:end])
            start = int((self.ECG_R_Peaks[i]) * self.sampling_rate)
            end = int((self.ECG_T_Peaks[i]) * self.sampling_rate)
            matrix_R_T.append(self.signals[self.ecg_config.getSigName()][start:end])
            start = int((self.ECG_T_Peaks[i]) * self.sampling_rate)
            end = int((self.ECG_P_Peaks[i + 1]) * self.sampling_rate)
            matrix_T_P.append(self.signals[self.ecg_config.getSigName()][start:end])

        self.matrix_T_P = matrix_T_P
        self.matrix_P_R = matrix_P_R
        self.matrix_R_T = matrix_R_T

        if self.Q_S_exist:
            self.ECG_Q_Peaks = ecg_fr["ECG_Q_Peaks"]
            self.ECG_S_Peaks = ecg_fr["ECG_S_Peaks"]

    def getSigNameDir(self):
        return f'{self.ecg_config.getConfigBlock()}'