from loguru import logger
import wfdb

class ReadPhysionetFile:

    def __init__(self, ecg_config):
        data_path = f'{ecg_config.getDataPath()}/{ecg_config.getFileName()}'
        logger.info("Read physionet file")
        logger.info(data_path)

        signals, self.fileds = wfdb.rdsamp(data_path)
        self.signals = signals.transpose()

        self.sampling_rate = self.fileds['fs']
        logger.info(f'Fileds: {self.fileds["sig_name"]}')
        logger.info(f'Sampling rate: {self.sampling_rate}')
        