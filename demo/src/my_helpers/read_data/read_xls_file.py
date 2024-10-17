from loguru import logger
import pandas as pd

class ReadXLSFile:

    def __init__(self, ecg_config):
        data_path = f'{ecg_config.getXLSPath()}/{ecg_config.getFileName()}.{ecg_config.getDataType()}'
        logger.info("Read XLS file")
        logger.info(data_path)

        columns = ["'Elapsed time'", "'I'", "'II'", "'ECG1'", "'ECG2'", "'III'", "'V'"]

        excel_data = pd.read_excel(data_path)
        data = pd.DataFrame(excel_data, columns=columns).to_numpy()
        self.signals = data.transpose()

        # print(self.signals)

        self.sampling_rate = 1 / (self.signals[0][1] - self.signals[0][0])
        logger.info(f'Fileds: {columns}')
        logger.info(f'Sampling rate: {self.sampling_rate}')
