from loguru import logger
from get_config.ecg_config import ECGConfig
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.data_preparation import DataPreparation
from my_helpers.fourier_series import FourierSeries
import sys
import argparse

if __name__ == '__main__':
    logger.info("ECG Research")
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gen-fr', 'plot-fr', 'main', 'fourier-series-demonstration'))
    parser.add_argument('-c', type=str, required=True)
    parser.add_argument('-s', type=str)
    a = parser.parse_args()
    config_block = a.c
    logger.debug("Read config file")
    ecg_config = ECGConfig(config_block)
    logger.debug(ecg_config)

    if a.action == "gen-fr":
        GenerateRhythmFunction(ecg_config).genFr()
    if a.action == "plot-fr":
        GenerateRhythmFunction(ecg_config).plotFr()
    if a.action == "fourier-series-demonstration":
        data = DataPreparation(ecg_config)
        FourierSeries(ecg_config, data).getFourierSeriesDemonstration(int(a.s or 0))
    if a.action == "main":
        data = DataPreparation(ecg_config)
        data.plotAllCycles()
        FourierSeries(ecg_config, data).getFourierSpectrum([0, 1, 2, 3, 4, 5, 100, 101, 102])
        

        