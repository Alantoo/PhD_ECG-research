from loguru import logger
from get_config.ecg_config import ECGConfig
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.data_preparation import DataPreparation
import sys
import argparse

if __name__ == '__main__':
    logger.info("ECG Research")
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gen-fr', 'plot-fr', 'main'))
    parser.add_argument('-c', type=str, required=True)
    a = parser.parse_args()
    config_block = a.c
    logger.debug("Read config file")
    ecg_config = ECGConfig(config_block)
    logger.debug(ecg_config)

    if a.action == "gen-fr":
        GenerateRhythmFunction(ecg_config).genFr()
    if a.action == "plot-fr":
        GenerateRhythmFunction(ecg_config).plotFr()
    if a.action == "main":
        data = DataPreparation(ecg_config)
        data.plotAllCycles()
        