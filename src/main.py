from loguru import logger
from get_config.ecg_config import ECGConfig
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.data_preparation import DataPreparation
from my_helpers.fourier_series import FourierSeries
from my_helpers.mathematical_statistics import MathematicalStatistics
import sys
import argparse

if __name__ == '__main__':
    logger.info("ECG Research")
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('gen-fr', 'plot-fr', 'main', 'fourier-series-demonstration', 'fourier-series-test-1'))
    parser.add_argument('-c', type=str, required=True)
    parser.add_argument('-d', type=str)
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
    if a.action == "fourier-series-test-1":
        data = DataPreparation(ecg_config)
        data.plotAllCycles()
        FourierSeries(ecg_config, data).getFourierSpectrum([0, 1, 2, 3, 4, 5, 100, 101, 102])

    if a.action == "main":
        config_block_2 = a.d or config_block
        logger.debug("Read config file 2")
        ecg_config_2 = ECGConfig(config_block_2)
        logger.debug(ecg_config_2)
        data = DataPreparation(ecg_config)
        data_2 = DataPreparation(ecg_config_2)
        statistics = MathematicalStatistics(data.getPreparedData())
        statistics_2 = MathematicalStatistics(data_2.getPreparedData())
        statistics.setSamplingRate(data.getModSamplingRate())
        statistics_2.setSamplingRate(data.getModSamplingRate())
        statistics_mean = statistics.getMean(statistics_2)
        statistics_fourier_mean = statistics.getFourierMean(statistics_2)

        print("The average value of the moment function")
        print("Mathematical expectation")
        print("%.5f" % statistics_mean.getMathematicalExpectation())
        print("Initial moments of the second order")
        print("%.5f" % statistics_mean.getInitialMomentsSecondOrder())
        print("Initial moments of the third order")
        print("%.5f" % statistics_mean.getInitialMomentsThirdOrder())
        print("Initial moments of the fourth order")
        print("%.5f" % statistics_mean.getInitialMomentsFourthOrder())
        print("Variance")
        print("%.5f" % statistics_mean.getVariance())
        print("Central moment functions of the fourth order")
        print("%.5f" % statistics_mean.getCentralMomentFunctionsFourthOrder())
        print()
        print("The average value of the spectral components of the moment function")
        print("Mathematical expectation")
        print("%.5f" % statistics_fourier_mean.getMathematicalExpectation())
        print("Initial moments of the second order")
        print("%.5f" % statistics_fourier_mean.getInitialMomentsSecondOrder())
        print("Initial moments of the third order")
        print("%.5f" % statistics_fourier_mean.getInitialMomentsThirdOrder())
        print("Initial moments of the fourth order")
        print("%.5f" % statistics_fourier_mean.getInitialMomentsFourthOrder())
        print("Variance")
        print("%.5f" % statistics_fourier_mean.getVariance())
        print("Central moment functions of the fourth order")
        print("%.5f" % statistics_fourier_mean.getCentralMomentFunctionsFourthOrder())
        



        