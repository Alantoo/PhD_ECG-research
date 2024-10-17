from classifiers_test.test import Test
from classifiers_test.test_src import Test as Test_src
from loguru import logger
import pandas as pd
from get_config.ecg_config import ECGConfig
from my_helpers.generate_rhythm_function import GenerateRhythmFunction
from my_helpers.plot_statistics import PlotStatistics
from my_helpers.data_preparation import DataPreparation
from my_helpers.fourier_series import FourierSeries
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.classifiers import Classifiers
from my_helpers.test_classifiers import TestClassifiers
from my_helpers.plot_classifier import PlotClassifier
from my_helpers.test_diff_fr import TestDiffFr
import sys
import argparse
import os
import time

from napolitano.plot_classifiers import PlotClassifiers
from classifiers_test.plot_classifiers import PlotClassifiers as PlotClassifiers2

if __name__ == '__main__':
    logger.info("ECG Research")
    # time.sleep(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('test-classifier-src', 'plot-classifier2', 'test-classifier2', 'plot-classifier', 'show-intervals', 'test-classifier', 'diff-test', 'plot-ekg', 'get-intervals', 'gen-intervals', 'gen-fr', 'plot-fr', 'main', 'fourier-series-demonstration', 'fourier-series-test-1', 'train_classifier', 'plot_classifier', 'plot-statistics', 'plot-fourier-statistics', 'plot-autocorrelation', 'plot-autocovariation'))
    parser.add_argument('-c', type=str, required=True)
    parser.add_argument('-d', type=str)
    parser.add_argument('-s', type=str)
    a = parser.parse_args()
    config_block = a.c
    logger.debug("Read config file")
    ecg_config = ECGConfig(config_block)
    logger.debug(ecg_config)
    if a.action == "test-classifier-src":
        Test_src(ecg_config).TestRed()
    if a.action == "plot-classifier2":
        PlotClassifiers2(ecg_config)
    if a.action == "test-classifier2":
        # for i in range(1, 31):
        #     Test(ecg_config).TestRed(i)
        Test(ecg_config).TestRed(1)
    if a.action == "plot-classifier":
        PlotClassifiers(ecg_config)
    if a.action == "diff-test":
        TestDiffFr(ecg_config).getTestData()
    if a.action == "show-intervals":
        GenerateRhythmFunction(ecg_config).showIntervals()
    if a.action == "gen-intervals":
        GenerateRhythmFunction(ecg_config).genIntervals()
    if a.action == "get-intervals":
        GenerateRhythmFunction(ecg_config).getIntervals()
    if a.action == "gen-fr":
        GenerateRhythmFunction(ecg_config).genFr()
    if a.action == "plot-fr":
        GenerateRhythmFunction(ecg_config).plotFr()
    if a.action == "plot-ekg":
        GenerateRhythmFunction(ecg_config).plotECG()
    if a.action == "fourier-series-demonstration":
        data = DataPreparation(ecg_config)
        FourierSeries(ecg_config, data).getFourierSeriesDemonstration(int(a.s or 0))
    if a.action == "fourier-series-test-1":
        data = DataPreparation(ecg_config)
        data.plotAllCycles()
        # FourierSeries(ecg_config, data).getFourierSpectrum([1, 3, 10])
    if a.action == "plot-statistics":
        data = DataPreparation(ecg_config)
        statistics = MathematicalStatistics(data.getPreparedData())

        # path = f'{ecg_config.getImgPath()}/All Mean/CSV/Arrhythmia Mathematical Expectation.csv'
        path = f'{ecg_config.getImgPath()}/All Mean/CSV/Healthy Mathematical Expectation.csv'
        df = pd.read_csv(path)
        no_mean = df["Data"]
        statistics.setNoVariance(no_mean)

        PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAllStatistics()

    if a.action == "plot-fourier-statistics":
        data = DataPreparation(ecg_config)
        statistics = MathematicalStatistics(data.getPreparedData())
        PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAllFourierStatistics()

    if a.action == "plot-autocorrelation":
        data = DataPreparation(ecg_config)
        statistics = MathematicalStatistics(data.getPreparedData())
        PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAutocorrelation()

    if a.action == "plot-autocovariation":
        data = DataPreparation(ecg_config)
        statistics = MathematicalStatistics(data.getPreparedData())
        PlotStatistics(statistics, data.getModSamplingRate(), ecg_config, data.getPreparedData()).plotAutocovariation()

    if a.action == "plot_classifier":
        data = DataPreparation(ecg_config)
        PlotClassifier(ecg_config, data)

    if a.action == "test-classifier":
        TestClassifiers(ecg_config)

    if a.action == "train_classifier":
        config_block_2 = a.d or config_block
        logger.debug("Read config file 2")
        ecg_config_2 = ECGConfig(config_block_2)
        logger.debug(ecg_config_2)
        data = DataPreparation(ecg_config)
        data_2 = DataPreparation(ecg_config_2)
        Classifiers(ecg_config, data, ecg_config_2, data_2)

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
        



        
