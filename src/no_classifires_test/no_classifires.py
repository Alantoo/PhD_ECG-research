from loguru import logger
import no_classifires_test.a_data as a_data
import no_classifires_test.h_data as h_data
from get_config.ecg_config import ECGConfig
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt


class NoClassidire():

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        self.sampling_rate = 360
        logger.debug("No Classifires")

    def NoTest(self):
        logger.debug("No Test Sigma")
        path = f'{self.ecg_config.getImgPath()}/All Mean'
        healthy_sigma = pd.read_csv(f'{path}/CSV/Healthy Sigma.csv')["Data"]
        arrhythmia_sigma = pd.read_csv(f'{path}/CSV/Arrhythmia Sigma.csv')["Data"]
        healthy_mean = pd.read_csv(f'{path}/CSV/Healthy Mathematical Expectation.csv')["Data"]
        arrhythmia_mean = pd.read_csv(f'{path}/CSV/Arrhythmia Mathematical Expectation.csv')["Data"]
        mean = pd.read_csv(f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Mathematical Statistics/CSV/1 Mathematical Expectation.csv')["Data"]

        n_sigma = 2

        h_upper_bound = healthy_mean + (n_sigma * healthy_sigma)
        h_lower_bound = healthy_mean - (n_sigma * healthy_sigma)

        a_upper_bound = arrhythmia_mean + (n_sigma * arrhythmia_sigma)
        a_lower_bound = arrhythmia_mean - (n_sigma * arrhythmia_sigma)

        h_within_bounds = (mean >= h_lower_bound) & (mean <= h_upper_bound)
        a_within_bounds = (mean >= a_lower_bound) & (mean <= a_upper_bound)

        percent_h_within_bounds = np.mean(h_within_bounds) * 100
        percent_a_within_bounds = np.mean(a_within_bounds) * 100

        with open(f'{path}/test_res.txt', 'a', encoding='utf-8') as file:

            print(f'Healthy: {percent_h_within_bounds:.2f}%')
            print(f'Arrhythmia: {percent_a_within_bounds:.2f}%')

            file.write(f'Healthy: {percent_h_within_bounds:.2f}%\n')
            file.write(f'Arrhythmia: {percent_a_within_bounds:.2f}%\n')

            if percent_h_within_bounds > percent_a_within_bounds :
                print(f'Patient {self.ecg_config.getConfigBlock()} -> Healthy')
                file.write(f'Patient {self.ecg_config.getConfigBlock()} -> Healthy\n')
            else:
                print(f'Patient {self.ecg_config.getConfigBlock()} -> Arrhythmia')
                file.write(f'Patient {self.ecg_config.getConfigBlock()} -> Arrhythmia\n')
            file.write('\n')


        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Mathematical Statistics'


        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(12, 6)
        axis.grid(True)
        time = np.arange(0, len(mean), 1) / self.sampling_rate
        axis.plot(time, mean, linewidth=3)
        # axis.plot(time, healthy_mean, linewidth=3)
        axis.plot(time, h_upper_bound, linewidth=3)
        axis.plot(time, h_lower_bound, linewidth=3)
        plt.savefig(f'{path}/Healthy test.png', dpi=300)

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(12, 6)
        axis.grid(True)
        time = np.arange(0, len(mean), 1) / self.sampling_rate
        axis.plot(time, mean, linewidth=3)
        # axis.plot(time, arrhythmia_mean, linewidth=3)
        axis.plot(time, a_upper_bound, linewidth=3)
        axis.plot(time, a_lower_bound, linewidth=3)
        plt.savefig(f'{path}/Arrhythmia test.png', dpi=300)



    def NoAllSigma(self):
        logger.debug("No All Sigma")
        
        mean_a_data = []
        mean_h_data = []

        for conf in a_data.read_files:
            ecg_conf = ECGConfig(conf)
            path = f'{ecg_conf.getImgPath()}/{ecg_conf.getConfigBlock()}/Mathematical Statistics/CSV/No Sigma.csv'
            df = pd.read_csv(path)
            mean_a_data.append(df["Data"])

        for conf in h_data.read_files:
            ecg_conf = ECGConfig(conf)
            path = f'{ecg_conf.getImgPath()}/{ecg_conf.getConfigBlock()}/Mathematical Statistics/CSV/No Sigma.csv'
            df = pd.read_csv(path)
            mean_h_data.append(df["Data"])

        mean_a_data = np.transpose(mean_a_data)
        all_mean_a_data = [np.mean(i) for i in mean_a_data]
        mean_h_data = np.transpose(mean_h_data)
        all_mean_h_data = [np.mean(i) for i in mean_h_data]

        xtext = "$t, s$"
        path = f'{self.ecg_config.getImgPath()}/All Mean'
        self.plot_to_png(path, all_mean_a_data, "Arrhythmia Sigma", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")
        self.plot_to_png(path, all_mean_h_data, "Healthy Sigma", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")
        self.plot_to_csv(all_mean_a_data, "Arrhythmia Sigma")
        self.plot_to_csv(all_mean_h_data, "Healthy Sigma")

    def NoAllMean(self):
        logger.debug("No All Mean")
        
        mean_a_data = []
        mean_h_data = []

        for conf in a_data.read_files:
            ecg_conf = ECGConfig(conf)
            path = f'{ecg_conf.getImgPath()}/{ecg_conf.getConfigBlock()}/Mathematical Statistics/CSV/1 Mathematical Expectation.csv'
            df = pd.read_csv(path)
            mean_a_data.append(df["Data"])

        for conf in h_data.read_files:
            ecg_conf = ECGConfig(conf)
            path = f'{ecg_conf.getImgPath()}/{ecg_conf.getConfigBlock()}/Mathematical Statistics/CSV/1 Mathematical Expectation.csv'
            df = pd.read_csv(path)
            mean_h_data.append(df["Data"])

        mean_a_data = np.transpose(mean_a_data)
        all_mean_a_data = [np.mean(i) for i in mean_a_data]
        mean_h_data = np.transpose(mean_h_data)
        all_mean_h_data = [np.mean(i) for i in mean_h_data]

        xtext = "$t, s$"
        path = f'{self.ecg_config.getImgPath()}/All Mean'
        self.plot_to_png(path, all_mean_a_data, "Arrhythmia Mathematical Expectation", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")
        self.plot_to_png(path, all_mean_h_data, "Healthy Mathematical Expectation", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")
        self.plot_to_csv(all_mean_a_data, "Arrhythmia Mathematical Expectation")
        self.plot_to_csv(all_mean_h_data, "Healthy Mathematical Expectation")

    def plot_to_png(self, path, plot, name, xlim = None, ylim = None, ytext="", xtext="", size=(12, 6)):
        logger.info(f"Plot {name}.png")
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        axis.plot(time, plot, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.set_title(ytext, loc = 'left', fontsize=14, position=(-0.06, 0))
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)

    def plot_to_csv(self, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{self.ecg_config.getImgPath()}/All Mean/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        data = pd.DataFrame({"Time" : time, "Data" : plot})
        nk.write_csv(data, f'{path}/{name}.csv')