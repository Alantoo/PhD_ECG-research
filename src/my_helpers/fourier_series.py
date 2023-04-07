from loguru import logger
import numpy as np
import scipy.interpolate as interp
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.signal import square

class FourierSeries():

    def __init__(self, ecg_config, data):
        print("Test")
        self.all_matrix = data.getPreparedData()
        self.sampling_rate = data.getModSamplingRate()
        self.ecg_config = ecg_config
        self.plot_path = f'{self.ecg_config.getImgPath()}/{data.getSigNameDir()}'
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)


    def getFourierSpectrum(self, signals, terms = 50):
        for i in signals:
            self.plotSpectrum(i, terms)


    def plotSpectrum(self, signal, terms):
        a0, an, bn, _, _ = self.getFourierSeries(self.all_matrix[signal])
        list_a = [a0, *[an(k) for k in range(1, terms + 1)]]
        list_b = [0, *[bn(k) for k in range(1, terms + 1)]]

        abs_list_a = np.abs(list_a)
        abs_list_b = np.abs(list_b)

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.stem(abs_list_a, markerfmt=" ")
        plt.savefig(f'{self.plot_path}/{signal}_Example_a_n.png', dpi=300)

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.stem(abs_list_b, markerfmt=" ")
        plt.savefig(f'{self.plot_path}/{signal}_Example_b_n.png', dpi=300)


    def getFourierSeriesDemonstration(self, signal):
        fourier_series = self.getFourierSeries(self.all_matrix[signal])
        for terms in [2, 3, 5, 10, 20, 30, 40, 50]:
            self.plotFourierSeriesDemonstration(*fourier_series, terms, signal)

    def plotFourierSeriesDemonstration(self, a0, an, bn, x, y, terms, i, L = 1):
        # sum of the series
        s=a0/2.+sum([an(k)*np.cos(2.*np.pi*k*x/L)+bn(k)*np.sin(2.*np.pi*k*x/L) for k in range(1,terms+1)])

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.plot(x,s,label="Fourier series", linewidth=2)
        axis.plot(x,y,label="Original square wave", linewidth=2)
        axis.legend(loc='best',prop={'size':10})
        Path(f'{self.plot_path}/Fourier Series Demonstration').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{self.plot_path}/Fourier Series Demonstration/{i}_Example_c_{terms}.png', dpi=300)


    def getFourierSeries(self, y, L = 1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
        return a0, an, bn, x, y