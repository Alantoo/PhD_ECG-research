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
        self.ecg_config = ecg_config

        # ffttest = self.tfft(self.all_matrix[0])[:10]

        plot_path = f'{self.ecg_config.getImgPath()}/{data.getSigNameDir()}'

        x=np.linspace(0,1,1000,endpoint=False)

        Path(plot_path).mkdir(parents=True, exist_ok=True)

        L = 1 # Periodicity of the periodic function f(x)
        samples = 1000
        terms=40
        # Generation of square wave
        x = np.linspace(0,L,samples,endpoint=False)
        y = self.all_matrix[0]
        # Calculation of Fourier coefficients
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)


        # sum of the series
        s=a0/2.+sum([an(k)*np.cos(2.*np.pi*k*x/L)+bn(k)*np.sin(2.*np.pi*k*x/L) for k in range(1,terms+1)])
        # Plotting
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.plot(x, self.all_matrix[0], linewidth=2)
        axis.plot(x,s,label="Fourier series")
        axis.plot(x,y,label="Original square wave")
        axis.legend(loc='best',prop={'size':10})
        plt.savefig(f'{plot_path}/test_c_{terms}.png', dpi=300)

