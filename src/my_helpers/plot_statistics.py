from my_helpers.mathematical_statistics_data import MathematicalStatisticsData
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk
from matplotlib import cm
from pathlib import Path
from loguru import logger


class PlotStatistics():
    def __init__(self, statistics, sampling_rate, ecg_config, matrix_all):
        self.sampling_rate = sampling_rate
        self.ecg_config = ecg_config
        self.statistics = statistics
        self.matrix_all = matrix_all
        self.plot_path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'

    def plotAutocorrelation(self):
        logger.info("Plot Autocorrelation")
        # c1 = self.getCorrelation(correlation = False, deep = 1, multiply = False)
        # self._3d_plot_to_file(c1, "Autocorrelation function d-1 m-f", "Autocorrelation", size=(10, 10, 10), ztext=r'$\hat{C}_{2_{\xi}} (t_1, t_2), mV^2$')
        # c2 = self.getCorrelation(correlation = False, deep = 3, multiply = False)
        # self._3d_plot_to_file(c2, "Autocorrelation function d-3 m-f", "Autocorrelation", size=(10, 10, 10), ztext=r'$\hat{C}_{2_{\xi}} (t_1, t_2), mV^2$')
        c3 = self.getCorrelation(correlation=False, deep=3, multiply=True)
        self._3d_plot_to_file(c3, "Autocorrelation function d-3 m-t", "Autocorrelation", size=(10, 10, 10),
                              ztext=r'$\hat{C}_{2_{\xi}} (t_1, t_2), NU^2$')
        # self._3d_plot_to_file(c3, "L-Autocorrelation function d-3 m-t", "Autocorrelation", size=(10, 10, 10), ztext=r'${\widehat{C}}_{\xi_{\hat{T}_{av}}}^2 (t_1, t_2), NU^2$')

    def plotAutocovariation(self):
        logger.info("Plot Autocovariation")
        # c1 = self.getCorrelation(correlation = True, deep = 1, multiply = False)
        # self._3d_plot_to_file(c1, "Autocovariation function d-1 m-f", "Autocovariation", size=(10, 10, 10), ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), mV^2$')
        # c2 = self.getCorrelation(correlation = True, deep = 3, multiply = False)
        # self._3d_plot_to_file(c2, "Autocovariation function d-3 m-f", "Autocovariation", size=(10, 10, 10), ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), mV^2$')
        c3 = self.getCorrelation(correlation=True, deep=3, multiply=True)
        self._3d_plot_to_file(c3, "Autocovariation function d-3 m-t", "Autocovariation", size=(10, 10, 10),
                              ztext=r'$\hat{R}_{2_{\xi}} (t_1, t_2), NU^2$')
        # self._3d_plot_to_file(c3, "L-Autocovariation function d-3 m-t", "Autocovariation", size=(10, 10, 10), ztext=r'${\widehat{R}}_{\xi_{\hat{T}_{av}}}^2 (t_1, t_2), NU^2$')

    def get_math_stats_points(self):
        mathematical_statistics = self.statistics.getMathematicalStatistics()
        return {
            "mathematical_expectation": mathematical_statistics.getMathematicalExpectation(),
            "initial_moments_second_order": mathematical_statistics.getInitialMomentsSecondOrder(),
            "initial_moments_third_order": mathematical_statistics.getInitialMomentsThirdOrder(),
            "initial_moments_fourth_order": mathematical_statistics.getInitialMomentsFourthOrder(),
            "central_moment_functions_fourth_order": mathematical_statistics.getCentralMomentFunctionsFourthOrder(),
            "variance": mathematical_statistics.getVariance(),
        }

    def plotAllStatistics(self):
        logger.info("Plot Mathematical Statistics")
        mathematical_statistics = self.statistics.getMathematicalStatistics()
        xtext = "$t, s$"
        self.plot_to_png([*mathematical_statistics.getMathematicalExpectation()] * 6, "1 Mathematical Expectation",
                         xtext=xtext, ytext=r"$\widehat{m}_{\xi_{\omega}} (t), NU$")
        self.plot_to_png([*mathematical_statistics.getInitialMomentsSecondOrder()] * 6,
                         "2 Initial Moments Second Order", xtext=xtext,
                         ytext=r"${\widehat{m}}_{\xi_{\omega}}^2 (t), NU^2$")
        self.plot_to_png([*mathematical_statistics.getInitialMomentsThirdOrder()] * 6, "3 Initial Moments Third Order",
                         xtext=xtext, ytext=r"${\widehat{m}}_{\xi_{\omega}}^3 (t), NU^3$")
        self.plot_to_png([*mathematical_statistics.getVariance()] * 6, "5 Variance", xtext=xtext,
                         ytext=r"${\widehat{d}}_{\xi_{\omega}}^2 (t), NU^2$")

        # self.plot_to_png([*mathematical_statistics.getMathematicalExpectation()] * 6, "1 Mathematical Expectation", xtext=xtext, ytext=r"$\widehat{m}_{\xi_{\hat{T}_{av}}} (t), NU$")
        # self.plot_to_png([*mathematical_statistics.getInitialMomentsSecondOrder()] * 6, "2 Initial Moments Second Order", xtext=xtext, ytext=r"$\widehat{m}_{\xi_{\hat{T}_{av}}}^2 (t), NU^2$")
        # self.plot_to_png([*mathematical_statistics.getInitialMomentsThirdOrder()] * 6, "3 Initial Moments Third Order", xtext=xtext, ytext=r"$\widehat{m}_{\xi_{\hat{T}_{av}}}^3 (t), NU^3$")
        # self.plot_to_png([*mathematical_statistics.getVariance()] * 6, "5 Variance", xtext=xtext, ytext=r"$\widehat{d}_{\xi_{\hat{T}_{av}}}^2 (t), NU^2$")

        # self.plot_to_png([*mathematical_statistics.getInitialMomentsFourthOrder()] * 6, "4 Initial Moments Fourth Order", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$")

        # self.plot_to_png(mathematical_statistics.getVariance(), "No Variance", xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^2$")
        # self.plot_to_png(np.sqrt(mathematical_statistics.getVariance()), "No Sigma", xtext=xtext, ytext=r"$\hat{\sigma}_{{\xi}} (N), mV$")
        # self.plot_to_png(mathematical_statistics.getCentralMomentFunctionsFourthOrder(), "6 Central Moment Functions Fourth Order",  xtext=xtext, ytext=r"$d_{{\xi}} (t), mV^4$")є
        # self.plot_to_png(np.sqrt(mathematical_statistics.getVariance()), "No Sigma", xtext=xtext, ytext=r"$\hat{\sigma}_{{\xi}} (N), mV$")

        self.plot_to_csv(mathematical_statistics.getMathematicalExpectation(), "1 Mathematical Expectation")
        self.plot_to_csv(mathematical_statistics.getInitialMomentsSecondOrder(), "2 Initial Moments Second Order")
        self.plot_to_csv(mathematical_statistics.getInitialMomentsThirdOrder(), "3 Initial Moments Third Order")
        self.plot_to_csv(mathematical_statistics.getInitialMomentsFourthOrder(), "4 Initial Moments Fourth Order")
        self.plot_to_csv(mathematical_statistics.getVariance(), "5 Variance")
        # self.plot_to_csv(mathematical_statistics.getVariance(), "No Variance")
        # self.plot_to_csv(np.sqrt(mathematical_statistics.getVariance()), "No Sigma")
        self.plot_to_csv(mathematical_statistics.getCentralMomentFunctionsFourthOrder(),
                         "6 Central Moment Functions Fourth Order")

    def plotAllFourierStatistics(self):
        logger.info("Plot Mathematical Statistics Fourier")
        self.statistics.setSamplingRate(self.sampling_rate)
        mathematical_statistics = self.statistics.getMathematicalStatisticsFourierSeries()
        xtext = "$n$"
        self.fs_plot_to_png(mathematical_statistics.getMathematicalExpectation(), "1 Mathematical Expectation",
                            xtext=xtext, ytext=(r"$a_n, mV$", r"$b_n, mV$"))
        self.fs_plot_to_png(mathematical_statistics.getInitialMomentsSecondOrder(), "2 Initial Moments Second Order",
                            xtext=xtext, ytext=(r"$a_n, mV^2$", r"$b_n, mV^2$"))
        self.fs_plot_to_png(mathematical_statistics.getInitialMomentsThirdOrder(), "3 Initial Moments Third Order",
                            xtext=xtext, ytext=(r"$a_n, mV^3$", r"$b_n, mV^3$"))
        self.fs_plot_to_png(mathematical_statistics.getInitialMomentsFourthOrder(), "4 Initial Moments Fourth Order",
                            xtext=xtext, ytext=(r"$a_n, mV^4$", r"$b_n, mV^4$"))
        self.fs_plot_to_png(mathematical_statistics.getVariance(), "5 Variance", xtext=xtext,
                            ytext=(r"$a_n, mV^2$", r"$b_n, mV^2$"))
        self.fs_plot_to_png(mathematical_statistics.getCentralMomentFunctionsFourthOrder(),
                            "6 Central Moment Functions Fourth Order", xtext=xtext,
                            ytext=(r"$a_n, mV^4$", r"$b_n, mV^4$"))

        self.fs_plot_to_csv(mathematical_statistics.getMathematicalExpectation(), "1 Mathematical Expectation")
        self.fs_plot_to_csv(mathematical_statistics.getInitialMomentsSecondOrder(), "2 Initial Moments Second Order")
        self.fs_plot_to_csv(mathematical_statistics.getInitialMomentsThirdOrder(), "3 Initial Moments Third Order")
        self.fs_plot_to_csv(mathematical_statistics.getInitialMomentsFourthOrder(), "4 Initial Moments Fourth Order")
        self.fs_plot_to_csv(mathematical_statistics.getVariance(), "5 Variance")
        self.fs_plot_to_csv(mathematical_statistics.getCentralMomentFunctionsFourthOrder(),
                            "6 Central Moment Functions Fourth Order")

    def fs_plot_to_png(self, plot2, name, xlim=None, ylim=None, ytext=(r"$a_n, mV$", r"$b_n, mV$"), xtext="",
                       size=(19, 6)):
        path = f'{self.plot_path}/Mathematical Statistics Fourier'
        Path(path).mkdir(parents=True, exist_ok=True)
        an, bn = plot2
        logger.info(f"Plot {name} an.png")
        plt.clf()
        plt.rcParams.update({'font.size': 16})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size)
        axis.set_xlabel(xtext, loc='right')
        axis.set_title(ytext[0], loc='left', fontsize=15, position=(-0.05, 0))
        axis.grid(True)
        _, stemlines, _ = axis.stem([0, *an[1:]])
        plt.setp(stemlines, 'linewidth', 3)
        plt.savefig(f'{path}/{name} an.png', dpi=300)

        logger.info(f"Plot {name} bn.png")
        plt.clf()
        plt.rcParams.update({'font.size': 16})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size)
        axis.set_xlabel(xtext, loc='right')
        axis.set_title(ytext[1], loc='left', fontsize=15, position=(-0.05, 0))
        axis.grid(True)
        _, stemlines, _ = axis.stem([0, *bn])
        plt.setp(stemlines, 'linewidth', 3)
        plt.savefig(f'{path}/{name} bn.png', dpi=300)

    def plot_to_png(self, plot, name, xlim=(0, 4.5), ylim=None, ytext="", xtext="", size=(19, 6)):
        logger.info(f"Plot {name}.png")
        path = f'{self.plot_path}/Mathematical Statistics'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 16})
        f, axis = plt.subplots(1)
        f.set_size_inches(size[0], size[1])
        # f.tight_layout()
        axis.grid(True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        axis.plot(time, plot, linewidth=3)
        axis.set_xlabel(xtext, loc='right')
        axis.set_title(ytext, loc='left', fontsize=18, position=(-0.04, 0))
        if xlim is not None:
            axis.axis(xmin=xlim[0], xmax=xlim[1])
        if ylim is not None:
            axis.axis(ymin=ylim[0], ymax=ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)

    def plot_to_csv(self, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{self.plot_path}/Mathematical Statistics/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        data = pd.DataFrame({"Time": time, "Data": plot})
        nk.write_csv(data, f'{path}/{name}.csv')

    def fs_plot_to_csv(self, plot2, name):
        an, bn = plot2
        logger.info(f"Save {name}.csv")
        path = f'{self.plot_path}/Mathematical Statistics Fourier/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(an), 1)
        data = pd.DataFrame({"n": time, "an": an, "bn": [0, *bn]})
        nk.write_csv(data, f'{path}/{name}.csv')

    def getCorrelation(self, correlation=False, deep=1, multiply=False):
        mathematical_statistics = self.statistics.getMathematicalStatistics()
        tmp = None

        matrix_all_len = len(self.matrix_all)

        for i in range(matrix_all_len - deep + 1):
            concated = np.ravel(self.matrix_all[i: i + deep])
            one = self.matrix_all[i]
            if correlation:
                one = one - mathematical_statistics.getMathematicalExpectation()
                concated = concated - np.tile(mathematical_statistics.getMathematicalExpectation(), deep)

            r1, r2 = np.meshgrid(one, concated)
            r = r1 * r2
            if tmp is None:
                tmp = r
            else:
                tmp = tmp + r

        res2 = tmp / (matrix_all_len - deep + 1)

        if (deep == 3) and multiply:
            res_len = len(res2[0])
            r1 = res2[0:res_len]
            r2 = res2[res_len:(2 * res_len)]
            r3 = res2[(2 * res_len):(3 * res_len)]
            w_1_1 = r1
            w_1_2 = r2
            w_1_3 = r3
            w_2_1 = np.rot90(np.fliplr(r2))
            w_2_2 = r1
            w_2_3 = r2
            w_3_1 = np.rot90(np.fliplr(r3))
            w_3_2 = np.rot90(np.fliplr(r2))
            w_3_3 = r1

            w1 = np.concatenate((w_1_1, w_1_2, w_1_3))
            w2 = np.concatenate((w_2_1, w_2_2, w_2_3))
            w3 = np.concatenate((w_3_1, w_3_2, w_3_3))

            res2 = np.concatenate((w1, w2, w3), axis=1)

        return res2

    def _3d_plot_to_file(self, plot1, name, path, size=(10, 10, 10), ztext="", v=None):
        logger.info(f"Plot {name}.png")
        path = f'{self.plot_path}/{path}'
        Path(path).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        fig.set_size_inches(size[0], size[1], size[2])

        t1 = np.arange(0, len(plot1[0]), 1) / self.sampling_rate
        t2 = np.arange(0, len(plot1), 1) / self.sampling_rate
        t1, t2 = np.meshgrid(t1, t2)
        ax.set_xlabel('$t_1, s$', fontsize=13)
        ax.set_ylabel('$t_2, s$', fontsize=13)
        ax.set_zlabel(ztext, fontsize=12)
        # Plot the surface.
        if v is None:
            surf = ax.plot_surface(t1, t2, plot1, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0)
        else:
            surf = ax.plot_surface(t1, t2, plot1, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0, vmin=v[0],
                                   vmax=v[1])
        # ax.zaxis.set_major_formatter('{x:.02f}')

        plt.gca().invert_xaxis()
        plt.savefig("{}/{}.png".format(path, name), dpi=300)
