from get_config.ecg_config import ECGConfig
from loguru import logger
from my_helpers.read_data.read_data_file import ReadDataFile
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from napolitano.confusion_matrix import ConfusionMatrix
import napolitano.napolitano_module as napolitano_module

class Napolitano():

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        logger.debug("Napolitano Init")

    def TestRed(self, i):
        logger.debug("Napolitano Test")
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Naive Bayes"
        ]

        confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "False Negative Rate",
            "False Positive Rate", "Accuracy", "Balanced Accuracy",
            "F1 score", "Learning_time", "Testing_time"
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
        ]
        test_data = []
        target_data = []
        for conf in napolitano_module.read_files:
            ecg_conf = ECGConfig(conf)
            data = ReadDataFile(ecg_conf)
            shift_real_parts = [x.real for x in data.RD_FourierCoefficients]
            shift_imag_parts = [x.imag for x in data.RD_FourierCoefficients]

            # shift_real_parts = [x.real for x in data.RD_Ryxa]
            # shift_imag_parts = [x.imag for x in data.RD_Ryxa]

            # shift_real_parts = [yi for xi, yi in zip(data.RD_Harmonics, real_parts) if xi >= 0]
            # shift_imag_parts = [yi for xi, yi in zip(data.RD_Harmonics, imag_parts) if xi >= 0]

            # shift_real_parts = [x.real for x in data.RD_Pyxa]
            # shift_imag_parts = [x.imag for x in data.RD_Pyxa]

            # shift_real_parts = [yi for xi, yi in zip(data.RD_F1, real_parts) if xi >= 0]
            # shift_imag_parts = [yi for xi, yi in zip(data.RD_F1, imag_parts) if xi >= 0]
            
            # shift_imag_parts = np.fft.fftshift(imag_parts).tolist()
            # shift_imag_parts_h = shift_imag_parts[(len(shift_imag_parts) // 2):]

            # shift_real_parts = np.fft.fftshift(real_parts).tolist()
            # shift_real_parts_h = shift_real_parts[(len(shift_real_parts) // 2):]

            

            # real_imag_pairs = shift_imag_parts[:i]

            # real_imag_pairs = []
            # for x in data.RD_FourierCoefficients:
            #     real_imag_pairs.append(x.real)
            #     real_imag_pairs.append(x.imag)
            test_data.append([*shift_real_parts, *shift_imag_parts])
            # test_data.append(real_imag_pairs)
            target_data.append(ecg_conf.getPathology())

        max_size = max(len(inner_list) for inner_list in test_data)
        for inner_list in test_data:
            while len(inner_list) < max_size:
                inner_list.append(0)

        # print(test_data)

        data_train, data_test, target_values_train, target_values_test = train_test_split(test_data, target_data, test_size=0.3, random_state=42)

        _cm = []
        for name, clf in zip(names, classifiers):
            clf = make_pipeline(StandardScaler(), clf)
            lstart = time.time()
            clf.fit(data_train, target_values_train)
            lend = tstart = time.time()
            y_true = np.array(target_values_test)
            y_pred = clf.predict(data_test)
            tend = time.time()
            ltime = (lend-lstart)*10**3
            ttime = (tend-tstart)*10**3
            confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, ttime)
            res = confusion_matrix.getAllVariables()
            _cm.append(res)

        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier coefficients/data/Imag parts'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier coefficients/data/Real parts'
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier coefficients/data/Real and Imag parts'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Polyspectrums/data/Imag parts'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Polyspectrums/data/Real parts'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Polyspectrums/data/Real and Imag parts'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Cyclic Autocorrelation/data/Real and Imag parts'
        
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier coefficients/data/Real and Imag parts'
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(np.transpose(np.round(_cm, 4)), index=confusion_matrix_names, columns=names)
        df.to_csv(f'{path}/n-{i}.csv') 

    def PlotDataRed(self):
        logger.debug("Napolitano Plot")
        data = ReadDataFile(self.ecg_config)
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Red'
        Path(path).mkdir(parents=True, exist_ok=True)

        rd_plot_list = [
            {"name": "cyclic autocorrelation of x(t) (real part)", "value": [x.real for x in data.RD_Ryxa]},
            {"name": "cyclic autocorrelation of x(t) (imag part)", "value": [x.imag for x in data.RD_Ryxa]},
            # {"name": "tau axis for cyclic autocorrelation", "value": data.RD_Tau1},
            # {"name": "2nd-order cyclic polyspectrum (real part)", "value": [x.real for x in data.RD_Pyxa]},
            # {"name": "2nd-order cyclic polyspectrum (imag part)", "value": [x.imag for x in data.RD_Pyxa]},
            # {"name": "frequency axis for 2nd-order cyclic polyspectrum", "value": data.RD_F1},
            # {"name": "Fourier coefficients (real part)", "value": [x.real for x in data.RD_FourierCoefficients]},
            # {"name": "Fourier coefficients (imag part)", "value": [x.imag for x in data.RD_FourierCoefficients]},
            # {"name": "Harmonics", "value": data.RD_Harmonics}
        ]

        # print(data.RD_Ryxa)

        for plot_list in rd_plot_list:
            print(f'{path}/{plot_list["name"]}.png')
            plt.clf()
            plt.rcParams.update({'font.size': 14})
            f, axis = plt.subplots(1)
            f.tight_layout()
            f.set_size_inches(17, 6)
            axis.grid(True)
            # filtered_x = [xi for xi in data.RD_F1 if xi >= 0]
            # filtered_y = [yi for xi, yi in zip(data.RD_F1, plot_list["value"]) if xi >= 0]
            # filtered_x = [xi for xi in data.RD_Harmonics if xi >= 0]
            # filtered_y = [yi for xi, yi in zip(data.RD_Harmonics, plot_list["value"]) if xi >= 0]
            y = np.fft.fftshift(plot_list["value"])
            # hl = len(y) // 2
            # yh = y[hl:]
            # xh = data.RD_Harmonics[hl:]
            # xh = data.RD_F1[hl:]
            # axis.plot(filtered_x, filtered_y, linewidth=3)
            # axis.plot(xh[::-1], yh, linewidth=3)
            axis.plot(data.RD_Tau1, plot_list["value"], linewidth=4)
            # axis.plot(data.RD_F1, plot_list["value"], linewidth=4)
            # axis.plot(y, linewidth=4)
            # print(len(yh))
            # print(len(xh))
            # print(xh[::-1])
            # print(yh)
            # axis.set_xlabel("$t, s$", loc = 'right')
            # axis.legend(['$T(t, 1), s$'])
            # axis.axis(ymin = 0, ymax = 1.2)
            # axis.axis(xmin = -0.4, xmax = 0.4)
            plt.savefig(f'{path}/{plot_list["name"]}.png', dpi=300)

        # for plot_list in rd_plot_list:
        #     print(f'{path}/{plot_list["name"]}.png')
        #     plt.clf()
        #     plt.rcParams.update({'font.size': 14})
        #     f, axis = plt.subplots(1)
        #     f.tight_layout()
        #     f.set_size_inches(19, 6)
        #     axis.grid(True)
        #     axis.plot(data.RD_Harmonics, plot_list["value"], linewidth=3)
        #     # axis.set_xlabel("$t, s$", loc = 'right')
        #     # axis.legend(['$T(t, 1), s$'])
        #     # axis.axis(ymin = 0, ymax = 1.2)
        #     # axis.axis(xmin = 0, xmax = 5)
        #     plt.savefig(f'{path}/{plot_list["name"]}.png', dpi=300)


    def PlotDataCyc(self):
        logger.debug("Napolitano Plot")
        data = ReadDataFile(self.ecg_config)
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Cyc'
        Path(path).mkdir(parents=True, exist_ok=True)

        all_plot_list = [
            {"name": "Fourier coefficients", "value": data.FourierSeries_FourierCoefficients},
            {"name": "Harmonics", "value": data.FourierSeries_Harmonics},
            {"name": "cyclic autocorrelation of x(t)", "value": data.cycRSCP1_Ryxa},
            {"name": "cyclic autocovaraince of x(t)", "value": data.cycRSCP1_Cyxa},
            {"name": "tau axis for cyclic autocorrelation and autocovariance", "value": data.cycRSCP1_Tau1},
            {"name": "cyclic spectrum", "value": data.cycRSCP1_Syxa},
            {"name": "2nd-order cyclic polyspectrum", "value": data.cycRSCP1_Pyxa},
            {"name": "frequency axis for cyclic spectrum and 2nd-order cyclic polyspectrum", "value": data.cycRSCP1_F1},
            {"name": "autocorrelation of x(t)", "value": data.stationary_cycRSCP1_Ryxa},
            {"name": "autocovariance of x(t)", "value": data.stationary_cycRSCP1_Cyxa},
            {"name": "tau axis for autocorrelation and autocovariance", "value": data.stationary_cycRSCP1_Tau1},
            {"name": "power spectral density", "value": data.stationary_cycRSCP1_Syxa},
            {"name": "2nd-order polyspectrum", "value": data.stationary_cycRSCP1_Pyxa},
            {"name": "frequency axis for power spectral density and 2nd-order polyspectrum", "value": data.stationary_cycRSCP1_F1}
        ]

        for plot_list in all_plot_list:
            plt.clf()
            plt.rcParams.update({'font.size': 14})
            f, axis = plt.subplots(1)
            f.tight_layout()
            f.set_size_inches(19, 6)
            axis.grid(True)
            axis.plot(plot_list["value"], linewidth=3)
            # axis.set_xlabel("$t, s$", loc = 'right')
            # axis.legend(['$T(t, 1), s$'])
            # axis.axis(ymin = 0, ymax = 1.2)
            # axis.axis(xmin = 0, xmax = 5)
            plt.savefig(f'{path}/{plot_list["name"]}.png', dpi=300)
