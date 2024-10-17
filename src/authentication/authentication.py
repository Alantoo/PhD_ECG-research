import random
from authentication.confusion_matrix import ConfusionMatrix
from get_config.ecg_config import ECGConfig
from loguru import logger
from my_helpers.data_preparation import DataPreparation
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
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import neurokit2 as nk
import authentication.authentication_data as used_authentication
from scipy.stats import probplot

class Authentication():

    def __init__(self, ecg_config):
        logger.debug("Authentication Init")

        self.ecg_config = ecg_config
        self.ltime = 0
        self.a_path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'
        Path(self.a_path).mkdir(parents=True, exist_ok=True)

        self.names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Naive Bayes"
        ]

        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
        ]


        self.confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "False Negative Rate", "False Positive Rate", "Accuracy", "Balanced Accuracy",
            "F1 score", "Learning_time", "Testing_time"
        ]

    def Diff(self):
        logger.debug("Diff")
        # tmp_path = "Confusion matrix pleth_1"
        # tmp_path = "Confusion matrix"
        # tmp_path = "Confusion matrix line"
        # tmp_path = "Confusion matrix line SCG"
        tmp_path = "Confusion matrix SCG"
        for average_elements in np.arange(1, 20 + 1):
            all_data = [pd.read_csv(f'{self.ecg_config.getImgPath()}/{operator}/{tmp_path}/Authentication n-{average_elements}.csv').drop('Unnamed: 0', axis=1, errors='ignore') for operator in used_authentication.diff]
            # df = pd.DataFrame(np.round(np.mean(all_data, axis=0), 3), index=self.confusion_matrix_names, columns=[*self.names, "SIC"])
            std = pd.DataFrame(np.round(np.std(all_data, axis=0, ddof=1), 3), index=self.confusion_matrix_names, columns=[*self.names, "SIC"])
            Path(f'{self.a_path}/{tmp_path}').mkdir(parents=True, exist_ok=True)
            # df.to_csv(f'{self.a_path}/{tmp_path}/Authentication n-{average_elements}.csv')
            std.to_csv(f'{self.a_path}/{tmp_path}/STD-Authentication n-{average_elements}.csv')

        # all_data = [pd.read_csv(f'{self.ecg_config.getImgPath()}/{operator}/{tmp_path}/Sigma mean.csv').drop('Unnamed: 0', axis=1, errors='ignore') for operator in used_authentication.diff]  
        # df = pd.DataFrame(np.round(np.mean(all_data, axis=0), 4), columns=["N", "Data"])
        # df.to_csv(f'{self.a_path}/{tmp_path}/Sigma mean.csv')    

    def QQplot(self):
        logger.debug("QQplot")
        tmp_path = "Confusion matrix pleth_1"
        # tmp_path = "Confusion matrix"
        # tmp_path = "Confusion matrix SCG"
        confusion_matrix_names = [
            "Accuracy"
        ]
        # all_data = [pd.read_csv(f'{self.ecg_config.getImgPath()}/{operator}/{tmp_path}/Authentication n-1.csv') for operator in used_authentication.diff2]

        # for cm_name in confusion_matrix_names:
        #     read_data = []
        #     accuracy_row = all_data.loc[all_data['Unnamed: 0'] == cm_name]
        #     accuracy_array = accuracy_row.values.flatten()[1:]
        #     read_data.append(accuracy_array)

        # print(all_data[0])
        for cm_name in confusion_matrix_names:
            read_data = []
            for operator in used_authentication.diff2:
                df = pd.read_csv(f'{self.ecg_config.getImgPath()}/{operator}/{tmp_path}/Authentication n-1.csv')
                accuracy_row = df.loc[df['Unnamed: 0'] == cm_name]
                accuracy_array = accuracy_row.values.flatten()[1:]
                read_data.append(accuracy_array)

        flattened_list = [item for sublist in read_data for item in sublist]
        # print(flattened_list)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.set_size_inches(19, 6)
        f.tight_layout()
        axis.grid(True)
        probplot(flattened_list, dist="norm", plot=axis)
        axis.title.set_text('QQ-Plot for Accuracy')
        axis.set_xlabel('Theoretical Quantiles')
        axis.set_ylabel('Accuracy Quantiles')
        axis.axis(ymin = 0, ymax = 1.2)
        plt.savefig(f'{self.a_path}/{tmp_path}/QQ-Accuracy-Authentication n-1.png', dpi=300)

    def Classifiers(self):
        logger.debug("Classifiers")
        selected_array = used_authentication.arrays.get(self.ecg_config.getConfigBlock(), [])
        # other_array = used_authentication.arrays.get("OTHER1", [])
        other_array = self.form_res_array(used_authentication.arrays, self.ecg_config.getConfigBlock())

        all_data_in = []
        other_data_in = []

        # DataPreparation(ECGConfig(selected_array[0])).getPreparedData()

        all_data_in.extend([item for conf in selected_array for item in DataPreparation(ECGConfig(conf)).getPreparedData()])
        for conf in other_array:
               other_data_in.append(DataPreparation(ECGConfig(conf)).getPreparedData()[:int(len(all_data_in) / 10)])


        sigma_mean = []

        for average_elements in np.arange(1, 1 + 1):

            self.ltime = 0

            other_data = []

            all_data = self.average_elements_np(all_data_in, average_elements)

            for item in other_data_in:
                other_data.extend(self.average_elements_np(item, average_elements))

            target_data = [1] * len(all_data)
            other_target_data = [2] * len(other_data)

            all_data_ = [*all_data, *other_data]
            target_data_ = [*target_data, *other_target_data]

            data_train, data_test, target_values_train, target_values_test = train_test_split(all_data_, target_data_, test_size=0.3, random_state=42)

            y_true = np.array(target_values_test)

            # filtered_data = [data for data, target in zip(data_train, target_values_train) if target == 1]
            # filtered_other_data = [data for data, target in zip(data_train, target_values_train) if target == 2]
            self.NoAllSigma(self.a_path, all_data)

            a_sigmas = self.read_channel_data(self.a_path, "All Sigma")
            a_means = self.read_channel_data(self.a_path, "All Mathematical Expectation")

            sigma_mean.append(np.mean(a_sigmas))

            n_sigma = 2.6  #2.6

            lstart = time.time()
            a_upper_bounds = np.power(a_means + (n_sigma * a_sigmas), 1)
            a_lower_bounds = np.power(a_means - (n_sigma * a_sigmas), 1)


            mean = all_data[10]
            # mean = other_data[10]

            plt.clf()
            plt.rcParams.update({'font.size': 15})
            f, axis = plt.subplots(1)
            f.set_size_inches(12, 6)
            f.tight_layout()
            axis.grid(True)
            m_time = np.arange(0, len(mean), 1) / 360
            axis.plot(m_time, mean, linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
            axis.plot(m_time, a_upper_bounds, linewidth=3, label=r"$Upper_{{\xi}} (t), mV$")
            axis.plot(m_time, a_lower_bounds, linewidth=3, label=r"$Lower_{{\xi}} (t), mV$")
            axis.set_xlabel("$t, s$", loc = 'right')
            axis.axis(xmin = 0, xmax = 1)
            axis.legend(loc='upper right')
            plt.savefig(f'{self.a_path}/Other-Authentication.png', dpi=300)
            # plt.savefig(f'{self.a_path}/Authentication.png', dpi=300)


            lend = time.time()
            ltime = self.ltime + (lend-lstart)*10**3

            y_pred = []

            tstart = time.time()
            y_pred = [1 if (np.mean((mean >= a_lower_bounds) & (mean <= a_upper_bounds)) * 100) >= 93.0 else 2 for mean in data_test] #82---86
            tend = time.time()
            ttime = (tend-tstart)*10**3

            confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, ttime)
            sic_res = confusion_matrix.getAllVariables()
            print(("%s: %.2f" % ("SIC", confusion_matrix.getACC() * 100)))

            _cm = []
            for name, clf in zip(self.names, self.classifiers):
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
                print(("%s: %.2f" % (name, confusion_matrix.getACC() * 100)))
                _cm.append(res)

            _cm.append(sic_res)

            path = f'{self.a_path}/Confusion matrix pleth_1 line'
            # path = f'{self.a_path}/Confusion matrix line 0'
            # path = f'{self.a_path}/Confusion matrix line SCG'
            # path = f'{self.a_path}/Confusion matrix SCG'
            Path(path).mkdir(parents=True, exist_ok=True)

            df = pd.DataFrame(np.transpose(np.round(_cm, 4)), index=self.confusion_matrix_names, columns=[*self.names, "SIC"])
            df.to_csv(f'{path}/Authentication n-{average_elements}.csv')

        data = pd.DataFrame({"N" : np.arange(1, average_elements + 1), "Data" : sigma_mean})
        nk.write_csv(data, f'{path}/Sigma mean.csv')

    def Plot_n(self):
        logger.debug("Plot_n")
        confusion_matrix_names = [
            # "Accuracy",
            "Accuracy", "Balanced Accuracy",
            "F1 score", "Learning_time", "Testing_time"
        ]
        names = [*self.names, "SIC"]
        # path = f'{self.a_path}/Confusion matrix pleth_1 line'
        # path = f'{self.a_path}/Confusion matrix line'
        # path = f'{self.a_path}/Confusion matrix'
        # path = f'{self.a_path}/Confusion matrix line SCG'
        path = f'{self.a_path}/Confusion matrix SCG'
        # arr = np.arange(1, 20 + 1)
        # for cm_name in confusion_matrix_names:
        #     read_data = []
        #     for i in arr:
        #         df = pd.read_csv(f'{path}/Authentication n-{i}.csv')
        #         accuracy_row = df.loc[df['Unnamed: 0'] == cm_name]
        #         accuracy_array = accuracy_row.values.flatten()[1:]
        #         read_data.append(accuracy_array)
        #     t2 = np.transpose(np.array(read_data))
        #     plt.clf()
        #     plt.rcParams.update({'font.size': 14})
        #     f, axis = plt.subplots(1)
        #     f.set_size_inches(19, 6)
        #     f.tight_layout()
        #     axis.grid(True)
        #     axis.set_xlabel("N", loc = 'right')
        #     if "time" in cm_name:
        #         t2 = t2 / 1000.0
        #         axis.set_title("t, s", loc = 'left', fontsize=14, position=(-0.02, 0))
        #     for i, name in zip(t2[::-1], names[::-1]):
        #         axis.plot(arr, i, linewidth=2, label=name, marker='o')
        #     axis.legend(loc='best',prop={'size':10})

        #     max_value = np.nanmax(t2) if not np.isnan(np.nanmax(t2)) else 0
        #     min_value = np.nanmin(t2) if not np.isnan(np.nanmax(t2)) else 0
        #     ymin = 0
        #     ymax = 1.1
        #     if "time" in cm_name:
        #         ymax = 0.1

        #     if ymin > min_value:
        #         ymin = min_value

        #     if ymax < max_value and not np.isinf(max_value):
        #         ymax = max_value + (max_value / 11.0)

        #     axis.axis(ymin = ymin, ymax = ymax)
        #     axis.axis(xmin = 1, xmax = 20)
        #     plt.savefig(f'{path}/{cm_name}.png', dpi=300)

        df = pd.read_csv(f'{path}/Sigma mean.csv')
        # df2 = pd.read_csv(f'{path} line/Sigma mean.csv')

        # df['Data'] /= df2['Data']

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(12, 6)
        axis.grid(True)
        axis.plot(df['N'], df['Data'], marker='o')
        axis.set_xlabel("N", loc = 'right')
        axis.axis(xmin = 0.9, xmax = 20.1)
        # axis.legend(loc='upper right')
        plt.savefig(f'{path}/Sigma mean.png', dpi=300)
        

    
    def average_elements_np(self, arr, i):
        if i == 1:
            return arr
        arr = np.array(arr)
        n_segments = arr.shape[0] // i
        averaged = np.mean(arr[:n_segments*i].reshape(-1, i, arr.shape[1]), axis=1)
        return averaged

    def read_channel_data(self, base_path, file_pattern):
        file_name = f'{file_pattern}.csv'
        full_path = f'{base_path}/All Mean/CSV/{file_name}'
        data = pd.read_csv(full_path)["Data"]
        return np.array(data)

    def NoAllSigma(self, a_path, all_matrix):
        logger.debug("No All Sigma")
        path = f'{a_path}/All Mean'
        self.process_matrix(all_matrix, "All", path)

    def form_res_array(self, arrays, t):
        res = []
        i = 0
        for key, value in arrays.items():
            if i == 11:
                break
            if key != t:
                res.append(value[0])
            i = i + 1 
        return res
    
    def process_matrix(self, matrix, matrix_type, path):
        lstart = time.time()
        data_matrix = np.transpose(matrix)
        all_mean_data = np.mean(data_matrix, axis=1)
        all_sigma_data = np.std(data_matrix, axis=1, ddof=1)
        lend = time.time()
        self.ltime = (lend-lstart)*10**3
        title = f"{matrix_type} Mathematical Expectation"
        self.plot_to_csv(path, all_mean_data, title)
        # self.plot_to_png(path, all_mean_data, title)
        title = f"{matrix_type} Sigma"
        self.plot_to_csv(path, all_sigma_data, title)
        # self.plot_to_png(path, all_sigma_data, title)

    def plot_to_csv(self, path, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{path}/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(plot), 1) / 360
        data = pd.DataFrame({"Time" : time, "Data" : plot})
        nk.write_csv(data, f'{path}/{name}.csv')

    def plot_to_png(self, path, plot, name, xlim = None, ylim = None, ytext="", xtext="", size=(12, 6)):
        logger.info(f"Plot {name}.png")
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        time = np.arange(0, len(plot), 1) / 360
        axis.plot(time, plot, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.set_title(ytext, loc = 'left', fontsize=14, position=(-0.06, 0))
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)

        # plt.clf()
        # plt.rcParams.update({'font.size': 15})
        # f, axis = plt.subplots()
        # f.set_size_inches(19, 6)
        # f.tight_layout()
        # axis.grid(True)
        # m_time = np.arange(0, len(np.concatenate(all_data)), 1) / 360
        # axis.plot(m_time, np.concatenate(all_data), linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
        # for P_Peaks in range(0,6):
        #     axis.axvline(x = P_Peaks, color = '#d62728', linewidth=4)
        # axis.set_xlabel("$t, s$", loc = 'right')
        # axis.legend(loc='upper right')
        # axis.axis(xmin = -0.1, xmax = 5.1)
        # # plt.show()
        # plt.savefig(f'{self.a_path}/test.png', dpi=300)

        # mean = filtered_data[10]
        # mean = filtered_other_data[10]

        # plt.clf()
        # plt.rcParams.update({'font.size': 15})
        # f, axis = plt.subplots(1)
        # f.set_size_inches(12, 6)
        # f.tight_layout()
        # axis.grid(True)
        # m_time = np.arange(0, len(mean), 1) / 360
        # axis.plot(m_time, mean, linewidth=3, label=r"$\xi_{{\omega}} (t), mV$")
        # axis.plot(m_time, a_upper_bounds, linewidth=3, label=r"$Upper_{{\xi}} (t), mV$")
        # axis.plot(m_time, a_lower_bounds, linewidth=3, label=r"$Lower_{{\xi}} (t), mV$")
        # axis.set_xlabel("$t, s$", loc = 'right')
        # axis.axis(xmin = 0, xmax = 1)
        # axis.legend(loc='upper right')
        # # plt.savefig(f'{self.a_path}/Other-Authentication.png', dpi=300)
        # plt.savefig(f'{self.a_path}/Authentication.png', dpi=300)