from get_config.ecg_config import ECGConfig
from loguru import logger
from my_helpers.read_data.read_data_file import ReadDataFile
import numpy as np
import time
from scipy.integrate import simps
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
from classifiers_test.confusion_matrix import ConfusionMatrix
import classifiers_test.test_module as test_module

class Test():

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        logger.debug("Test Init")

    def TestRed(self, i):
        logger.debug("Test")
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
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio", "Learning_time", "Testing_time"
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

        r_data = "1 Mathematical Expectation"
        # r_data = "2 Initial Moments Second Order"
        # r_data = "3 Initial Moments Third Order"
        # r_data = "4 Initial Moments Fourth Order"

        test_data = []
        target_data = []
        all_f_time = 0
        for conf in test_module.read_files:
            ecg_conf = ECGConfig(conf)
            path = f'{ecg_conf.getImgPath()}/{ecg_conf.getConfigBlock()}/Mathematical Statistics/CSV/{r_data}.csv'
            df = pd.read_csv(path)

            fstart = time.time()
            an, bn = self.getFourierSeries(df["Data"], 360, terms=i)
            fend = tstart = time.time()

            ftime = (fend-fstart)*10**3

            all_f_time = all_f_time + ftime

            # test_data.append(an)
            # test_data.append(bn)
            test_data.append([*an, *bn])
            # print([*an, *bn])
            # test_data.append(df["Data"])
            target_data.append(ecg_conf.getPathology())

        f_train = all_f_time * 0.7
        f_test = all_f_time * 0.3


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
            ltime = ((lend-lstart)*10**3) + f_train
            ttime = ((tend-tstart)*10**3) + f_test
            confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, ttime)
            res = confusion_matrix.getAllVariables()
            _cm.append(res)


        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier Series/{r_data}/data/an'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier Series/{r_data}/data/bn'
        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Fourier Series/{r_data}/data/an_bn'
        # path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/{r_data}/data'
        Path(path).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(np.transpose(np.round(_cm, 4)), index=confusion_matrix_names, columns=names)
        df.to_csv(f'{path}/n-{i}.csv') 


    def getFourierSeries(self, y, sampling_rate, terms = 30, L = 1,):
        x = np.linspace(0, L, sampling_rate, endpoint=False)
        # a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
        list_a = np.abs([*[an(k) for k in range(1, terms + 1)]])
        list_b = np.abs([*[bn(k) for k in range(1, terms + 1)]])
        return list_a, list_b
