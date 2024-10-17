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
import scipy.interpolate as interp

class Test():

    def __init__(self, ecg_config):
        self.ecg_config = ecg_config
        logger.debug("Test Init")

    def TestRed(self):
        logger.debug("Test src")
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


        test_data = []
        target_data = []
        for conf in test_module.read_files:
            ecg_conf = ECGConfig(conf)
            data = ReadDataFile(ecg_conf)
            mod_sampling_rate = int(data.sampling_rate * self.ecg_config.getMultiplier())
            signal = data.signals[self.ecg_config.getSigName()]
            two_min_signal = signal[:120 * data.sampling_rate]
            arr = np.array(two_min_signal)
            arr_interp = interp.interp1d(np.arange(arr.size), arr)
            arr_stretch = arr_interp(np.linspace(0, arr.size - 1, 360 * 120))

            test_data.append(arr_stretch)
            target_data.append(ecg_conf.getPathology())


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


        path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}/Confusion matrix/Src Data/data'
        Path(path).mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(np.transpose(np.round(_cm, 4)), index=confusion_matrix_names, columns=names)
        df.to_csv(f'{path}/n-all.csv')


