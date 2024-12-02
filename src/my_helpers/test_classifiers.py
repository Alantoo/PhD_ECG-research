import numpy as np
import time
# from scipy.integrate import simps
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
import neurokit2 as nk
from pathlib import Path

class TestClassifiers():
        
    def __init__(self, ecg_config):
        print("Test Classifiers")

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Naive Bayes",
            "QDA"
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
            QuadraticDiscriminantAnalysis()
        ]

        path = f'{ecg_config.getImgPath()}/Segments/7_1_II_norma/ALL.csv'
        data1 = pd.read_csv(path).to_numpy()
        target_data1 = [1] * len(data1)

        path = f'{ecg_config.getImgPath()}/Segments/9_3_II/ALL.csv'
        data2 = pd.read_csv(path).to_numpy()
        target_data2 = [2] * len(data2)

        path = f'{ecg_config.getImgPath()}/Segments/9_4_II/ALL.csv'
        data3 = pd.read_csv(path).to_numpy()
        target_data3 = [3] * len(data3)

        matrix = [*data1, *data2, *data3]
        target_matrix = [*target_data1, *target_data2, *target_data3]

        data_train, data_test, target_values_train, target_values_test = train_test_split(matrix, target_matrix, test_size=0.3, random_state=42)

        _l1 = []
        _t1 = []
        _t2 = []

        for name, clf in zip(names, classifiers):
            clf = make_pipeline(StandardScaler(), clf)
            lstart = time.time()
            clf.fit(data_train, target_values_train)
            lend = tstart = time.time()
            score = clf.score(data_test, target_values_test)
            tend = time.time()
            _l1.append(score * 100)
            _t1.append((lend-lstart)*10**3)
            _t2.append((tend-tstart)*10**3)
            print(("%s: %.2f" % (name, score * 100)))
            print(f'Learning time: {(lend-lstart)*10**3:.03f} ms | Testing time: {(tend-tstart)*10**3:.03f} ms')

        l = pd.DataFrame({"Classifier": names, "Score" : np.round(_l1, 2), "Learning_time" : np.round(_t1, 3), "Testing_time" : np.round(_t2, 3)})
        path = f'{ecg_config.getImgPath()}/Segments/7_1_II_norma/TestClassifiers.csv'
        nk.write_csv(l, path)

        
        