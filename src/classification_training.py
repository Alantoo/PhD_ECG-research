"""
ClassificationTraining:
  Призначення:
    - Агрегує вхідні вектори сигналів (cycles) та їхні мітки pathology_label.
    - Підтримує обмеження кількості доданих циклів (cycles_limit) та групове усереднення (average_coef).
      * cycles_limit: обрізає кількість отриманих циклів.
      * average_coef > 1: об'єднує послідовні групи циклів та усереднює їх покомпонентно.
    - Зберігає накопичені дані в numpy масивах для ефективної обробки.
    - Виконує розбиття даних на тренувальну та тестову вибірки (train_test_split).
    - Навчає набір базових класифікаторів (kNN, SVM, Gaussian Process, Decision Tree, Random Forest, MLP, AdaBoost)
      всередині пайплайна зі StandardScaler.
    - Зберігає кожну навчену модель у директорії models_dir у форматі .joblib для подальшого використання.
    - Надає доступ до тестових даних через getTestData.
  Основні методи:
    setTrainData(new_data, pathology_label, cycles_limit=None, average_coef=None):
        Додає нові цикли з опціональною попередньою обробкою.
    runTraining():
        Розбиває дані, навчає класифікатори та зберігає моделі.
    getTestData():
        Повертає тестовий піднабір.
"""
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import numpy as np
import os
import joblib

class ClassificationTraining:
    def __init__(self):
        self.data = None
        self.pathology_labels = None
        self.data_train = None
        self.data_test = None
        self.pathology_labels_train = None
        self.pathology_labels_test = None
        self.models_dir = "trained_classifiers"
        self.names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
        ]
        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            GaussianProcessClassifier(1.0 * RBF(2.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(learning_rate=0.5, n_estimators=150),
        ]


    def runTraining(self):
        classifier_model_paths = []
        if self.data is None or self.pathology_labels is None or self.data.shape[0] < 2:
            logger.warning("Not enough data to train.")
            return
        logger.debug("Training started.")
        self.data_train, self.data_test, self.pathology_labels_train, self.pathology_labels_test = train_test_split(
            self.data, self.pathology_labels, test_size=0.3, random_state=42
        )
        os.makedirs(self.models_dir, exist_ok=True)
        for idx, (name, clf) in enumerate(zip(self.names, self.classifiers)):
            pipeline = make_pipeline(StandardScaler(), clf)
            pipeline.fit(self.data_train, self.pathology_labels_train)
            model_filename = f"{idx:02d}_{name.lower().replace(' ', '_')}.joblib"
            model_path = os.path.join(self.models_dir, model_filename)
            joblib.dump(pipeline, model_path)
            classifier_model_paths.append(model_path)
            logger.debug(f"{name}: model saved to {model_path}.")
        logger.debug("Training completed. Models saved.")
        return classifier_model_paths

    def getTestData(self):
        return getattr(self, "data_test", None), getattr(self, "pathology_labels_test", None)
    
    def setTrainData(self, new_data, pathology_label, cycles_limit=None, average_coef=None):
        if not isinstance(new_data, list):
            raise TypeError("new_data must be a list of vectors")
        if any(not isinstance(vec, (list, tuple)) for vec in new_data):
            raise TypeError("Each item in new_data must be a list or tuple")

        # Convert to numpy array (2D)
        arr = np.asarray(new_data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("new_data must be a rectangular 2D list (all vectors same length)")

        # cycles_limit
        if cycles_limit is not None:
            if not isinstance(cycles_limit, int) or cycles_limit < 0:
                raise ValueError("cycles_limit must be a non-negative integer or None")
            arr = arr[:cycles_limit]

        # average_coef
        if average_coef is not None:
            if not isinstance(average_coef, int) or average_coef < 1:
                raise ValueError("average_coef must be a positive integer or None")
        if average_coef and average_coef > 1:
            group_size = average_coef
            n_groups = arr.shape[0] // group_size
            if n_groups == 0:
                logger.debug("No full groups for averaging; nothing added.")
                return
            trimmed = arr[: n_groups * group_size]
            arr = trimmed.reshape(n_groups, group_size, arr.shape[1]).mean(axis=1)

        count = arr.shape[0]
        if count == 0:
            return

        # Append data
        if self.data is None:
            self.data = arr
        else:
            self.data = np.vstack((self.data, arr))

        labels_vec = np.full(count, pathology_label, dtype=int)
        if self.pathology_labels is None:
            self.pathology_labels = labels_vec
        else:
            self.pathology_labels = np.concatenate((self.pathology_labels, labels_vec))

        logger.debug(
            f"Training data updated: +{count} samples (label={pathology_label}, "
            f"limit={cycles_limit}, average_coef={average_coef}). Total={self.data.shape[0]}"
        )
