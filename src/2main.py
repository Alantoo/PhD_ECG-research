import numpy as np
from matplotlib import pyplot as plt

from classification_signal import EnsembleSignalClassifier
from classification_tester import SavedModelsTester
from classification_training import ClassificationTraining
import os
import json

from preparedSignal import PreparedSignal
from simulation import show_plot


def load_vectors_from_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return []
    # Якщо вже список векторів
    if isinstance(obj, list) and all(isinstance(x, (list, tuple)) for x in obj):
        return obj
    # Якщо dict і є ключ 'slices'
    if isinstance(obj, dict):
        if 'slices' in obj and isinstance(obj['slices'], list):
            cand = obj['slices']
            if all(isinstance(x, (list, tuple)) for x in cand):
                return cand
        # Скан інших значень
        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, (list, tuple)) for x in v):
                return v
    return []


def train():
    classifier = ClassificationTraining()
    # downloads_dir = "/home/roman/Завантаження/ECG2"
    downloads_dir = "/Users/alantoo/Workspace/Edu/ECG-research/src/ecg_slices"
    file_labels = {
        "1_P_П1_Екс_1хв_2сигн_AHA_0201++.csv_slices.json": 1,  # ??
        "1_P_П3_РитмAF_шк_1 хв_2сигн_N 03++.csv_slices.json": 2,  # ??
        # "2_Р_П5_2_Ш_екстр_10c(лише є патал)_2сигн_М-В_Supr_AR_812++.csv_slices.json": 1, #??
        "2_P_П1_Екс_1хв_1сигн_ANSI_3b++.csv_slices.json": 1,
        "2_P_П3_РитмAF_шк_1 хв_2сигн_N 07++.csv_slices.json": 2,
        # "3_Р_П5_3_Надш_екстр_10c(лише є патал)_2сигн_М-В_Supr_AR_820++.csv_slices.json": 1,  #??
        "3_P_П1_Екс_1хв_1сигн_AHSI_3с++.csv_slices.json": 1,
        "3_P_П3_РитмAF_шк_1 хв_2сигн_N 08++.csv_slices.json": 2,
        "4_Р_П_8_Спарена_екстр_BIDM PPG_Resp_1 хв_5сигн_26++.csv_slices.json": 3,  # ??
        "4_P_П1_Екс_1хв_1сигн_ANSI_3d++.csv_slices.json": 1,
        "5_Р_П_8_Част_шл_екстр_BIDM PPG_Resp_1 хв_5сигн_33++.csv_slices.json": 4,  # ??
        "5_P_П1_Екс_1хв_2_сигн_BIDMC_06++.csv_slices.json": 1,
        "6_М_П_4_Морфологія_блокада_PTB_1 хв_12сигн_014 ++.csv_slices.json": 5,
        "7_М_П_4_Морфологія_блокада_PTB_1 хв_12сигн_015 ++.csv_slices.json": 5,
    }
    target_len = None
    for fname, pathology_label in file_labels.items():
        path = os.path.join(downloads_dir, fname)
        if not os.path.isfile(path):
            print(f"File missing: {path}")
            continue
        vectors = load_vectors_from_json(path)
        if not vectors:
            print(f"No usable vectors in: {path}")
            continue

        signal = np.concatenate(vectors)

        prepared = PreparedSignal(signal, 500)
        test_data, _ = prepared.get_interpolated_matrix()
        classifier.setTrainData(test_data, pathology_label=pathology_label)

    classifier.runTraining()
    test_data, test_labels = classifier.getTestData()

    testing = SavedModelsTester()
    result = testing.testData(
        model_paths=None,
        data_test=test_data,
        true_labels=test_labels
    )
    print(f"Testing result: {result}")


def classify():
    global fullname, signal, testing, result
    slice_name = "1_P_П1_Екс_1хв_2сигн_AHA_0201++.csv_slices.json"
    # base_dir = "/Users/alantoo/Workspace/Edu/ECG-research/src/ecg_interpolated"
    base_dir = "/Users/alantoo/Workspace/Edu/ECG-research/src/ecg_slices"
    fullname = os.path.join(base_dir, slice_name)
    signal_matrix = load_vectors_from_json(fullname)

    signal = np.concatenate(signal_matrix)

    prepared = PreparedSignal(signal, 500)
    test_data, _ = prepared.get_interpolated_matrix()

    traw = np.arange(0, len(signal_matrix[0]), 1).tolist()
    show_plot("raw", traw, signal_matrix[0])
    ttest = np.arange(0, len(test_data[0]), 1).tolist()
    show_plot("raw", ttest, test_data[0])

    testing = EnsembleSignalClassifier()
    result = testing.classifySignal(
        model_paths=None,
        test_data=test_data
    )
    print(result)


if __name__ == '__main__':
    # train()

    classify()


def show_plot(title, t_data, sig_data):
    # Візуалізація
    plt.plot(t_data, sig_data, label="Змодельований сигнал")
    plt.xlabel("Час")
    plt.ylabel("Сигнал")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
