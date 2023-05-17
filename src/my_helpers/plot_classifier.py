import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import ast

class PlotClassifier():
        
    def __init__(self, ecg_config, data):
        print("Plot Classifiers")
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Gaussian Naive Bayes",
            "Quadratic Discriminant Analysis"
        ]

        plot_path = f'{ecg_config.getImgPath()}/{data.getSigNameDir()}'
        l = pd.read_csv(f'{plot_path}/Classifiers/L1.csv')

        l1 = np.transpose([ast.literal_eval(i) for i in l.l1])
        t1 = np.transpose([ast.literal_eval(i) for i in l.t1])
        t2 = np.transpose([ast.literal_eval(i) for i in l.t2])

        # plt.clf()
        # plt.rcParams.update({'font.size': 14})
        # f, axis = plt.subplots(1)
        # f.tight_layout()
        # f.set_size_inches(19, 6)
        # axis.grid(True)
        # axis.set_xlabel("Number of coefficients", loc = 'right')
        # axis.set_title("Efficiency, %", loc = 'left', fontsize=14, position=(-0.06, 0))
        # for i, name in zip(l1, names):
        #     axis.plot([3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], i, linewidth=2, label=name)
        # axis.legend(loc='best',prop={'size':10})
        # axis.axis(ymin = 40, ymax = 95)
        # axis.axis(xmin = 3, xmax = 60)
        # Path(f'{plot_path}/Classifiers').mkdir(parents=True, exist_ok=True)
        # plt.savefig(f'{plot_path}/Classifiers/Efficiency.png', dpi=300)

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(19, 6)
        axis.grid(True)
        axis.set_xlabel("Number of coefficients", loc = 'right')
        axis.set_title("$t, ms$", loc = 'left', fontsize=14, position=(-0.06, 0))
        for i, name in zip(t1, names):
            axis.plot([3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], i, linewidth=2, label=name)
        axis.legend(loc='best',prop={'size':10})
        axis.axis(ymin = 0, ymax = 100)
        axis.axis(xmin = 3, xmax = 60)
        Path(f'{plot_path}/Classifiers').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'{plot_path}/Classifiers/Learn time.png', dpi=300)

        # plt.clf()
        # plt.rcParams.update({'font.size': 14})
        # f, axis = plt.subplots(1)
        # f.tight_layout()
        # f.set_size_inches(19, 6)
        # axis.grid(True)
        # axis.set_xlabel("Number of coefficients", loc = 'right')
        # axis.set_title("$t, ms$", loc = 'left', fontsize=14, position=(-0.06, 0))
        # for i, name in zip(t2, names):
        #     axis.plot([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], i, linewidth=2, label=name)
        # axis.legend(loc='best',prop={'size':10})

        # axis.axis(xmin = 5, xmax = 60)
        # Path(f'{plot_path}/Classifiers').mkdir(parents=True, exist_ok=True)
        # plt.savefig(f'{plot_path}/Classifiers/Test time.png', dpi=300)