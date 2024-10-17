from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class PlotClassifiers():
        
    def __init__(self, ecg_config):
        logger.info("Plot Classifiers")

        # fft = "Fourier coefficients"
        fft = "Polyspectrums"
        ffr = "Real and Imag parts"
        # ffr = "Imag parts"
        # ffr = "Real parts"

        path = f'{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}/Confusion matrix/{fft}/data/{ffr}'
        path2 = f'{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}/Confusion matrix/{fft}/img/{ffr}'
        Path(path2).mkdir(parents=True, exist_ok=True)

        confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness"
        ]

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Gaussian Naive Bayes"
        ]

        arr = range(1, 201)

        for cm_name in confusion_matrix_names:
            read_data = []
            for i in arr:
                df = pd.read_csv(f'{path}/n-{i}.csv')
                accuracy_row = df.loc[df['Unnamed: 0'] == cm_name]
                accuracy_array = accuracy_row.values.flatten()[1:]
                read_data.append(accuracy_array)

            t2 = np.transpose(np.array(read_data))

            plt.clf()
            plt.rcParams.update({'font.size': 14})
            f, axis = plt.subplots(1)
            f.tight_layout()
            f.set_size_inches(19, 6)
            axis.grid(True)
            axis.set_xlabel("Number of coefficients", loc = 'right')
            # axis.set_title("$t, ms$", loc = 'left', fontsize=14, position=(-0.06, 0))
            ttt = [*range(2, 402, 2)]
            print(ttt)
            for i, name in zip(t2, names):
                axis.plot(ttt, i, linewidth=2, label=name)
            axis.legend(loc='best',prop={'size':10})

            # max_value = np.nanmax(t2, initial=0)
            # min_value = np.nanmin(t2, initial=0)

            max_value = np.nanmax(t2) if not np.isnan(np.nanmax(t2)) else 0
            min_value = np.nanmin(t2) if not np.isnan(np.nanmax(t2)) else 0

            ymin = 0
            ymax = 1.1

            if ymin > min_value:
                ymin = min_value

            if ymax < max_value and not np.isinf(max_value):
                ymax = max_value

            axis.axis(ymin = ymin, ymax = ymax)

            axis.axis(xmin = 1, xmax = 400)
            
            plt.savefig(f'{path2}/{cm_name}.png', dpi=300)