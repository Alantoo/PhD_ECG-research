from loguru import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class PlotClassifiers():
        
    def __init__(self, ecg_config):
        logger.info("Plot Classifiers")

        # r_data = "1 Mathematical Expectation"
        # r_data = "2 Initial Moments Second Order"
        # r_data = "3 Initial Moments Third Order"
        r_data = "4 Initial Moments Fourth Order"

        # co = "an"
        # co = "bn"
        co = "an_bn"

        path = f'{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}/Confusion matrix/Fourier Series/{r_data}/data/{co}'
        path2 = f'{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}/Confusion matrix/Fourier Series/{r_data}/img/{co}'
        Path(path2).mkdir(parents=True, exist_ok=True)

        confusion_matrix_names = [
            "Accuracy", "Balanced Accuracy", "F1 score", "Learning_time", "Testing_time"
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

        arr = range(1, 31)

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
            ttt = [*range(2, 62, 2)]
            # print(ttt)
            for i, name in zip(t2, names):
                axis.plot(ttt, i, linewidth=2, label=name)
                # axis.plot(i, linewidth=2, label=name)
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

            axis.axis(xmin = 1, xmax = 60)
            # axis.axis(xmin = 1, xmax = 30)
            
            plt.savefig(f'{path2}/{cm_name}.png', dpi=300)