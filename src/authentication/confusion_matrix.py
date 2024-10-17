from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import numpy as np

class ConfusionMatrix():
    
    def __init__(self, y_true, y_pred, ltime, ttime):

        CM = confusion_matrix(y_true, y_pred)

        TN, FP, FN, TP = CM.ravel()

        # True Positive Rate
        TPR = TP / (TP + FN)

        # True Negative Rate
        TNR = TN / (TN + FP)

        # Positive Predictive Value
        PPV = TP / (TP + FP)

        # # Negative Predictive Value
        # NPV = TN / (TN + FN)

        # False Negative Rate
        FNR = FN / (FN + TP)

        # False Positive Rate
        FPR = FP / (FP + TN)

        # # False Discovery Rate
        # FDR = FP / (FP + TP)

        # # False Omission Rate
        # FOR = FN / (FN + TN)

        # # Positive Likelihood Ratio
        # LR_P_ = TPR / FPR

        # # Negative Likelihood Ratio
        # LR_N_ = FNR / TNR

        # # Prevalence Threshold
        # PT = np.sqrt(FPR) / (np.sqrt(TPR) + np.sqrt(FPR))

        # # Threat Score
        # TS = TP / (TP + FN + FP)

        # Accuracy
        ACC = (TP + TN) / (TP + TN + FP + FN)

        # Balanced Accuracy
        BA = (TPR + TNR) / 2.0

        # F1 score
        # F1 = 2 * ((PPV * TPR) / (PPV + TPR))
        F1 = (2 * TP) / ((2 * TP) + FP + FN)

        # # Matthews Correlation Coefficient
        # # MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        # MCC = matthews_corrcoef(y_true, y_pred)

        # # Fowlkes-Mallows index
        # FM = np.sqrt(PPV * TPR)

        # # Bookmaker Informedness
        # BM = TPR + TNR - 1

        # # Markedness
        # MK = PPV + NPV - 1

        # # print(class_likelihood_ratios(y_true, y_pred))

        # # Diagnostic Odds Ratio
        # DPR = LR_P_ / LR_N_

        self.CM = CM
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.TP = TP
        self.TPR = TPR
        self.TNR = TNR
        self.PPV = PPV
        # self.NPV = NPV
        self.FNR = FNR
        self.FPR = FPR
        # self.FDR = FDR
        # self.FOR = FOR
        # self.LR_P_ = LR_P_
        # self.LR_N_ = LR_N_
        # self.PT = PT
        # self.TS = TS
        self.ACC = ACC
        self.BA = BA
        self.F1 = F1
        # self.MCC = MCC
        # self.FM = FM
        # self.BM = BM
        # self.MK = MK
        # self.DPR = DPR
        self.ltime = ltime
        self.ttime = ttime

    def getCM(self):
        return self.CM

    def getTN(self):
        return self.TN

    def getFP(self):
        return self.FP

    def getFN(self):
        return self.FN

    def getTP(self):
        return self.TP

    def getTPR(self):
        return self.TPR

    def getTNR(self):
        return self.TNR

    def getPPV(self):
        return self.PPV

    # def getNPV(self):
    #     return self.NPV

    def getFNR(self):
        return self.FNR

    def getFPR(self):
        return self.FPR

    # def getFDR(self):
    #     return self.FDR

    # def getFOR(self):
    #     return self.FOR

    # def getLR_P_(self):
    #     return self.LR_P_

    # def getLR_N_(self):
    #     return self.LR_N_

    # def getPT(self):
    #     return self.PT

    # def getTS(self):
    #     return self.TS

    def getACC(self):
        return self.ACC

    def getBA(self):
        return self.BA

    def getF1(self):
        return self.F1

    # def getMCC(self):
    #     return self.MCC

    # def getFM(self):
    #     return self.FM

    # def getBM(self):
    #     return self.BM

    # def getMK(self):
    #     return self.MK

    # def getDPR(self):
    #     return self.DPR

    def getAllVariables(self):
        return [
            self.TPR, self.TNR, self.FNR, self.FPR,
            self.ACC, self.BA, self.F1, np.round(self.ltime, 3), np.round(self.ttime, 3)
        ]