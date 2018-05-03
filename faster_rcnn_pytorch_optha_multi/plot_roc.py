import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp

##############################################################################
# Plot of a ROC curve for a specific class
def plot_roc(title, savefile, y_pred, y_true):
    plt.figure()
    lw = 2
    fp, tp, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fp, tp)
    plt.plot(fp, tp, color='darkorange',
            lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(savefile)
