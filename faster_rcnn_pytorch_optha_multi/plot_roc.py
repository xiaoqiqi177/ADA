import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import os
import pickle as pkl
##############################################################################
# Plot of a ROC curve for a specific class
def plot_roc(title, savefile, y_pred, y_true):
    #add wsdcnn results
    wsdcnn_dir = '/home/qiqix/EyeWeS/wsdcnn/experiments/wsdcnn15'
    wsdcnn_prediction_path = os.path.join(wsdcnn_dir, 'test_predictions.pkl')
    y_true_e, y_pred_e = pkl.load(open(wsdcnn_prediction_path, 'rb'))
    
    plt.figure()
    lw = 1
    fp, tp, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fp, tp)
    plt.plot(fp, tp, color='darkorange',
            lw=lw, label='Our ROC curve (area = %0.4f)' % roc_auc)
    
    fp_e, tp_e, _ = roc_curve(y_true_e, y_pred_e)
    eyewes_roc_auc = auc(fp_e, tp_e)
    plt.plot(fp_e, tp_e, color='turquoise',
            lw=lw, label='EyeWeS ROC curve (area = %0.4f)' % eyewes_roc_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(savefile)
