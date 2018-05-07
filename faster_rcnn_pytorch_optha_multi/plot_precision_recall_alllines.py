import matplotlib.pyplot as plt
import sys
from itertools import cycle
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='train ma dataset')
parser.add_argument('--datasetnames', nargs='+', required=True)
parser.add_argument('--ratio-name', default='721', type=str)
parser.add_argument('--task-name', default='ma_double', type=str)
parser.add_argument('--method-names', nargs='+', required=True)

args = parser.parse_args()

def plot_precision_recall(precisions, recalls, title, savefile):
    lt.figure()
    plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                 color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{}'.format(title))
    #plt.title('{}: AP={}'.format(title, average_precision))
    plt.savefig(savefile)

def plot_precision_recall_all(precisions_all, recalls_all, titles, savefile):
    colors = cycle(['gold', 'navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    n_number = len(precisions_all)
    for i, color in zip(range(len(recalls_all)), colors):
        l, = plt.plot(recalls_all[i], precisions_all[i], color=color, lw=1)
        lines.append(l)
        labels.append('Precision-recall for {}'.format(titles[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves of different experiments')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(savefile)

def main():
    precisions_set = []
    recalls_set = []
    title_set = ['NegFRCNN with finetune', 'FRCNN with finetune', 'NegFRCNN without finetune', 'FRCNN without finetune', 'NegFRCNN with finetune', 'NegFRCNN with finetune block4']
    title_set = [ title + ' on ' + datasetname for title, datasetname in zip(title_set, args.datasetnames)]
    for method_name, datasetname in zip(args.method_names, args.datasetnames):
        logname = 'log_'+datasetname + '_' + method_name
        f = open(logname, 'r')
        lines = f.readlines()
        object_number = len(lines) // 5
        recalls = []
        precisions = []
        for o_id in range(object_number):
            recall = float(lines[o_id*5+2].split(':')[-1])
            precision = float(lines[o_id*5+3].split(':')[-1])
            recalls.append(recall)
            precisions.append(precision)
        #title = method_name + ' on '+datasetname
        precisions_set.append(precisions)
        recalls_set.append(recalls)
        #title_set.append(title)
    plot_precision_recall_all(precisions_set, recalls_set, title_set, 'final_recall_precision.png') 

if __name__ == '__main__':
    main()
