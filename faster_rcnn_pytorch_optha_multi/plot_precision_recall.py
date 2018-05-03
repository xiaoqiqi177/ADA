import matplotlib.pyplot as plt
import sys

def plot_precison_recall(precisions, recalls, title, savefile):
    plt.figure()
    plt.step(recalls, precisions, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2,
                 color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('{}: AP={}'.format(title, average_precision))
    plt.savefig(savefile)
    
def main():
    logname = sys.argv[1]
    f = open(logname, 'r')
    lines = f.readlines()
    object_number = len(lines) // 7
    recalls = []
    precisions = []
    for o_id in range(object_number):
        thresh = lines[o_id*7+1]
        recall = lines[o_id*7+4]
        precision = lines[o_id*7+5]
        recalls.append(recall)
        precisions.append(precision)
    plot_precision_recall(precisions, recalls, 'Precision-Recall curve', sys.argv[2]) 

if __name__ == '__main__':
    main()
