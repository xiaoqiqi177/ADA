import os
import sys
import torch
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from plot_roc import plot_roc

def iou(bb1, bb2):
    w = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    h = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    overlap = w * h
    s1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    s2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou_value = overlap / (s1 + s2 - overlap)
    return iou_value

import argparse
parser = argparse.ArgumentParser(description='train ma dataset')
parser.add_argument('--datasetname', default='trainval', type=str)
parser.add_argument('--ratio-name', default='721', type=str)
parser.add_argument('--task-name', default='ma_double', type=str)
parser.add_argument('--method-name', default='10000', type=str)
parser.add_argument('--thresh', default=0.5, type=float)

args = parser.parse_args()

def _evaluate_predictions(y_true, y_pred, metric_fn):
    if isinstance(y_true, list):
        return [metric_fn(yti, ypi) for yti, ypi in zip(y_true, y_pred)]
    return metric_fn(y_true, y_pred)


def _evaluate_metric(y_true, y_pred, metric, output_dir, thresh_id):
    if metric == 'AUC':
        plot_roc('ROC curve on image-level', os.path.join(output_dir, 'roc_{}.png'.format(thresh_id)), y_pred, y_true) 
        return _evaluate_predictions(y_true, y_pred, roc_auc_score)
    if metric == 'AP':
        return _evaluate_predictions(y_true, y_pred, average_precision_score)
    raise ValueError('Unknown metric. Given {} but only know how to compute AUC and AP')


def evaluate(y_true, y_pred, output_dir, thresh_id, metrics=['AUC']):
    if y_true is None or y_pred is None:
        return [], ([], [])

    results = []
    for metric in metrics:
        results.append(_evaluate_metric(y_true, y_pred, metric, output_dir, thresh_id))

    return results

if __name__ == '__main__':
    output_dir = './output/'+ args.datasetname + '_'+ args.ratio_name+'_'+ args.task_name + '_' + args.method_name
    
    if os.path.exists(output_dir) is False:
        print('exists not outputdir {}.'.format(output_dir))
        exit(0)

    with open(os.path.join(output_dir, 'result.pkl'), 'rb') as f:
        detected_bboxes, gt_bboxes = pickle.load(f)
    
    thresh_num = len(detected_bboxes[0])
    y_true_all = [[] for _ in range(thresh_num)]
    y_pred_all = [[] for _ in range(thresh_num)]
    
    sum_overlaps_all = [0. for _ in range(thresh_num)]
    sum_gt_bbs_all = [0. for _ in range(thresh_num)]
    sum_detected_bbs_all = [0. for _ in range(thresh_num)]
    y_true_pred_all = [0. for _ in range(thresh_num)]

    for img_id, (cls_dets_all_all, gt_dets) in enumerate(zip(detected_bboxes, gt_bboxes)):
        for thresh_id, cls_dets_all in enumerate(cls_dets_all_all):
            gt_bb = 0
            overlap = {}
            for bbox_detected in cls_dets_all:
                for gt_id, bbox_gt in enumerate(gt_dets):
                    iou_value = iou(bbox_detected, bbox_gt)
                    if iou_value > 0.5:
                        overlap[gt_id] = 1
        
            gt_bb = len(gt_dets)
            detected_bb = len(cls_dets_all)
            cls_dets_all = np.array(cls_dets_all)
            if detected_bb > 0:
                y_pred_all[thresh_id].append(np.clip(cls_dets_all[:,-1].sum()/15., 0., 1.))
            else:
                y_pred_all[thresh_id].append(0.)

            if gt_bb > 0:
                y_true_all[thresh_id].append(1)
            else:
                y_true_all[thresh_id].append(0)
            sum_overlaps_all[thresh_id] += len(overlap)
            sum_gt_bbs_all[thresh_id] += gt_bb
            sum_detected_bbs_all[thresh_id] += detected_bb

    thresh_id = 0
    for y_true, y_pred, sum_overlaps, sum_gt_bbs, sum_detected_bbs in zip(y_true_all, y_pred_all, sum_overlaps_all, sum_gt_bbs_all, sum_detected_bbs_all):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred).reshape(-1, 1)
        results = evaluate(y_true, y_pred, output_dir, thresh_id, metrics=['AUC', 'AP'])
        thresh_id += 1
        if sum_detected_bbs == 0.:
            continue
        print('AUC: ', results[0])
        print('AP: ', results[1])
        total_recall = sum_overlaps / sum_gt_bbs
        print('total_recall: ', total_recall)
        total_precision = sum_overlaps / sum_detected_bbs
        print('total_precison: ', total_precision)
        print('total_F1: ', 2 * total_precision * total_recall / (total_precision + total_recall))
