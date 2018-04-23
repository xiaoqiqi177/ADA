import os
import sys
import torch
import cv2
import pickle
import numpy as np
from tqdm import tqdm

def iou(bb1, bb2):
    w = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    h = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    overlap = w * h
    s1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    s2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou_value = overlap / (s1 + s2 - overlap)
    return iou_value

if __name__ == '__main__':
    output_dir = './output/'+sys.argv[1] + '_'+ sys.argv[2]+'_'+sys.argv[3] + '_' + sys.argv[4]
    
    if os.path.exists(output_dir) is False:
        print('exists not outputdir {}.'.format(output_dir))
        exit(0)

    with open(os.path.join(output_dir, 'result.pkl'), 'rb') as f:
        detected_bounding_boxes, gt_bounding_boxes = pickle.load(f)

    ma_total = 0
    ma_right = 0
    healthy_total = 0
    healthy_right = 0

    img_keys = detected_bounding_boxes.keys()
    for key in img_keys:
        detected = detected_bounding_boxes[key]
        groundtruth = gt_bounding_boxes[key]
        sum_gt_bb = 0
        sum_detected_bb = 0
        sum_overlap = 0

        for patch_detected, patch_gt in zip(detected, groundtruth):
            for bbox_detected in patch_detected:
                for bbox_gt in patch_gt:
                    iou_value = iou(bbox_detected, bbox_gt)
                    if iou_value > 0.5:
                        sum_overlap += 1
            
            sum_detected_bb += len(patch_detected)
            sum_gt_bb += len(patch_gt)
        if sum_gt_bb > 0:
            ma_total += 1
            if sum_detected_bb > 0:
                ma_right += 1
        else:
            healthy_total += 1
            if sum_detected_bb == 0:
                healthy_right += 1
        print('sum_gt_bb / sum_detected_bb / sum_overlap: {} / {} / {}'.format(sum_gt_bb, sum_detected_bb, sum_overlap))
    print('ma_right / ma_total: {} / {}'.format(ma_right, ma_total))
    print('healthy_right / healthy_total: {} / {}'.format(healthy_right, healthy_total))
