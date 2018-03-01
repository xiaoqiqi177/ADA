#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
import pickle as pkl
from preprocess import get_dataset_info
import sys
sys.path.append('faster_rcnn_pytorch')
from extract_feature_from_bb import extractfeatures, build_extractor

def test_features():
    return None

if __name__ == '__main__':
    model_file = 'faster_rcnn_pytorch/models/VGGnet_fast_rcnn_iter_70000.h5'
    extractor = build_extractor(model_file)

    dataset_name = 'trainval'
    img_paths, gts = get_dataset_info(dataset_name) 
    features_gt = []
    classes_gt = []
    for img_path, gt in zip(img_paths, gts):
        classes = [ g[0] for g in gt]
        dets = np.array([[0.] + g[1] for g in gt], dtype='float32')
        feature_gt = extractfeatures(img_path, extractor, dets)
        feature_gt = feature_gt.data.cpu().numpy()
        classes_gt.append(classes)
        features_gt.append(feature_gt)
    gt_pkl = 'vot_{}_gt.pkl'.format(dataset_name)
    pkl.dump([classes_gt, features_gt], open(gt_pkl, 'wb'))
