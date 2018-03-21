#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
import pickle as pkl
from preprocess import get_dataset_info
import sys
from numpy.linalg import norm

CLASS_NAMES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
def test_features(dataset_name):
    gt_pkl = '../pkls/vot_{}_gt.pkl'.format(dataset_name)
    classes_gt, features_gt = pkl.load(open(gt_pkl, 'rb'))
    feature_summary = { classname:[] for classname in CLASS_NAMES }
    feature_all = []
    for classes, features in zip(classes_gt, features_gt):
        for classname, feature in zip(classes, features):
            newfeature = feature / norm(feature)
            #newfeature = feature
            feature_summary[classname].append(newfeature)
            feature_all.append(newfeature)
    for classname in CLASS_NAMES:
        features = np.array(feature_summary[classname])
        average_feature = features.mean(axis=0)
        bias_feature = np.sum((features - average_feature)**2, axis=1).mean()
        print(classname, ':', bias_feature)
    features = np.array(feature_all)
    average_feature = features.mean(axis=0)
    bias_feature = np.sum((features - average_feature)**2, axis=1).mean()
    print('total :', bias_feature)
    return None

def extract_gt_features(dataset_name):
    sys.path.append('faster_rcnn_pytorch')
    from extract_feature_from_bb import extractfeatures, build_extractor
    model_file = 'faster_rcnn_pytorch/models/VGGnet_fast_rcnn_iter_70000.h5'
    extractor = build_extractor(model_file)

    img_paths, gts = get_dataset_info(dataset_name) 
    features_gt = []
    classes_gt = []
    for img_path, gt in zip(img_paths, gts):
        classes = [ g[0] for g in gt]
        dets = np.array([[0.] + [g[1][1], g[1][0], g[1][3], g[1][2]] for g in gt], dtype='float32')
        #dets = np.array([[0.] + list(g[1]) for g in gt], dtype='float32')
        DEBUG = False
        if DEBUG:
            img = cv2.imread(img_path)
            for det in dets:
                cv2.rectangle(img, (int(det[1]), int(det[2])), (int(det[3]), int(det[4])), (0, 204, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        feature_gt = extractfeatures(img_path, extractor, dets)
        feature_gt = feature_gt.data.cpu().numpy()
        classes_gt.append(classes)
        features_gt.append(feature_gt)
    gt_pkl = '../pkls/vot_{}_gt.pkl'.format(dataset_name)
    pkl.dump([classes_gt, features_gt], open(gt_pkl, 'wb'))

if __name__ == '__main__':
    dataset_name = 'test'
    extract_gt_features(dataset_name)
    test_features(dataset_name)
