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
    classname = 'person'
    dets_info = pkl.load(open('vot_{}_bbslist_{}.pkl'.format(dataset_name, classname), 'rb'))
    img_paths = dets_info[0]
    bbslist = dets_info[1]
    features_dets = []
    for img_path, bbs in zip(img_paths, bbslist):
        dets = np.array([[0.] + bb[:-1] for bb in bbs], dtype='float32')
        feature_dets = extractfeatures(img_path, extractor, dets)
        feature_dets = feature_dets.data.cpu().numpy()
        features_dets.append(feature_dets)
    dets_pkl = 'vot_features_{}_bbslist_{}.pkl'.format(dataset_name, classname)
    pkl.dump(features_dets, open(dets_pkl, 'wb'))
