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
from edge_boxes_with_python.edge_boxes import get_windows

if __name__ == '__main__':

    dataset_name = 'trainval'
    classname = 'person'
    img_paths, gts = get_dataset_info(dataset_name) 
    
    bbslist_pkl = '../pkls/vot_{}_bbslist_{}.pkl'.format(dataset_name, classname)
    if os.path.exists(bbslist_pkl):
        dets_info = pkl.load(open(bbslist_pkl, 'rb'))
        img_paths = dets_info[0]
        bbslist = dets_info[1]
    else:
        print('no bounding boxes pkls, exit')
        exit(0)

    model_file = 'faster_rcnn_pytorch/models/VGGnet_fast_rcnn_iter_70000.h5'
    extractor = build_extractor(model_file)
    #features_dets = []
    MAX_NO = 250
    
    pkl_dir = '../pkls/vot_features_{}_bbslist_{}'.format(dataset_name, classname)
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    for img_path, bbs in zip(img_paths, bbslist):
        dets = np.array([[0.] + list(bb[:-1]) for bb in bbs[:MAX_NO]], dtype='float32')
        DEBUG = True
        if DEBUG:
            img = cv2.imread(img_path)
            for det in dets:
                cv2.rectangle(img, (int(det[1]), int(det[2])), (int(det[3]), int(det[4])), (0, 204, 0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
        feature_dets = extractfeatures(img_path, extractor, dets)
        feature_dets = feature_dets.data.cpu().numpy()
        #features_dets.append(feature_dets)
        img_id = img_path.split('/')[-1].split('.')[0]
        dets_pkl = os.path.join(pkl_dir, 'vot_features_{}_bbslist_{}_{}.pkl'.format(dataset_name, classname, img_id)
        pkl.dump(feature_dets, open(dets_pkl, 'wb'))
