#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
from numpy.linalg import norm
import pickle as pkl
from multiprocessing import Pool
import sys
#sys.path.append('faster_rcnn_pytorch')
#from extract_feature_from_bb import extractfeatures, build_extractor
from edge_boxes_with_python.edge_boxes import get_windows
from preprocess import get_dataset_info, get_dataset_info_oneclass

tol = 1e-3
step_size = 0.1

def extract_features(feature_type, theta_size, img_path, extractor, dets):
    if feature_type == 'fc7':
        return None
        #return extractfeatures(img_path, extractor, dets)
    elif feature_type == 'hist':
        img = cv2.imread(img_path)
        features = []
        for det in dets:
            try:
                cropped_img = img[int(det[0]):int(det[2]), int(det[1]):int(det[3]), :]
            except:
                import IPython
                IPython.embed()
            hist = cv2.calcHist([cropped_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist / norm(hist)
            features.append(hist.flatten())
        return features

def iou_loss1(bb1, bb2):
    w = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    h = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    overlap = w * h
    s1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    s2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou_loss1 = 1 - overlap / ( s1 + s2 - overlap)
    return iou_loss1

def iou_loss2(bb1, bb2, alpha):
    iou_loss1 = iou_loss1(bb1, bb2)
    iou_loss2 = int(iou_loss1 > alpha)
    return iou_loss2

def matrix_loss(dets, Sf, Sp):
    lenf = len(Sf)
    lenp = len(Sp)
    bbs1 = dets[Sf]
    bbs2 = dets[Sp]
    w_matrix = np.maximum(0, \
            np.minimum(np.tile(bbs1[:, 2], [lenp, 1]).transpose(), np.tile(bbs2[:, 2], [lenf, 1])) \
            - np.maximum(np.tile(bbs1[:, 0], [lenp, 1]).transpose(), np.tile(bbs2[:, 0], [lenf, 1])))
    h_matrix = np.maximum(0, \
            np.minimum(np.tile(bbs1[:, 3], [lenp, 1]).transpose(), np.tile(bbs2[:, 3], [lenf, 1])) \
            - np.maximum(np.tile(bbs1[:, 1], [lenp, 1]).transpose(), np.tile(bbs2[:, 1], [lenf, 1])))
    overlap_matrix = w_matrix * h_matrix
    s1_matrix = (bbs1[:,2] - bbs1[:, 0]) * (bbs1[:,3] - bbs1[:, 1])
    s2_matrix = (bbs2[:,2] - bbs2[:, 0]) * (bbs2[:,3] - bbs2[:, 1])
    iou_loss_matrix = 1 - overlap_matrix / (np.tile(s1_matrix, [lenp, 1]).transpose() + np.tile(s2_matrix, [lenf, 1]) - overlap_matrix)
    return iou_loss_matrix

if __name__ == '__main__':
    np.random.seed(1)

    feature_types = ['fc7', 'hist']
    feature_type = 'hist'
    assert feature_type in feature_types
    if feature_type == 'fc7':
        model_file = 'faster_rcnn_pytorch/models/VGGnet_fast_rcnn_iter_70000.h5'
        #extractor = build_extractor(model_file)
        extractor = None
        theta_size = 1024
    elif feature_type == 'hist':
        extractor = None
        theta_size = 512

    stepsize = 0.1

    #img_paths nd gts example
    #img_paths = ['faster_rcnn_pytorch/demo/004545.jpg']
    #img_paths = [ os.path.abspath(img_path) for img_path in img_paths ]
    #gts = np.asarray([[0., 261.32, 8.64, 354.23, 228.82]])
    target_classname = 'chair'
    print("Loading infomation of class {}...".format(target_classname))
    img_paths, gts = get_dataset_info_oneclass('trainval', 'chair') 
    #img_paths, gts = img_paths[:10], gts[:10]
    #simplified the problem to only one gt
    gts = [gt[0] for gt in gts]

    print("Using edge_boxes for bounding box proposals...")
    bbslist_pkl = 'vot_train_bbslist_{}.pkl'.format(target_classname)
    if os.path.exists(bbslist_pkl):
        bbslist = pkl.load(open(bbslist_pkl, 'rb'))
    else:
        bbslist = get_windows(img_paths)
        pkl.dump(bbslist, open(bbslist_pkl, 'wb'))
    #retain top250 bbs
    bb_number_threshold = 250
    detslist = [ bbs[:bb_number_threshold] for bbs in bbslist ]

    #choose from iou > 0.5
    iou_threshold = 0.5
    detslist_gt = []
    for dets, gt in zip(bbslist, gts):
        newdets = [ det for det in dets if iou_loss1(det[:-1], gt) < iou_threshold ]
        detslist_gt.append(newdets)
    #init theta
    theta = init_theta()
    
    print("Starting loops for optimizing theta...")
    convergence = False
    iter_number = 0
    
    total_no = len(gts)

    S = []
    lenS = 0
    while not convergence:
        print("Loop {}...".format(iter_number))
        iter_number += 1
        theta, xi = solve_qp(S)

        def thread_bid(bid):
            img_path, dets_gt, fi_gt, Y = img_paths[bid], gts[bid], detslist_gt[bid], detslist[bid]
            dets = np.array([ det[:-1] for det in Y])
            dets_gt = np.array([ det[:-1] for det in dets_gt])
            
            fi_set = extract_features(feature_type, theta_size, img_path, extractor, dets)
            fi_gt_set = extract_features(feature_type, theta_size, img_path, extractor, dets_gt)
            y_hat_id = -1
            min_value = -float("inf")
            
            for y_id in range(len(dets)):
                value = 0.
                for fi_gt, det_gt in zip(fi_gt_set, dets_gt):
                    value += iou_loss1(dets[y_id]) + theta * ( fi_set[y_id], fi_gt )
                value /= len(fi_gt)
                if value < min_value:
                    min_value = value
                    y_hat_id = y_id
            return y_hat_id

        pool = Pool(processes = 16)
        data = pool.map(thread_bid, [ i for i in range(total_no) ])
        pool.close()
        
        for i, y_hat_id in enumerate(data):
            if iou_loss1(dets[i][y_hat_id], gts[i]) > iou_threshold and (i, y_hat_id) not in S:
                S.append((i, y_hat_id))
        if len(S) == lenS:
            convergence = True
        else:
            lenS = len(S)

    print("Optimization completed!")
    import IPython
    IPythn.embed()
    print("Starting to test images using theta...")
