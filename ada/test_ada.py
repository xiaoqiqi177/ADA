#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
from numpy.linalg import norm
import pickle as pkl
from multiprocessing import Pool
import signal
import sys
from edge_boxes_with_python.edge_boxes import get_windows
from solve_lp_gurobi import solve_f, solve_p
from preprocess import get_dataset_info, get_dataset_info_oneclass

tol = 1e-3

def expected_feature_difference(ps, Sps, fi_sets, average_feature):
    new_feature = np.zeros(average_feature.shape, dtype='float32')
    for p, Sp, fi_set in zip(ps, Sps, fi_sets): #lenp * 1024
        fi_set_p = fi_set[Sp]
        new_feature += np.sum(p.reshape(p.shape[0], 1) * fi_set_p, axis=0)
    new_feature /= len(ps)
    return new_feature - average_feature

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def thread_test_bid(bid, img_path, Y):
    img_path, Y = img_paths_test[bid], detslist_test[bid]
    dets = [ det[:-1] for det in Y]
    dets = np.array(dets)
    fi_set = extract_features(feature_type, theta_size, img_path, extractor, dets)
    psi_set = np.array([ sum(theta * fi) for fi in fi_set ])
    Sf, f, Sp, p = nash_equilibrium(img_path, theta, dets, psi_set)
    maxp = 0.
    maxid = -1
    for p, fid in zip(f, Sf):
        if p > maxp:
            maxp = p
            maxid = fid 
    print('finish testing image {}'.format(img_path))
    return maxid

def test_newimgs(img_paths_test, feature_type, extractor, detslist_test, theta):

    total_no = len(img_paths_test)
    pool_test = Pool(4, init_worker)
    maxids = [] 
    def maxid_log(result):
        maxids.append(result)
    test_pids = []
    for idx in range(total_no):
        img_path, Y = img_paths_test[idx], detslist_test[idx]
        test_pids.append(pool_test.apply_async(thread_test_bid, (idx, img_path, Y), callback=maxid_log))
    pool_test.close()
    pool_test.join()
    return maxids

def test_accuracy(detslist_test, results, gts_test, iou_threshold):
    total_no = len(detslist_test)
    right_no = 0
    for dets, result, gt in zip(detslist_test, results, gts_test):
        iou_loss = iou_loss1(dets[result][:-1], gt)
        if iou_loss < iou_threshold:
            right_no += 1
    accuracy = right_no * 1. / total_no
    return right_no, total_no, accuracy

if __name__ == '__main__':
    np.random.seed(1)

    bb_number_threshold = 250
    iou_threshold = 0.5

    target_classname = 'person'

    print("Preprocessing average feature of training set...")
        
    dataset_names = ['trainval', 'test']
    theta_size = 4096
    features_gt_ = {}
    img_paths_ = {}
    gts_ = {}

    average_feature = np.zeros(theta_size)
    for dataset_name in dataset_names:
        all_img_paths, all_gts = get_dataset_info(dataset_name)
        gt_info = pkl.load(open('../pkls/vot_'+dataset_name+'_gt.pkl', 'rb'))
        all_classes_gt = gt_info[0]
        all_features_gt = gt_info[1]
        features_gt = []
        img_paths = []
        gts = []
        for img_path, class_infos, feature_gt in zip(all_img_paths, all_classes_gt, all_features_gt):
            for class_info, f_gt in zip(class_infos, feature_gt):
                classname = class_info[0]
                bbox_gt = class_info[1]
                if classname == target_classname:
                    #normalize feature
                    features_gt.append(f_gt/norm(f_gt))
                    img_paths.append(os.path.abspath(os.path.join('../data/JPEGImages', img_path)))
                    gts.append(bbox_gt)
                    break
        if dataset_name == 'trainval':
            for feature_gt in features_gt:
                average_feature += feature_gt
            average_feature /= len(img_paths)
        features_gt_[dataset_name] = features_gt
        img_paths_[dataset_name] = img_paths
        gts_[dataset_name] = gts

    print("Using edge_boxes for bounding box proposals...")
    detslist_ = {}
    for dataset_name in dataset_names:
        bbslist_pkl = '../pkls/vot_{}_bbslist_{}.pkl'.format(dataset_name, target_classname)
        if os.path.exists(bbslist_pkl):
            bbslist = pkl.load(open(bbslist_pkl, 'rb'))
        else:
            bbslist = get_windows(img_paths_[dataset_name])
            pkl.dump([img_paths_[dataset_name], bbslist], open(bbslist_pkl, 'wb'))
        #top 250
        detslist = [ bbs[:bb_number_threshold] for bbs in bbslist ]
        detslist_[dataset_name] = detslist
    
    #extract bbs feature by outter extractor
    features_dets_ = {}
    for dataset_name in dataset_names:
        bbslist_feature_pkl = '../pkl/vot_features_{}_bbslist_{}.pkl'.format(dataset_name, target_classname)
        bbslist_features = pkl.load(open(bbslist_feature_pkl, 'rb'))
        features_dets_[dataset_name] = bbslist_features
    
    print("Starting to test images using theta...")
    predicted_results = test_newimgs(img_paths_test, feature_type, extractor, detslist_test, theta) 
    right_no, total_no, accuracy = test_accuracy(detslist_test, predicted_results, gts_test, iou_threshold)
    print('Test Accuracy: {} / {}, {}'.format(right_no, total_no, accuracy))

