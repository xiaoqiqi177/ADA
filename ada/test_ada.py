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
from solve_lp_gurobi import solve_f, solve_p
from preprocess import get_dataset_info, get_dataset_info_oneclass
from losses import *
from nash_equilibrium import nash_equilibrium
import argparse

def expected_feature_difference(ps, Sps, fi_sets, average_feature):
    new_feature = np.zeros(average_feature.shape, dtype='float32')
    for p, Sp, fi_set in zip(ps, Sps, fi_sets): #lenp * 1024
        fi_set_p = fi_set[Sp]
        new_feature += np.sum(p.reshape(p.shape[0], 1) * fi_set_p, axis=0)
    new_feature /= len(ps)
    return new_feature - average_feature

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def test_thread_bid(bid, img_path, Y, fi_set):
    dets = [ det[:-1] for det in Y]
    dets = np.array(dets)
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


def test_accuracy(detslist_test, results, gts_test, iou_threshold):
    total_no = len(detslist_test)
    right_no = 0
    for dets, result, gt_list in zip(detslist_test, results, gts_test):
        for gt in gt_list:
            iou_loss = iou_loss1(dets[result], gt)
            if iou_loss < iou_threshold:
                right_no += 1
                break
    accuracy = right_no * 1. / total_no
    return right_no, total_no, accuracy

def load_info_test(dataset_name, target_classname):
    img_paths, bboxs_gts = get_dataset_info(dataset_name)
    gt_info = pkl.load(open('../pkls/vot_'+dataset_name+'_gt.pkl', 'rb'))
    classes_gt = gt_info[0]
    features_gt = gt_info[1]
    features_gt_use = []
    img_paths_use = []
    bboxs_gt_use = []
    for img_path, class_info, feature_gt, bboxs_gt in zip(img_paths, classes_gt, features_gt, bboxs_gts):
        exist_this_class = False
        features_gt_list = []
        bboxs_gt_list = []
        for classname, f_gt, bbox_gt in zip(class_info, feature_gt, bboxs_gt):
            bbox_gt = bbox_gt[1]
            if classname == target_classname:
                #normalize feature
                features_gt_list.append(f_gt / norm(f_gt))
                #features_gt_use.append(f_gt)
                bboxs_gt_list.append(bbox_gt)
                exist_this_class = True
        if exist_this_class is True:
            features_gt_use.append(bboxs_gt_list)
            img_paths_use.append(img_path)
            bboxs_gt_use.append(bboxs_gt_list)
    return features_gt_use, img_paths_use, bboxs_gt_use

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='trainval')
    parser.add_argument('--classname', type=str, default='car')
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    target_classname = args.classname
    iou_threshold = args.iou_threshold
    bb_number_threshold = 250


    #processing average feature of training set
    global DEBUG
    DEBUG = False
    np.random.seed(1)
    
    features_gt, img_paths, bboxs_gt = load_info_test(dataset_name, target_classname)

    theta_size = 4096
    
    
    bbslist_pkl = '../pkls/vot_{}_bbslist_{}.pkl'.format(dataset_name, target_classname)
    if os.path.exists(bbslist_pkl):
        _, bbslist = pkl.load(open(bbslist_pkl, 'rb'))
    else:
        print('exists not {}'.format(bbslist_pkl))
    
    #load theta
    if DEBUG:
        saved_theta = pkl.load(open('saved_theta_debug.pkl', 'rb'))
    else:
        saved_theta = pkl.load(open('saved_theta.pkl', 'rb'))
    theta = saved_theta[-1]
    
    #extract bbs feature by outter extractor
    pkl_dir = '../pkls/vot_features_{}_bbslist_{}'.format(dataset_name, target_classname)
    if not os.path.exists(pkl_dir):
        print('exists not {}'.format(pkl_dir))
        exit(0)
    bbslist_pkl = '../pkls/vot_{}_bbslist_{}.pkl'.format(dataset_name, target_classname)
    img_id_map = {}
    if os.path.exists(bbslist_pkl):
        bbs_imgpaths, bbslist = pkl.load(open(bbslist_pkl, 'rb'))
        for bbs_id, bbs_imgpaths in enumerate(bbs_imgpaths):
            img_id = bbs_imgpaths.split('/')[-1].split('.')[0]
            img_id_map[img_id] = bbs_id
    else:
        print('exists not {}'.format(bbslist_pkl))
        exit(0)
    
    #start testing
    print("Starting to test images using theta...")
    if DEBUG:
        total_no = 1
    else:
        total_no = len(img_paths)
    
    if DEBUG:
        pool = Pool(1, init_worker)
    else:
        pool = Pool(8, init_worker)
    global maxids
    maxids = [] 
    def maxid_log(result):
        maxids.append(result)
    
    test_pids = []
    dets_list = []
    for idx in range(total_no):
        img_path, gt_list, fi_gt = img_paths[idx], bboxs_gt[idx], features_gt[idx] 
        img_id = img_path.split('/')[-1].split('.')[0]
        dets_pkl = os.path.join(pkl_dir, 'vot_features_{}_bbslist_{}_{}.pkl'.format(dataset_name, target_classname, img_id)) 
        bbs_id = img_id_map[img_id]
        Y = bbslist[bbs_id]
        dets = [ det[:-1] for det in Y]
        dets = np.array(dets)
        dets_list.append(dets)
        with open(dets_pkl, 'rb') as fdets:
            fi_set = pkl.load(fdets)
        fi_set = [ fi / norm(fi) for fi in fi_set ]
        
        if len(fi_set) < bb_number_threshold:
            print('bb proposals number of {} is {} < {}'.format(img_path, len(fi_set), bb_number_threshold))
        
        test_pids.append(pool.apply_async(test_thread_bid, (idx, img_path, Y[:min(len(Y), bb_number_threshold)], fi_set[:min(len(fi_set), bb_number_threshold)]), callback = maxid_log))
    pool.close()
    pool.join()
    
    if DEBUG:
        for idx in range(total_no):
            img_path, gt_list = img_paths[idx], bboxs_gt[idx]
            img = cv2.imread(img_path)
            dets = dets_list[idx]
            det = dets[maxids[idx]]
            cv2.rectangle(img, (int(det[1]), int(det[0])), (int(det[3]), int(det[2])), (255, 0, 0), 1)
            for gt in gt_list:
                cv2.rectangle(img, (int(gt[1]), int(gt[0])), (int(gt[3]), int(gt[2])), (0, 255, 0), 1)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    
    right_no, total_no, accuracy = test_accuracy(dets_list, maxids, bboxs_gt[:total_no], iou_threshold)
    print('Test Accuracy: {} / {}, {}'.format(right_no, total_no, accuracy))

