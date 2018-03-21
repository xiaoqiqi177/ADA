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
from preprocess import get_dataset_info, get_dataset_info_oneclass
from losses import *
from nash_equilibrium import nash_equilibrium

tol = 1e-3

def init_theta():
    #can init with SVM?
    theta = np.random.normal(0, 0.1, theta_size)
    theta = theta / norm(theta)
    return theta

def expected_feature_difference(ps, Sps, fi_sets, average_feature):
    new_feature = np.zeros(average_feature.shape, dtype='float32')
    for p, Sp, fi_set in zip(ps, Sps, fi_sets): #lenp * 1024
        fi_set_p = fi_set[Sp]
        new_feature += np.sum(p.reshape(p.shape[0], 1) * fi_set_p, axis=0)
    new_feature /= len(ps)
    return new_feature - average_feature

def update_theta(theta, average_feature, Sps, ps, fi_sets, stepsize):
    gradient = expected_feature_difference(ps, Sps, fi_sets, average_feature)
    new_theta = theta - stepsize * gradient
    return new_theta

def test_convergence(theta, average_feature, Sps, ps, fi_sets):
    feature_difference = expected_feature_difference(ps, Sps, fi_sets, average_feature)
    delta = np.sum(feature_difference**2)
    return delta

def produce_batch_size_ids(ids, batch_size):
    pid = 0
    while True:
        if pid + batch_size <= total_no:
            yield ids[pid:pid+batch_size]
            pid += batch_size
        else:
            np.random.shuffle(ids)
            pid = 0
            continue

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def log_result(result):
    data.append(result)

def train_thread_bid(bid):
    img_path, gt, fi_gt, Y = img_paths[bid], gts[bid], features_gt[bid], detslist[bid]
    dets = [ det[:-1] for det in Y]
    dets = np.array(dets)
            
    #fi_set = extract_features(feature_type, theta_size, img_path, extractor, dets)
    fi_set = feautres_det[bid]

    psi_set = np.array([ sum(theta * (fi - fi_gt)) for fi in fi_set ])
    #print("start solving nash_equibrium for {} of bid {}...".format(img_path, bid))
    Sf, f, Sp, p = nash_equilibrium(img_path, theta, dets, psi_set)
    return f, p, Sf, Sp, fi_set

def load_info(dataset_name = 'trainval', target_classname):
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
                features_gt.append(f_gt / norm(f_gt))
                img_paths.append(os.path.abspath(os.path.join('../data/JPEGImages', img_path)))
                gts.append(bbox_gt)
                break
    return features_gt, img_paths, gts

if __name__ == '__main__':
    np.random.seed(1)

    stepsize = 0.03
    bb_number_threshold = 250
    iou_threshold = 0.5

    #load trainval dataset of certain class
    target_classname = 'person'
    dataset_name = 'trainval'
    features_gt, img_paths, gts = laod_info(dataset_name, target_classname)
     
    #processing average feature of training set
    if dataset_name == 'trainval':
        theta_size = 4096
        average_feature = np.zeros(theta_size)
        for feature_gt in features_gt:
            average_feature += feature_gt
        average_feature /= len(img_paths)

    #use edge_boxes for bounding box proposals
    bbslist_pkl = '../pkls/vot_{}_bbslist_{}.pkl'.format(dataset_name, target_classname)
    if os.path.exists(bbslist_pkl):
        bbslist = pkl.load(open(bbslist_pkl, 'rb'))
    else:
        bbslist = get_windows(img_paths_[dataset_name])
        pkl.dump([img_paths_[dataset_name], bbslist], open(bbslist_pkl, 'wb'))
    #top 250
    #detslist = [ bbs[:bb_number_threshold] for bbs in bbslist ]
    
    #extract bbs feature by outter extractor
    bbslist_feature_pkl = '../pkl/vot_features_{}_bbslist_{}.pkl'.format(dataset_name, target_classname)
    features_det = pkl.load(open(bbslist_feature_pkl, 'rb'))
    
    #init theta
    theta = init_theta()
    
    #start training
    print("Starting loops for optimizing theta...")
    convergence = False
    iter_number = 0
    
    total_no = len(gts)
    ids = np.random.permutation(total_no)
    batch_size = 32
    get_ids = produce_batch_size_ids(ids, batch_size) 
    saved_theta = []
    
    while not convergence:
        print("Loop {}...".format(iter_number))
        iter_number += 1
        fs, ps = [], []
        Sfs, Sps = [], []
        fi_sets = []
        batch_ids = next(get_ids)

        pool = Pool(4, init_worker)
        
        data = []
        try:
            pids = []
            for idx in range(batch_size):
                pids.append(pool.apply_async(train_thread_bid, (idx, ), callback = log_result))
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print("catch keyboard interrupt, prepare to stop")
            pool.terminate()
            pool.join()
            break
        fs = [ d[0] for d in data ]
        ps = [ d[1] for d in data ]
        Sfs = [ d[2] for d in data ]
        Sps = [ d[3] for d in data ]
        fi_sets = [ d[4] for d in data ]
        print('*---- update theta ----*')
        theta = update_theta(theta, average_feature, Sps, ps, np.array(fi_sets), stepsize)
        saved_theta.append(theta)
        
        #intermediate data periodically
        delta = test_convergence(theta, average_feature, Sps, ps, np.array(fi_sets))
        print('delta of Loop {}: {}'.format(iter_number, delta))
        if delta < tol:
            convergence = True
        print("Optimization completed!")
    
    import IPython
    IPython.embed()
    #save saved theta
    pkl.dump(saved_theta, open('saved_theta.pkl', 'wb'))
