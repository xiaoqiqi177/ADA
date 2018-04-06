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
from preprocess import get_dataset_info, get_dataset_info_oneclass
from losses import *
from nash_equilibrium import nash_equilibrium
import argparse

def init_theta():
    #can init with SVM?
    theta = np.random.normal(0, 0.1, theta_size)
    theta = theta / norm(theta)
    return theta

def expected_feature_difference(ps, Sps, fi_sets, average_feature):
    new_feature = np.zeros(average_feature.shape, dtype='float32')
    for p, Sp, fi_set in zip(ps, Sps, fi_sets): #lenp * 1024
        try:
            fi_set_p = fi_set[Sp]
        except:
            print('error in expected_feature_difference')
            import IPython
            IPython.embed()
        new_feature += np.sum(p.reshape(p.shape[0], 1) * fi_set_p, axis=0)
    new_feature /= len(ps)
    return new_feature - average_feature

def update_theta(theta, average_feature, Sps, ps, fi_sets, stepsize):
    try:
        difference = expected_feature_difference(ps, Sps, fi_sets, average_feature)
        gradient = difference
    except:
        print('error in update_theta')
        import IPython
        IPython.embed()
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


def train_thread_bid(bid, img_path, gt, fi_gt, Y, fi_set):
    dets = [ det[:-1] for det in Y]
    dets = np.array(dets)
    if DEBUG:
        feature_differences = np.array([ sum( (fi - fi_gt)**2 ) for fi in fi_set ])
        nearest_id = np.argmin(feature_differences)
        img = cv2.imread(img_path)
        det = dets[nearest_id]
        cv2.rectangle(img, (int(det[1]), int(det[0])), (int(det[3]), int(det[2])), (255, 0, 0), 1)
        cv2.rectangle(img, (int(gt[1]), int(gt[0])), (int(gt[3]), int(gt[2])), (0, 255, 0), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    psi_set = np.array([ sum(theta * (fi - fi_gt)) for fi in fi_set ])
    Sf, f, Sp, p = nash_equilibrium(img_path, theta, dets, psi_set)

def load_info_train(dataset_name, target_classname):
    img_paths, bboxs_gts = get_dataset_info(dataset_name)
    gt_info = pkl.load(open('../pkls/vot_'+dataset_name+'_gt.pkl', 'rb'))
    classes_gt = gt_info[0]
    features_gt = gt_info[1]
    features_gt_use = []
    img_paths_use = []
    bboxs_gt_use = []
    for img_path, class_info, feature_gt, bboxs_gt in zip(img_paths, classes_gt, features_gt, bboxs_gts):
        for classname, f_gt, bbox_gt in zip(class_info, feature_gt, bboxs_gt):
            bbox_gt = bbox_gt[1]
            if classname == target_classname:
                #normalize feature
                features_gt_use.append(f_gt / norm(f_gt))
                #features_gt_use.append(f_gt)
                img_paths_use.append(img_path)
                bboxs_gt_use.append(bbox_gt)
                #only store the first item for one image
                break
    return features_gt_use, img_paths_use, bboxs_gt_use

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', type=str, default='trainval')
    parser.add_argument('--classname', type=str, default='car')
    parser.add_argument('--tol', type=float, default=1e-3)
    parser.add_argument('--stepsize', type=float, default=0.1)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    args = parser.parse_args()
    dataset_name = args.dataset_name
    target_classname = args.classname
    tol = args.tol
    stepsize = args.stepsize
    iou_threshold = args.iou_threshold
    bb_number_threshold = 250
    global DEBUG
    DEBUG = False
    np.random.seed(1)


    features_gt, img_paths, bboxs_gt = load_info_train(dataset_name, target_classname)

    #processing average feature of training set
    if dataset_name == 'trainval':
        theta_size = 4096
        average_feature = np.zeros(theta_size)
        for feature_gt in features_gt:
            average_feature += feature_gt
        average_feature /= len(img_paths)

    #use edge_boxes for bounding box proposals
    bbslist_pkl = '../pkls/vot_{}_bbslist.pkl'.format(dataset_name)
    if os.path.exists(bbslist_pkl):
        _, bbslist = pkl.load(open(bbslist_pkl, 'rb'))
    else:
        print('exists not {}'.format(bbslist_pkl))
    
    #init theta
    global theta
    theta = init_theta()
    saved_theta = [ theta ]
    
    #start training
    print("Starting loops for optimizing theta...")
    convergence = False
    iter_number = 0
    total_no = len(features_gt)
    ids = np.random.permutation(total_no)
    if DEBUG:
        batch_size = 1
    else:
        batch_size = 32
    get_ids = produce_batch_size_ids(ids, batch_size) 
    
    #extract bbs feature by outter extractor
    pkl_dir = '../pkls/vot_features_{}_bbslist'.format(dataset_name)
    if not os.path.exists(pkl_dir):
        print('exists not {}'.format(pkl_dir))
        exit(0)
    bbslist_pkl = '../pkls/vot_{}_bbslist.pkl'.format(dataset_name)
    img_id_map = {}
    if os.path.exists(bbslist_pkl):
        bbs_imgpaths, bbslist = pkl.load(open(bbslist_pkl, 'rb'))
        for bbs_id, bbs_imgpaths in enumerate(bbs_imgpaths):
            img_id = bbs_imgpaths.split('/')[-1].split('.')[0]
            img_id_map[img_id] = bbs_id
    else:
        print('exists not {}'.format(bbslist_pkl))
        exit(0)
    
    while not convergence:
        print("Loop {}...".format(iter_number))
        iter_number += 1
        fs, ps = [], []
        Sfs, Sps = [], []
        fi_sets = []
        batch_ids = next(get_ids)
        if DEBUG:
            pool = Pool(1, init_worker)
        else:
            pool = Pool(8, init_worker)
        global data
        data = []
        def log_result(result):
            data.append(result)
        try:
            pids = []
            for idx in batch_ids:
                #gt[y1, x1, y2, x2], bbs[y1, x2, y2, x2, score]
                img_path, gt, fi_gt = img_paths[idx], bboxs_gt[idx], features_gt[idx] 
                img_id = img_path.split('/')[-1].split('.')[0]
                dets_pkl = os.path.join(pkl_dir, 'vot_features_{}_bbslist_{}.pkl'.format(dataset_name, img_id)) 
                bbs_id = img_id_map[img_id]
                Y = bbslist[bbs_id]
                with open(dets_pkl, 'rb') as fdets:
                    fi_set = pkl.load(fdets)
                fi_set = [ fi / norm(fi) for fi in fi_set ]
                if len(fi_set) < bb_number_threshold:
                    print(len(fi_set))
                    continue
                fi_sets.append(np.array(fi_set[:bb_number_threshold]))
                pids.append(pool.apply_async(train_thread_bid, (idx, img_path, gt, fi_gt, Y[:bb_number_threshold], fi_set[:bb_number_threshold]), callback = log_result))
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
        print('*---- update theta ----*')
        theta = update_theta(theta, average_feature, Sps, ps, np.array(fi_sets), stepsize)
        saved_theta.append(theta)
        #intermediate data periodically
        delta = test_convergence(theta, average_feature, Sps, ps, np.array(fi_sets))
        print('delta of Loop {}: {}'.format(iter_number, delta))
        if delta < tol:
            convergence = True
        print("Optimization completed!")
    
    #save saved theta
    pkl.dump(saved_theta, open('saved_theta_{}.pkl'.format(target_classname), 'wb'))
