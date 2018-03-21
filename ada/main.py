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
step_size = 0.1

'''
    #hist features
    if feature_type == 'hist':
        img = cv2.imread(img_path)
        features = []
        for det in dets:
            cropped_img = img[int(det[0]):int(det[2]), int(det[1]):int(det[3]), :]
            hist = cv2.calcHist([cropped_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist / norm(hist)
            features.append(hist.flatten())
        return features
'''

def iou_loss1(bb1, bb2):
    w = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0]))
    h = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    overlap = w * h
    s1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    s2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou_loss1 = 1 - overlap / ( s1 + s2 - overlap)
    return iou_loss1

def test_iou_loss1():
    bb1 = np.array([0, 0, 2, 2])
    bb2 = np.array([0, 0, 2, 3])
    iou_loss = iou_loss1(bb1, bb2)
    print(iou_loss)

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

def test_matrix_loss():
    bb1 = np.array([0, 0, 2, 2])
    bb2 = np.array([0, 0, 2, 3])
    dets = np.concatenate((np.tile(bb1, [5, 1]), np.tile(bb2, (3, 1))), axis=0)
    iou_loss_matrix = matrix_loss(dets, np.array([0]), np.array([6]))
    print(iou_loss_matrix)

def solveGame(dets, Sf, Sp, psi_set):
    G_loss = matrix_loss(dets, Sf, Sp)
    G_constraint = np.tile(psi_set[Sp], [len(Sf), 1])
    G = G_loss + G_constraint
    f, v1 = solve_f(G)
    p, v2 = solve_p(G)
    assert abs(v1 - v2) < tol
    return f, p, v1
    
def nash_equilibrium(imgpath, theta, dets, psi_set):
    Sp = []
    Sf = []
    #arbitrary pick at first, argmax according to the original paper
    first_y_id = np.argmax(psi_set)
    Sp.append(first_y_id)
    Sf.append(first_y_id)
    lenSp = len(Sp)
    lenSf = len(Sf)
    while True:
        f, p, vp = solveGame(dets, np.array(Sf), np.array(Sp), psi_set) 
        #find y_new
        max_expected_loss = 0.
        max_y_id = -1
        for new_y_id in range(len(psi_set)):
            expected_loss = 0.
            for i, f_id in enumerate(Sf):
                expected_loss += f[i] * (iou_loss1(dets[f_id], dets[new_y_id]) + psi_set[new_y_id])
            if expected_loss > max_expected_loss:
                max_expected_loss = expected_loss
                max_y_id = new_y_id
        vmax = max_expected_loss 
        #if abs(vp - vmax) > tol:
        if max_y_id not in Sp:
            Sp.append(max_y_id)
        f, p, vf = solveGame(dets, np.array(Sf), np.array(Sp), psi_set)
        
        #find y_prime_new
        min_expected_loss = float("inf")
        min_y_prime_id = -1
        for new_y_prime_id in range(len(psi_set)):
            expected_loss = 0.
            for i, p_id in enumerate(Sp):
                expected_loss += p[i] * iou_loss1(dets[p_id], dets[new_y_prime_id])
            if expected_loss < min_expected_loss:
                min_expected_loss = expected_loss
                min_y_prime_id = new_y_prime_id
        vmin = min_expected_loss
        #if abs(vf - vmin) > tol:
        if min_y_prime_id not in Sf:
            Sf.append(min_y_prime_id)
        #print("vp: {} vf: {} vmin: {} vmax: {}".format(vp, vf, vmin, vmax))
        #print("Sp: {} Sf: {}".format(len(Sp), len(Sf)))
        if lenSp == len(Sp) and lenSf == len(Sf):
        #if abs(vp - vmax) < tol and abs(vmax - vf) < tol and abs(vf - vmin) < tol:
            break
        else:
            lenSp, lenSf = len(Sp), len(Sf)
    return Sf, f, Sp, p

def init_theta():
    #theta = np.random.rand(theta_size)
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
    try:
        test_pids = []
        for idx in range(total_no):
            img_path, Y = img_paths_test[idx], detslist_test[idx]
            test_pids.append(pool_test.apply_async(thread_test_bid, (idx, img_path, Y), callback=maxid_log))
        pool_test.close()
        pool_test.join()
    except KeyboardInterrupt:
        import IPython
        IPython.embed()
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

    stepsize = 0.03
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
    
    import IPython
    IPython.embed()
    
    #init theta
    theta = init_theta()
    
    gts = gts_['trainval']
    img_paths = img_paths_['trainval']
    features_gt = features_gt_['trainval']
    detslist = detslist_['trainval']
    features_det = features_dets['trainval']

    print("Starting loops for optimizing theta...")
    convergence = False
    iter_number = 0
    
    batch_size = 8
    total_no = len(gts)
    ids = np.random.permutation(total_no)
    
    get_ids = produce_batch_size_ids(ids, batch_size)
     
    def thread_bid(bid):
        img_path, gt, fi_gt, Y = img_paths[bid], gts[bid], features_gt[bid], detslist[bid]
        dets = [ det[:-1] for det in Y]
        dets = np.array(dets)
            
        #fi_set = extract_features(feature_type, theta_size, img_path, extractor, dets)
        fi_set = feautres_det[bid]

        psi_set = np.array([ sum(theta * (fi - fi_gt)) for fi in fi_set ])
        #print("start solving nash_equibrium for {} of bid {}...".format(img_path, bid))
        Sf, f, Sp, p = nash_equilibrium(img_path, theta, dets, psi_set)
        return f, p, Sf, Sp, fi_set
	
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
        def log_result(result):
            data.append(result)
        try:
            pids = []
            for idx in range(batch_size):
                pids.append(pool.apply_async(thread_bid, (idx, ), callback = log_result))
            #produce_batch_size_idsa = pool.map(thread_bid, [ i for i in range(batch_size) ])
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
    print("Starting to test images using theta...")
    predicted_results = test_newimgs(img_paths_test, feature_type, extractor, detslist_test, theta) 
    right_no, total_no, accuracy = test_accuracy(detslist_test, predicted_results, gts_test, iou_threshold)
    print('Test Accuracy: {} / {}, {}'.format(right_no, total_no, accuracy))

