#!/usr/bin/env python
# coding: utf-8
import numpy as np

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
