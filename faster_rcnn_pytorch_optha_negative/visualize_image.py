#!/usr/bin/env python
# coding: utf-8

import cv2
import os
import numpy as np
import pickle as pkl
import sys
import glob
import shutil
import xml.dom.minidom as minidom
import argparse
from matplotlib import pyplot as plt

dataset_dir = '../datasets/e_optha_MA/'
MA_person_dir = os.path.join(dataset_dir, 'MA/')
MA_annotation_dir = os.path.join(dataset_dir, 'Annotation_MA/')
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
MA_persons = os.listdir(MA_person_dir)

person_number = len(MA_persons)
for person_id, person in enumerate(MA_persons):
    ori_img_paths = os.listdir(os.path.join(MA_person_dir, person))
    ori_img_paths = [ os.path.join(MA_person_dir, person, ori_img_path) for ori_img_path in ori_img_paths ]
    annotation_paths = os.listdir(os.path.join(MA_annotation_dir, person))
    annotation_paths = [ os.path.join(MA_annotation_dir, person, annotation_path) for annotation_path in annotation_paths ]
    for ori_img_path, annotation_path in zip(ori_img_paths, annotation_paths):
        ori_img = cv2.imread(ori_img_path)
        annotation = cv2.imread(annotation_path)
        h_total, w_total = ori_img.shape[:2]
        h_num = w_num = 20
        h_grid, w_grid = h_total // h_num, w_total // w_num
        for i in range(h_num):
            for j in range(w_num):
                h_begin = i * h_grid
                w_begin = j * w_grid
                h_end = min(h_begin + h_grid, h_total)
                w_end = min(w_begin + w_grid, w_total)
                new_img = ori_img[h_begin:h_end, w_begin:w_end, :]

                new_annotation = annotation[h_begin:h_end, w_begin:w_end, :]        
                ret, thresh = cv2.threshold(new_annotation[:,:,0], 127, 255, 0)
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                bboxes = []

                c_number = len(contours)
                
                new_img_b = new_img[:, :, 0]
                new_img_g = new_img[:, :, 1]
                new_img_r = new_img[:, :, 2]
                
                for c_id, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    x, y = x+1, y+1
                    center_x = x + w/2
                    center_y = y + h/2
                    x = int(max(center_x - w/2, 1))
                    y = int(max(center_y - h/2, 1))
                    w = int(min(w, new_img.shape[1]-x))
                    h = int(min(h, new_img.shape[0]-y))
                    cv2.rectangle(new_annotation, (x, y), (x + w, y + h), (0, 255, 0), 1)
                
                if len(contours) <= 0:
                    continue
                equ = cv2.equalizeHist(new_img_g)
                
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl1 = clahe.apply(new_img_g) 
                
                new_img_b = np.dstack((new_img_b, new_img_b, new_img_b))
                new_img_g = np.dstack((new_img_g, new_img_g, new_img_g))
                new_img_r = np.dstack((new_img_r, new_img_r, new_img_r))
                equ = np.dstack((equ, equ, equ))
                cl1 = np.dstack((cl1, cl1, cl1))
 
                show_img = np.concatenate((new_img, new_img_b, new_img_g, new_img_r, equ, cl1, new_annotation), axis=0)
                cv2.imshow('show_img', show_img)
                cv2.waitKey(0)
