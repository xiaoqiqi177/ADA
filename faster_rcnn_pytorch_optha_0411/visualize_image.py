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
                if c_number == 0:
                    continue
                new_img_g = new_img[:, :, 1]
                new_img_g = np.dstack((new_img_g, new_img_g, new_img_g))
                 
                edges = cv2.Canny(new_img_g, 30, 40)
                edges = np.dstack((edges, edges, edges))
                for c_id, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    x, y = x+1, y+1
                    center_x = x + w/2
                    center_y = y + h/2
                    #w *= 3
                    #h *= 3
                    x = int(max(center_x - w/2, 1))
                    y = int(max(center_y - h/2, 1))
                    w = int(min(w, new_img.shape[1]-x))
                    h = int(min(h, new_img.shape[0]-y))
                    #cv2.drawContours(new_annotation, [contour], 0, (255, 0, 0), 1)
                    cv2.rectangle(new_annotation, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    #cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                show_img = np.concatenate((new_img, new_img_g, edges, new_annotation), axis=0)
                cv2.imshow('show_img', show_img)
                cv2.waitKey(0)
