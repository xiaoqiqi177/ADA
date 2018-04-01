#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
import pickle as pkl
import sys
import glob

dataset_dir = '../datasets/e_optha_MA/'
MA_person_dir = os.path.join(dataset_dir, 'MA/')
MA_annotation_dir = os.path.join(dataset_dir, 'Annotation_MA/')
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')

MA_persons = os.listdir(MA_person_dir)

DEBUG = False
saved_info = []
for person in MA_persons:
    ori_img_paths = os.listdir(os.path.join(MA_person_dir, person))
    ori_img_paths = [ os.path.join(MA_person_dir, person, ori_img_path) for ori_img_path in ori_img_paths ]
    annotation_paths = os.listdir(os.path.join(MA_annotation_dir, person))
    annotation_paths = [ os.path.join(MA_annotation_dir, person, annotation_path) for annotation_path in annotation_paths ]
    for ori_img_path, annotation_path in zip(ori_img_paths, annotation_paths):
        ori_img = cv2.imread(ori_img_path)
        annotation = cv2.imread(annotation_path, 0)
        ret, thresh = cv2.threshold(annotation, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if DEBUG:
                cv2.drawContours(annotation, [contour], 0, (255, 0, 0), 2)
                cv2.rectangle(annotation, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                show_img = np.concatenate((ori_img, cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)), axis=1)
                cv2.imshow('img', show_img)
                cv2.waitKey(0)
            bboxes.append([x, y, x+w, y+h])
        #save info to pickle bboxes
        saved_info.append([ori_img_path, bboxes])

pkl.dump(saved_info, open('ma_info.pkl', 'wb'))
