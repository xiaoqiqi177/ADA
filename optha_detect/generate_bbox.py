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

def build_xml(dir_name, pure_name_suffix, bboxes, _name):
    doc = minidom.Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    fout = open(os.path.join(dir_name, pure_name_suffix+'.xml'), 'w')
    for bbox in bboxes:
        obj = doc.createElement("object")
    
        name = doc.createElement("name")
        textname = doc.createTextNode(_name)
        name.appendChild(textname)
        obj.appendChild(name)
    
        box = doc.createElement("nbdbox")
        obj.appendChild(box)
    
        xmin = doc.createElement("xmin")
        xmax = doc.createElement("xmax")
        ymin = doc.createElement("ymin")
        ymax = doc.createElement("ymax")
    
        textxmin = doc.createTextNode(str(bbox[0]))
        xmin.appendChild(textxmin)
        textxmax = doc.createTextNode(str(bbox[2]))
        xmax.appendChild(textxmax)
        textymin = doc.createTextNode(str(bbox[1]))
        ymin.appendChild(textymin)
        textymax = doc.createTextNode(str(bbox[3]))
        ymax.appendChild(textymax)

        box.appendChild(xmin)
        box.appendChild(xmax)
        box.appendChild(ymin)
        box.appendChild(ymax)
    
        annotation.appendChild(obj)

    doc.writexml(fout,"\t", "\t", "\n")
    fout.close()

dataset_dir = '../datasets/e_optha_MA/'
MA_person_dir = os.path.join(dataset_dir, 'MA/')
MA_annotation_dir = os.path.join(dataset_dir, 'Annotation_MA/')
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
MA_persons = os.listdir(MA_person_dir)

output_dir = './faster_rcnn_pytorch/data/Optha_MA_devkit/Optha_MA'
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

imageset_dir = os.path.join(output_dir, 'ImageSets')
f_main_test = open(os.path.join(imageset_dir, 'Main', 'test.txt'), 'w')
ma_main_test = open(os.path.join(imageset_dir, 'Main', 'ma_test.txt'), 'w')
f_main_train = open(os.path.join(imageset_dir, 'Main', 'train.txt'), 'w')
ma_main_train = open(os.path.join(imageset_dir, 'Main', 'ma_train.txt'), 'w')
f_main_val = open(os.path.join(imageset_dir, 'Main', 'val.txt'), 'w')
ma_main_val = open(os.path.join(imageset_dir, 'Main', 'ma_val.txt'), 'w')
f_main_trainval = open(os.path.join(imageset_dir, 'Main', 'trainval.txt'), 'w')
ma_main_trainval = open(os.path.join(imageset_dir, 'Main', 'ma_trainval.txt'), 'w')

DEBUG = False
saved_info = []
person_number = len(MA_persons)
for person_id, person in enumerate(MA_persons):
    if person_id < person_number * train_ratio:
        task = 'train'
    elif person_id < person_number * (train_ratio + val_ratio):
        task = 'val'
    else:
        task = 'test'
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
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if DEBUG:
                cv2.drawContours(annotation, [contour], 0, (255, 0, 0), 2)
                cv2.rectangle(annotation, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(ori_img, str(i), (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 2., (0, 0, 255), thickness=1)
            bboxes.append([x, y, x+w, y+h])
        if DEBUG:
            show_img = np.concatenate((ori_img, cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)), axis=0)
            cv2.imwrite('img_temp.png', show_img)
            import IPython
            IPython.embed()
        #save info to pickle bboxes
        #saved_info.append([ori_img_path, bboxes])
        name_suffix = ori_img_path.split('/')[-1]
        shutil.copy(ori_img_path, os.path.join(output_dir, 'JPEGImages', name_suffix))
        
        pure_name_suffix = name_suffix.split('.')[0]
        build_xml(os.path.join(output_dir, 'Annotations'), pure_name_suffix, bboxes, 'ma')
        
        pure_name_suffix += '\n'
        if task == 'test':
            f_main_test.write(pure_name_suffix)
            ma_main_test.write(pure_name_suffix)
        else:
            if task == 'train':
                f_main_train.write(pure_name_suffix)
                ma_main_train.write(pure_name_suffix)
            else:
                f_main_val.write(pure_name_suffix)
                ma_main_val.write(pure_name_suffix)
            f_main_trainval.write(pure_name_suffix)
            ma_main_trainval.write(pure_name_suffix)
        
