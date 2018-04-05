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
    
        box = doc.createElement("bndbox")
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
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
healthy_persons = os.listdir(healthy_person_dir)

if len(sys.argv) == 1:
    ispart = None
    output_dir = './faster_rcnn_pytorch/data/Optha_MA_devkit/Optha_MA'
else:
    ispart = sys.argv[1]
    output_dir = './faster_rcnn_pytorch/data/Optha_MA_devkit/Optha_MA_part'

imageset_dir = os.path.join(output_dir, 'ImageSets')
f_main_healthy = open(os.path.join(imageset_dir, 'Main', 'healthy.txt'), 'w')
ma_main_healthy = open(os.path.join(imageset_dir, 'Main', 'ma_healthy.txt'), 'w')

DEBUG = True
saved_info = []
person_number = len(healthy_persons)
for person_id, person in enumerate(healthy_persons):
    task = 'healthy'
    ori_img_paths = os.listdir(os.path.join(healthy_person_dir, person))
    ori_img_paths = [ os.path.join(healthy_person_dir, person, ori_img_path) for ori_img_path in ori_img_paths ]
    for ori_img_path in ori_img_paths:
        assert os.path.isfile(ori_img_path)
        ori_img = cv2.imread(ori_img_path)
        if ispart is None:
            name_suffix = ori_img_path.split('/')[-1]
            pure_name_suffix = name_suffix.split('.')[0]
            des_file = pure_name_suffix+'.jpg'
            shutil.copy(ori_img_path, os.path.join(output_dir, 'JPEGImages', des_file))
        
            bboxes = []
            build_xml(os.path.join(output_dir, 'Annotations'), pure_name_suffix, bboxes, 'ma')
        
            pure_name_suffix += '\n'
            f_main_healthy.write(pure_name_suffix)
            ma_main_healthy.write(pure_name_suffix)
        else:
            h_total, w_total = ori_img.shape[:2]
            h_num = w_num = 10
            h_grid, w_grid = h_total // h_num, w_total // w_num
            for i in range(h_num):
                for j in range(w_num):
                    h_begin = i * h_grid
                    w_begin = j * w_grid
                    h_end = min(h_begin + h_grid, h_total)
                    w_end = min(w_begin + w_grid, w_total)
                    new_img = ori_img[h_begin:h_end, w_begin:w_end, :]

                    bboxes = []

                    name_suffix = ori_img_path.split('/')[-1]
                    pure_name_suffix = name_suffix.split('.')[0]+'_{}_{}'.format(i, j)
                    des_file = pure_name_suffix+'.jpg'
        
                    cv2.imwrite(os.path.join(output_dir, 'JPEGImages', des_file), new_img)
        
                    build_xml(os.path.join(output_dir, 'Annotations'), pure_name_suffix, bboxes, 'ma')
        
                    pure_name_suffix += '\n'
                    f_main_healthy.write(pure_name_suffix)
                    ma_main_healthy.write(pure_name_suffix)
