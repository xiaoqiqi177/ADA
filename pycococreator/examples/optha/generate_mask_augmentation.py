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

def build_xml(dir_name, pure_name_suffix, bboxes, _name, filename, fake_bbox):
    doc = minidom.Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    fout = open(os.path.join(dir_name, pure_name_suffix+'.xml'), 'w')
    filenameobj = doc.createElement('filename')
    textfilename = doc.createTextNode(filename)
    filenameobj.appendChild(textfilename)
    annotation.appendChild(filenameobj)
    bbox_no = len(bboxes)
    for bid, bbox in enumerate(bboxes):
        obj = doc.createElement("object")
    
        name = doc.createElement("name")
        if bid == bbox_no:
            textname = doc.createTextNode('fake_bg')
        else:
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

dataset_dir = '../../../data/e_optha_MA/'
MA_person_dir = os.path.join(dataset_dir, 'MA/')
MA_annotation_dir = os.path.join(dataset_dir, 'Annotation_MA/')
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
MA_persons = os.listdir(MA_person_dir)

parser = argparse.ArgumentParser(description='generate ma data')
parser.add_argument('--task', default='ma', type=str, help='will be used for the output')
parser.add_argument('--ifskip', default=False, action='store_true', help='if skip patch/image without bounding boxes')
parser.add_argument('--ifimage', default=False, action='store_true', help='if need generate images and xmls')
parser.add_argument('--ratio-name', default='325', type=str)
parser.add_argument('--aug-steps', default=16, type=int)
parser.add_argument('--ifdebug', default=False, action='store_true', help='if debug')

args = parser.parse_args()
output_dir = './train'

#generate output_dir
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'annotations'))
    os.mkdir(os.path.join(output_dir, 'optha_train2018'))
    os.mkdir(os.path.join(output_dir, 'optha_val2018'))
    os.mkdir(os.path.join(output_dir, 'optha_test2018'))

person_number = len(MA_persons)
train_ratio, test_ratio = int(args.ratio_name[0])*0.1, int(args.ratio_name[-1])*0.1
for person_id, person in enumerate(MA_persons):
    if person_id < person_number * train_ratio:
        datasetname = 'train'
    elif person_id < person_number * (1 - test_ratio):
        datasetname = 'val'
    else:
        datasetname = 'test'
    ori_img_paths = os.listdir(os.path.join(MA_person_dir, person))
    ori_img_paths = [ os.path.join(MA_person_dir, person, ori_img_path) for ori_img_path in ori_img_paths ]
    annotation_paths = os.listdir(os.path.join(MA_annotation_dir, person))
    annotation_paths = [ os.path.join(MA_annotation_dir, person, annotation_path) for annotation_path in annotation_paths ]
    for ori_img_path, annotation_path in zip(ori_img_paths, annotation_paths):
        ori_img = cv2.imread(ori_img_path)
        annotation = cv2.imread(annotation_path)
        
        h_total, w_total = ori_img.shape[:2]
        
        h_num = w_num = 10
        h_grid, w_grid = h_total // h_num, w_total // w_num
            
        if datasetname == 'test':
            aug_steps = 1
        else:
            aug_steps = args.aug_steps
        sqrt_step = int(np.sqrt(aug_steps))
        for step in range(aug_steps):
            h_step = step // sqrt_step
            w_step = step % sqrt_step
            #h_bias = np.random.randint(h_grid)
            #w_bias = np.random.randint(w_grid)
            h_bias = (h_step * h_grid) // sqrt_step
            w_bias = (w_step * w_grid) // sqrt_step
            for i in range(h_num-1):
                for j in range(w_num-1):
                    h_begin = i * h_grid + h_bias
                    w_begin = j * w_grid + h_bias
                    h_end = min(h_begin + h_grid, h_total)
                    w_end = min(w_begin + w_grid, w_total)
                    new_img = ori_img[h_begin:h_end, w_begin:w_end, :]
                    new_annotation = annotation[h_begin:h_end, w_begin:w_end, :]
                    #remove edge images
                    if new_img[:,:,1].mean() < 10:
                        continue

                    name_suffix = ori_img_path.split('/')[-1]
                    pure_name_suffix = name_suffix.split('.')[0]+'-{}-{}-{}'.format(step, i, j)
                    des_file = pure_name_suffix+'.jpeg'
        
                    annotation_file = pure_name_suffix+'_ma_0.png'
                    
                    if new_annotation.sum() == 0:
                        continue

                    if (new_annotation[:, 0, 0] > 0).any() or (new_annotation[:, -1, 0] > 0).any() or (new_annotation[0, :, 0] > 0).any() or (new_annotation[-1,:,0]>0).any():
                        continue
                    
                    if args.ifimage:
                        cv2.imwrite(os.path.join(output_dir, 'annotations', annotation_file), new_annotation)
                        cv2.imwrite(os.path.join(output_dir, 'optha_'+datasetname+'2018', des_file), new_img)
