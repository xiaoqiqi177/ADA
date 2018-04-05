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
MA_person_dir = os.path.join(dataset_dir, 'MA/')
MA_annotation_dir = os.path.join(dataset_dir, 'Annotation_MA/')
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
MA_persons = os.listdir(MA_person_dir)

if len(sys.argv) == 1:
    ispart = None
    output_dir = './faster_rcnn_pytorch/data/Optha_MA_devkit/Optha_MA'
else:
    ispart = sys.argv[1]
    output_dir = './faster_rcnn_pytorch/data/Optha_MA_devkit/Optha_MA_part'
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

DEBUG = True
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
        
        if ispart is None:
            ret, thresh = cv2.threshold(annotation, 127, 255, 0)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bboxes = []
            for c_id, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                x, y = x+1, y+1
                if DEBUG:
                    cv2.drawContours(annotation, [contour], 0, (255, 0, 0), 2)
                    cv2.rectangle(annotation, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(ori_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(ori_img, str(c_id), (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 2., (0, 0, 255), thickness=1)
                bboxes.append([x, y, x+w, y+h])
            if DEBUG:
                show_img = np.concatenate((ori_img, cv2.cvtColor(annotation, cv2.COLOR_GRAY2BGR)), axis=0)
                #cv2.imwrite('img_temp.png', show_img)

            name_suffix = ori_img_path.split('/')[-1]
            pure_name_suffix = name_suffix.split('.')[0]
            des_file = pure_name_suffix+'.jpg'
            shutil.copy(ori_img_path, os.path.join(output_dir, 'JPEGImages', des_file))
        
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

                    new_annotation = annotation[h_begin:h_end, w_begin:w_end]        
                    ret, thresh = cv2.threshold(new_annotation, 127, 255, 0)
                    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    bboxes = []

                    c_number = len(contours)
                    #skip empty image
                    #if c_number == 0:
                    #    continue
                    for c_id, contour in enumerate(contours):
                        x, y, w, h = cv2.boundingRect(contour)
                        x, y = x+1, y+1
                        if DEBUG:
                            center_x = x + w/2
                            center_y = y + h/2
                            w *= 3
                            h *= 3
                            x = int(max(center_x - w/2, 1))
                            y = int(max(center_y - h/2, 1))
                            w = int(min(w, new_img.shape[1]-x))
                            h = int(min(h, new_img.shape[0]-y))
                            cv2.drawContours(new_annotation, [contour], 0, (255, 0, 0), 1)
                            cv2.rectangle(new_annotation, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                            #cv2.putText(new_img, str(c_id), (x, y + 15), cv2.FONT_HERSHEY_PLAIN, 1., (0, 0, 255), thickness=1)
                        bboxes.append([x, y, x+w, y+h])
                    if False:
                        show_img = np.concatenate((new_img, cv2.cvtColor(new_annotation, cv2.COLOR_GRAY2BGR)), axis=0)
                        cv2.imshow('show_img', show_img)
                        #cv2.imshow('new_img', new_img)
                        cv2.waitKey(0)
                        #cv2.imshow('new_annotation', new_annotation)
                        #cv2.waitKey(0)
                    
                    name_suffix = ori_img_path.split('/')[-1]
                    pure_name_suffix = name_suffix.split('.')[0]+'_{}_{}'.format(i, j)
                    des_file = pure_name_suffix+'.jpg'
        
                    cv2.imwrite(os.path.join(output_dir, 'JPEGImages', des_file), new_img)
        
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
