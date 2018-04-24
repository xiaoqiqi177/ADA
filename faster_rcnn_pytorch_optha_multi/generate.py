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

def build_xml(dir_name, pure_name_suffix, bboxes, _name, filename, fake_bbox, iffake):
    doc = minidom.Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)
    fout = open(os.path.join(dir_name, pure_name_suffix+'.xml'), 'w')
    filenameobj = doc.createElement('filename')
    textfilename = doc.createTextNode(filename)
    filenameobj.appendChild(textfilename)
    annotation.appendChild(filenameobj)
    bbox_no = len(bboxes)
    if iffake:
        all_bboxes = bboxes + [fake_bbox]
    else:
        all_bboxes = bboxes
    for bid, bbox in enumerate(all_bboxes):
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

dataset_dir = '../datasets/e_optha_MA/'
MA_person_dir = os.path.join(dataset_dir, 'MA/')
MA_annotation_dir = os.path.join(dataset_dir, 'Annotation_MA/')
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
MA_persons = os.listdir(MA_person_dir)
healthy_person_dir = os.path.join(dataset_dir, 'healthy/')
healthy_persons = os.listdir(healthy_person_dir)

parser = argparse.ArgumentParser(description='generate ma data')
parser.add_argument('--task', default='ma', type=str, help='will be used for the output')
parser.add_argument('--ifskip', default=False, action='store_true', help='if skip patch/image without bounding boxes')
parser.add_argument('--iffake', default=False, action='store_true', help='if ad fake_bg class')
parser.add_argument('--ifimage', default=False, action='store_true', help='if need generate images and xmls')
parser.add_argument('--ifaug', default=False, action='store_true', help='if need augmentation')
parser.add_argument('--ratio-name', default='325', type=str)
parser.add_argument('--size-times', default=1, type=int)
parser.add_argument('--aug-steps', default=16, type=int)
parser.add_argument('--ifdebug', default=False, action='store_true', help='if debug')
parser.add_argument('--ifhealthy', default=False, action='store_true', help='if debug')

args = parser.parse_args()
output_dir = './data/Optha_MA_devkit/Optha_MA_part_'+args.task

#generate output_dir
if os.path.exists(output_dir) is False:
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir, 'Annotations'))
    os.mkdir(os.path.join(output_dir, 'ImageSets'))
    os.mkdir(os.path.join(output_dir, 'ImageSets', 'Main'))
    os.mkdir(os.path.join(output_dir, 'JPEGImages'))

imageset_dir = os.path.join(output_dir, 'ImageSets')

files = {}
for file_name in ['test', 'ma_test', 'train', 'ma_train', 'val', 'ma_val', 'trainval', 'ma_trainval']:
    files[file_name+args.ratio_name] = open(os.path.join(imageset_dir, 'Main', '{}{}.txt'.format(file_name, args.ratio_name)), 'w')

train_ratio, test_ratio = int(args.ratio_name[0])*0.1, int(args.ratio_name[-1])*0.1

#deal with healthy images
if args.ifhealthy:
    person_number = len(healthy_persons)
    for person_id, person in enumerate(healthy_persons):
        if person_id < person_number * train_ratio:
            datasetname = 'train'
        elif person_id < person_number * (1 - test_ratio):
            datasetname = 'val'
        else:
            datasetname = 'test'
        ori_img_paths = os.listdir(os.path.join(healthy_person_dir, person))
        ori_img_paths = [ os.path.join(healthy_person_dir, person, ori_img_path) for ori_img_path in ori_img_paths ]
        for ori_img_path in ori_img_paths:
            ori_img = cv2.imread(ori_img_path)
            
            h_total, w_total = ori_img.shape[:2]
            
            h_num = w_num = 1
            h_grid, w_grid = h_total // h_num, w_total // w_num
            
            if datasetname == 'test' or args.ifaug is False:
                aug_steps = 1
            else:
                aug_steps = args.aug_steps
        
            if aug_steps > 1:
                h_num, w_num = h_num - 1, w_num - 1
            
            sqrt_step = int(np.sqrt(aug_steps))
            for step in range(aug_steps):
                h_step = step // sqrt_step
                w_step = step % sqrt_step
                
                h_bias = (h_step * h_grid) // sqrt_step
                w_bias = (w_step * w_grid) // sqrt_step

                for i in range(h_num):
                    for j in range(w_num):
                        h_begin = i * h_grid + h_bias
                        w_begin = j * w_grid + h_bias
                        h_end = min(h_begin + h_grid, h_total)
                        w_end = min(w_begin + w_grid, w_total)
                        new_img = ori_img[h_begin:h_end, w_begin:w_end, :]
                        bboxes = []
                        
                        #remove edge images
                        if new_img[:,:,1].mean() < 10:
                            continue

                        name_suffix = ori_img_path.split('/')[-1]
                        pure_name_suffix = name_suffix.split('.')[0]+'_{}_{}_{}'.format(step, i, j)
                        des_file = pure_name_suffix+'.jpg'
            
                        if args.ifimage:
                            cv2.imwrite(os.path.join(output_dir, 'JPEGImages', des_file), new_img)
                            build_xml(os.path.join(output_dir, 'Annotations'), pure_name_suffix, bboxes, 'ma', des_file, [1, 1, w_end-w_begin, h_end-h_begin], args.iffake)
                        pure_name_suffix += '\n'
                        files[datasetname+args.ratio_name].write(pure_name_suffix)
                        files['ma_'+datasetname+args.ratio_name].write(pure_name_suffix)
                        if datasetname != 'test':
                            files['trainval'+args.ratio_name].write(pure_name_suffix)
                            files['ma_trainval'+args.ratio_name].write(pure_name_suffix)

#add MA images
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
    ori_img_paths.sort()
    annotation_paths.sort()
    for ori_img_path, annotation_path in zip(ori_img_paths, annotation_paths):
        print(ori_img_path, annotation_path)
        ori_img = cv2.imread(ori_img_path)
        annotation = cv2.imread(annotation_path)
        
        h_total, w_total = ori_img.shape[:2]
        
        ret, thresh = cv2.threshold(annotation[:,:, 0], 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        prebboxes = []
        prebboxes_ori = []
        for c_id, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            prebboxes_ori.append([x, y, x+w, y+h])
            center_x = x + w//2
            center_y = y + h//2
            new_x = max(1, center_x - w//2 * args.size_times)
            new_y = max(1, center_y - h//2 * args.size_times)
            new_x2 = min(center_x + w//2 * args.size_times, w_total)
            new_y2 = min(center_y + h//2 * args.size_times, h_total)
            prebboxes.append([new_x, new_y, new_x2, new_y2])

        h_num = w_num = 1
        h_grid, w_grid = h_total // h_num, w_total // w_num
            
        if datasetname == 'test' or args.ifaug is False:
            aug_steps = 1
        else:
            aug_steps = args.aug_steps
        sqrt_step = int(np.sqrt(aug_steps))
        
        if aug_steps > 1:
            h_num, w_num = h_num - 1, w_num - 1
        
        for step in range(aug_steps):
            h_step = step // sqrt_step
            w_step = step % sqrt_step
            #h_bias = np.random.randint(h_grid)
            #w_bias = np.random.randint(w_grid)
            h_bias = (h_step * h_grid) // sqrt_step
            w_bias = (w_step * w_grid) // sqrt_step
            for i in range(h_num):
                for j in range(w_num):
                    h_begin = i * h_grid + h_bias
                    w_begin = j * w_grid + h_bias
                    h_end = min(h_begin + h_grid, h_total)
                    w_end = min(w_begin + w_grid, w_total)
                    new_img = ori_img[h_begin:h_end, w_begin:w_end, :]
                    new_annotation = annotation[h_begin:h_end, w_begin:w_end, :]
                    bboxes = []
                    for bbox, bbox_ori in zip(prebboxes, prebboxes_ori): 
                        if bbox_ori[0] > w_begin and bbox_ori[1] > h_begin and bbox_ori[2] < w_end and bbox_ori[3] < h_end:
                        #if bbox[0] > w_begin and bbox[1] > h_begin and bbox[2] < w_end and bbox[3] < h_end:
                            bboxes.append([max(1, bbox[0]-w_begin), max(1, bbox[1]-h_begin), min(w_end-w_begin, bbox[2]-w_begin), min(h_end-h_begin, bbox[3]-h_begin)])
                    c_number = len(bboxes)
                    #skip empty image
                    if args.ifskip:
                        if c_number == 0:
                            continue
                    
                    def contain(bbox1, bbox2):
                        if bbox1[0] <= bbox2[0] and bbox1[1] <= bbox2[1] and bbox1[2] >= bbox2[2] and bbox1[3] >= bbox2[3]:
                            return True
                        return False
                    #combine bboxes of the same position
                    neglected_ids = set([])
                    for id1 in range(len(bboxes)-1):
                        for id2 in range(id1+1, len(bboxes)):
                            if contain(bboxes[id1], bboxes[id2]):
                                neglected_ids.add(id2)
                            elif contain(bboxes[id2], bboxes[id1]):
                                neglected_ids.add(id1)
                    neglected_bboxes = [ bboxes[neglected_id] for neglected_id in neglected_ids]
                    for neglected_bbox in neglected_bboxes:
                        bboxes.remove(neglected_bbox)
                    if args.ifdebug:
                        for bbox in bboxes:
                            cv2.rectangle(new_annotation, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                            cv2.rectangle(new_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                        show_img = np.concatenate((new_img, new_annotation), axis=0)
                        cv2.imwrite(ori_img_path.split('/')[-1], show_img)
                        #cv2.imshow('show_img', show_img)
                        #cv2.waitKey(0)
                    
                    #remove edge images
                    if new_img[:,:,1].mean() < 10:
                        continue

                    name_suffix = ori_img_path.split('/')[-1]
                    pure_name_suffix = name_suffix.split('.')[0]+'_{}_{}_{}'.format(step, i, j)
                    des_file = pure_name_suffix+'.jpg'
        
                    if args.ifimage:
                        cv2.imwrite(os.path.join(output_dir, 'JPEGImages', des_file), new_img)
                        build_xml(os.path.join(output_dir, 'Annotations'), pure_name_suffix, bboxes, 'ma', des_file, [1, 1, w_end-w_begin, h_end-h_begin], args.iffake)
                    pure_name_suffix += '\n'
                    files[datasetname+args.ratio_name].write(pure_name_suffix)
                    files['ma_'+datasetname+args.ratio_name].write(pure_name_suffix)
                    if datasetname != 'test':
                        files['trainval'+args.ratio_name].write(pure_name_suffix)
                        files['ma_trainval'+args.ratio_name].write(pure_name_suffix)
