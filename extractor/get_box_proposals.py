#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
import pickle as pkl
from preprocess import get_dataset_info
import sys
from edge_boxes_with_python.edge_boxes import get_windows
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='trainval')
    parser.add_argument('--classname', default='car')
    args = parser.parse_args()
    dataset_name = args.dataset_name
    classname = args.classname
    img_paths, gts = get_dataset_info(dataset_name) 
    
    bbslist_pkl = '../pkls/vot_{}_bbslist_{}.pkl'.format(dataset_name, classname)
    if os.path.exists(bbslist_pkl):
        print('{} exists'.format(bbslist_pkl))
    else:
        img_paths = [os.path.abspath(img_path) for img_path in img_paths]
        bbslist = []
        step = 0
        gap = 1000
        img_no = len(img_paths)
        while step < img_no:
            try:
                bbslist.extend(get_windows(img_paths[step:min(img_no, step+gap)]))
            except:
                import IPython
                IPython.embed()
            step += gap
        pkl.dump([img_paths, bbslist], open(bbslist_pkl, 'wb'))
        print('generating bounding boxes completed, exit')
