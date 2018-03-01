#!/usr/bin/env python
# coding: utf-8
import cv2
import os
import numpy as np
import sys
import os
import glob
import xml.etree.ElementTree as ET
def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    info = []
    filename = root.findtext('filename')
    for object in root.findall('object'):
        classname = object.findtext('name')
        bndbox = object.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        objectinfo = [classname, [ymin, xmin, ymax, xmax]]
        info.append(objectinfo)
    return filename, info

def get_dataset_info(datasetname):
    data_dir = '../../data'
    data_dir = os.path.join('../../data', 'VOC2007-'+datasetname)
    
    xml_paths = glob.glob(os.path.join(data_dir, 'Annotations')+'/*.xml')
    xml_paths.sort()
    dataset_info = []
    img_paths = []

    for xml_path in xml_paths:
        imgpath, info = read_xml(xml_path)
        dataset_info.append(info)
        img_paths.append(imgpath)
    return img_paths, dataset_info

def read_xml_oneclass(xml_path, target_classname):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    info = []
    filename = root.findtext('filename')
    for object in root.findall('object'):
        classname = object.findtext('name')
        if classname != target_classname:
            continue
        bndbox = object.find('bndbox')
        xmin = int(bndbox.findtext('xmin'))
        ymin = int(bndbox.findtext('ymin'))
        xmax = int(bndbox.findtext('xmax'))
        ymax = int(bndbox.findtext('ymax'))
        objectinfo = [ymin, xmin, ymax, xmax]
        info.append(objectinfo)
    return filename, info

def get_dataset_info_oneclass(datasetname, target_classname):
    data_dir = os.path.join('../../data', 'VOC2007-'+datasetname)
    data_dir = os.path.abspath(data_dir)
    xml_paths = glob.glob(os.path.join(data_dir, 'Annotations')+'/*.xml')
    xml_paths.sort()
    dataset_info = []
    img_paths = []

    for xml_path in xml_paths:
        imgpath, info = read_xml_oneclass(xml_path, target_classname)
        if len(info) == 0:
            continue
        dataset_info.append(info)
        img_paths.append(os.path.join(data_dir, 'JPEGImages', imgpath))
    return img_paths, dataset_info


if __name__ == '__main__':
    img_paths, dataset_info = get_dataset_info('trainval')
    import IPython
    IPython.embed()
