#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
from tqdm import tqdm

ROOT_DIR = './train/'
ANNOTATION_DIR = './train/annotations'

INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'ma',
        'supercategory': 'optha',
    },
]

def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main():
    for datasetname in ['train', 'val', 'test']:
        image_dir = './train/optha_{}2018'.format(datasetname)
        coco_output = build_json(image_dir)
        with open('{}/instances_optha_{}2018.json'.format(ANNOTATION_DIR, datasetname), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)


def build_json(image_dir):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    for root, _, files in os.walk(image_dir):
        image_files = filter_for_jpeg(root, files)
        # go through each image
        for image_filename in tqdm(image_files):
            image = Image.open(image_filename)
            image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)

            # filter for associated png annotations
            #for root, _, files in os.walk(ANNOTATION_DIR):
            if True:
                #annotation_files = filter_for_annotations(root, files, image_filename)
                annotation_files = [ANNOTATION_DIR + '/'+image_filename.split('/')[-1].split('.')[0]+'_ma_0.png']

                # go through each associated annotation
                for annotation_filename in annotation_files:
                    
                    if 'ma' in annotation_filename:
                        class_id = 1

                    category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                    binary_mask = np.asarray(Image.open(annotation_filename)
                        .convert('1')).astype(np.uint8)
                    
                    annotation_info = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info, binary_mask,
                        image.size, tolerance=2)
                    
                    if annotation_info is None:
                        continue
                    
                    coco_output["images"].append(image_info)
                    coco_output["annotations"].append(annotation_info)

                    segmentation_id = segmentation_id + 1

            image_id = image_id + 1
    return coco_output

if __name__ == "__main__":
    main()
