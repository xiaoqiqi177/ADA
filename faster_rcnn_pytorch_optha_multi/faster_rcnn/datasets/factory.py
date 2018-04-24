# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

import numpy as np

from .pascal_voc import pascal_voc
from .imagenet3d import imagenet3d
from .optha_ma import optha_ma

def _selective_search_IJCV_top_k(split, year, top_k):
    """Return an imdb that uses the top k proposals from the selective search
    IJCV code.
    """
    imdb = pascal_voc(split, year)
    imdb.roidb_handler = imdb.selective_search_IJCV_roidb
    imdb.config['top_k'] = top_k
    return imdb


# Set up voc_<year>_<split> using selective search "fast" mode
for year in ['2007', '2012', '0712']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year:
                        pascal_voc(split, year))

splitset = ['healthy']
for trainratio in range(0, 9):
    for testratio in range(1, 10-trainratio):
        valratio = 10-trainratio-testratio
        ratio_name = str(trainratio)+str(valratio)+str(testratio)
        for datasetname in ['train', 'val', 'trainval', 'test']:
            for iffull in ['', 'full']:
                splitset.append(datasetname+ratio_name+iffull)

def get_imdb(name, dataset_name, ratio, task_name):
    """Get an imdb (image database) by name."""
    return optha_ma(dataset_name+ratio, task_name)

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
