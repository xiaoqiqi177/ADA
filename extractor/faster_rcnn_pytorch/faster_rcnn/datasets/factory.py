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
for trainratio in range(1, 9):
    for testratio in range(1, 10-trainratio):
        valratio = 10-trainratio-testratio
        ratio_name = str(trainratio)+str(valratio)+str(testratio)
        for datasetname in ['train', 'val', 'trainval', 'test']:
            splitset.append(datasetname+ratio_name)

for split in splitset:
    for ispart in [True, False]:
        if ispart is False:
            name = 'optha_ma_{}'.format(split)
        else:
            name = 'optha_ma_{}_{}'.format('part', split)
        __sets[name] = (lambda split=split, ispart=ispart:
                    optha_ma(split, ispart))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not name in __sets:
        # print (list_imdbs())
        raise KeyError('Unknown dataset: {}'.format(name))
    
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
