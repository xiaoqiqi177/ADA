import numpy as np
import torch.utils.data as data
from faster_rcnn.roi_data_layer.minibatch import get_minibatch

class OpthaDataset(data.Dataset):
    def __init__(self, dataset_name, roidb, num_classes):
        self._roidb = roidb
        self._num_classes = num_classes

    def __getitem__(self, index):
        #add transform
        minibatch_db = [ self._roidb[index] ]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        return blobs

    def __len__(self):
        return len(self._roidb)
