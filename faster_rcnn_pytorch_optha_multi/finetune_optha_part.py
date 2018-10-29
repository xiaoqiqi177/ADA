import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torch._six import string_classes, int_classes
import numpy as np
from datetime import datetime
import pickle as pkl

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from logger import Logger
from tqdm import tqdm
import sys
import shutil
import argparse
import cv2
import collections
import re
from data_loader import *

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='train ma dataset')
parser.add_argument('--datasetname', default='trainval', type=str)
parser.add_argument('--ratio-name', default='325', type=str)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--task-name', default='ma_double', type=str)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--ymlname', default='optha_half', type=str)

args = parser.parse_args()

imdb_name = 'optha_ma_part'
imdb_dataset_name = args.datasetname
imdb_ratio = args.ratio_name
imdb_task_name = args.task_name
#imdb_name = 'optha_ma_part_'+args.datasetname+args.ratio_name

cfg_file = 'experiments/cfgs/'+args.ymlname+'.yml'
output_dir = 'models/saved_model_optha_part_'+args.ratio_name+'_'+args.task_name + '_'+args.ymlname

start_epoch = args.start_epoch
end_epoch = start_epoch + 50000
start_step = 0
lr_decay_steps = {}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
#lr = cfg.TRAIN.LEARNING_RATE
lr = 0.0001
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
#log_dir = cfg.LOG_DIR+'_'+args.ratio_name
log_dir = cfg.LOG_DIR+'_'+args.ratio_name+'_'+args.task_name+'_'+args.ymlname
exp_dir = cfg.EXP_DIR

# load data
imdb = get_imdb(imdb_name, imdb_dataset_name, imdb_ratio, imdb_task_name)

rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
#data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)

if not args.resume:
    from keras2pytorch import transfer_keras2pytorch
    #load from keras model
    keras_mdl = '/home/qiqix/EyeWeS/wsdcnn/experiments/wsdcnn16/model.hdf5'
    if os.path.exists(keras_mdl) is False:
        print('exists not {}'.format(keras_mdl))
        exit(0)
    own_state = net.state_dict()
    transfer_keras2pytorch(keras_mdl, own_state)

# Move model to GPU and set train mode
net.cuda()
net.train()

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkout '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        end_epoch = start_epoch + 50000
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tensorboad
if use_tensorboard:
    logger = Logger('./logs', log_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step = 0
re_cnt = False
t = Timer()
t.tic()

def collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return np.array(batch)
    elif isinstance(batch[0], int_classes):
        return np.array(batch)
    elif isinstance(batch[0], float):
        return np.array(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


trainval_dataset = OpthaDataset('trainval', roidb, imdb.num_classes)
train_sampler = None
trainval_loader = torch.utils.data.DataLoader(
        trainval_dataset, batch_size = 1, shuffle = (train_sampler is None),
        num_workers = 0, collate_fn = collate, pin_memory = True, sampler = train_sampler)

step = start_step
step_cnt = 0

def train(loader, net, optimizer, epoch, logger):
    global tp, tf, fg, bg
    global train_loss, step, step_cnt
    global re_cnt
    global lr
    for i, blobs in enumerate(loader):
        im_data = blobs['data'][0]
        im_info = blobs['im_info'][0]
        gt_boxes = blobs['gt_boxes'][0]
        gt_ishard = blobs['gt_ishard'][0]
        dontcare_areas = blobs['dontcare_areas'][0]
        
        if False:
            show_img = np.uint8(im_data + cfg.PIXEL_MEANS)[0]
            for gt_box in gt_boxes:
                cv2.rectangle(show_img, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]), (0, 255, 0), 1)
            #cv2.imshow('img', show_img)
            cv2.imwrite('img{}.png'.format(step), show_img)
            cv2.waitKey(0)
        
        # forward
        net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
        loss = net.loss + net.rpn.loss

        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

        train_loss += loss.data[0]
        step += 1
        step_cnt += 1

        # backward
        optimizer.zero_grad()
        loss.backward()
        network.clip_gradient(net, 10.)
        optimizer.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration

            log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
                step, blobs['im_name'][0], train_loss / step_cnt, fps, 1./fps)
            log_print(log_text, color='green', attrs=['bold'])

            if fg > 0:
                log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            else:
                log_print('\tTF: %.2f%%, fg/bg=(%d/%d)' % (tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
            re_cnt = True

        if use_tensorboard and step % log_interval == 0:
            logger.scalar_summary('train_loss', train_loss / step_cnt, step=step)
            logger.scalar_summary('learning_rate', lr, step=step)
            if fg > 0:
                logger.scalar_summary('true_positive', tp/fg*100., step=step)
            logger.scalar_summary('true_negative', tf/bg*100., step=step)
            losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                      'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                      'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                      'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
            #logger.scalar_summary(losses, step=step)

        if (step % 10000 == 0) and step > 0:
            #save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
            save_name = os.path.join(output_dir, 'faster_rcnn_{}.pth.tar'.format(step))
            state = {
                'epoch': epoch,
                'step': step,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict()
                }
            torch.save(state, save_name)
            #network.save_net(save_name, net)
            #print('save model: {}'.format(save_name))
    
        if step in lr_decay_steps:
            lr *= lr_decay
            optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)

        if re_cnt:
            tp, tf, fg, bg = 0., 0., 0, 0
            train_loss = 0
            step_cnt = 0
            t.tic()
            re_cnt = False

for epoch in range(start_epoch, end_epoch+1):
    train(trainval_loader, net, optimizer, epoch, logger)

