import os
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
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
parser.add_argument('--start-step', default=0, type=int)

args = parser.parse_args()

imdb_name = 'optha_ma_part'
imdb_dataset_name = args.datasetname
imdb_ratio = args.ratio_name
imdb_task_name = args.task_name
#imdb_name = 'optha_ma_part_'+args.datasetname+args.ratio_name

cfg_file = 'experiments/cfgs/optha.yml'
output_dir = 'models/saved_model_optha_part_'+args.ratio_name+'_'+args.task_name

start_step = args.start_step
end_step = start_step + 500000
#lr_decay_steps = {60000, 80000}
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
log_dir = cfg.LOG_DIR+'_'+args.ratio_name+'_'+args.task_name
exp_dir = cfg.EXP_DIR

# load data
imdb = get_imdb(imdb_name, imdb_dataset_name, imdb_ratio, imdb_task_name)

rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)

if os.path.exists('pretrained_vgg.pkl'):
    pret_net = pkl.load(open('pretrained_vgg.pkl','rb'))
else:
    pret_net = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
    pkl.dump(pret_net, open('pretrained_vgg.pkl','wb'), pkl.HIGHEST_PROTOCOL)
own_state = net.state_dict()

pret_net_keys = list(pret_net.keys())
own_state_keys = list(own_state.keys())

if not args.resume:
    for name_pret, name_own in zip(pret_net_keys[:26], own_state_keys[:26]):
        param = pret_net[name_pret]
        if isinstance(param, Parameter):
            param = param.data
        try:
            own_state[name_own].copy_(param)
            print('Copied {} to {}'.format(name_pret, name_own))
        except:
            print('Did not find {}'.format(name_pret))
            continue

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
        start_step = checkpoint['step']
        end_step = start_step + 500000
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (step {})".format(args.resume, checkpoint['step']))
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
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):
    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
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
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True

    if use_tensorboard and step % log_interval == 0:
        logger.scalar_summary('train_loss', train_loss / step_cnt, step=step)
        logger.scalar_summary('learning_rate', lr, step=step)
        if _DEBUG:
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

