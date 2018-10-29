import os
import sys
import torch
import cv2
import pickle
import numpy as np

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir
import argparse

# hyper-parameters
# ------------
parser = argparse.ArgumentParser(description='train ma dataset')
parser.add_argument('--datasetname', default='trainval', type=str)
parser.add_argument('--ratio-name', default='721', type=str)
parser.add_argument('--task-name', default='ma_double', type=str)
parser.add_argument('--method-name', default='10000', type=str)
parser.add_argument('--ymlname', default='optha_half', type=str)
parser.add_argument('--trained-model', default='', type=str)

args = parser.parse_args()

imdb_name = 'optha_ma_part'
imdb_dataset_name = args.datasetname
imdb_ratio = args.ratio_name
imdb_task_name = args.task_name

cfg_file = 'experiments/cfgs/'+args.ymlname+'.yml'
trained_model = args.trained_model

save_name = 'faster_rcnn_'+args.ratio_name +'_'+args.task_name + '_' + args.datasetname

max_per_image = 300
#thresh_set = [0.95]
thresh_set = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.996, 0.997, 0.998, 0.999, 0.9995, 0.9999, 1.]

# load config
cfg_from_file(cfg_file)

def im_detect(net, image):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)
    cls_prob, bbox_pred, rois = net(im_data, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

def test_net(name, net, imdb, max_per_image=300):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    output_dir = './output/'+ args.datasetname + '_'+ args.ratio_name+'_'+ args.task_name + '_' + args.method_name
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    detected_bboxes = []
    gt_bboxes = []
    for i in range(num_images):
        img_path = imdb.image_path_at(i)
        ori_name, step_id, h_id, w_id = img_path.split('/')[-1].split('.')[0].split('_')
        ori_name = ori_name + '_' + step_id
        w_id, h_id = int(w_id), int(h_id)
        im = cv2.imread(img_path)
        h, w = im.shape[:2]
        
        w_patch = (w - 1) // cfg.TRAIN.CROP_W + 1
        h_patch = (h - 1) // cfg.TRAIN.CROP_H + 1
        new_im = np.zeros((h_patch * cfg.TRAIN.CROP_H, w_patch * cfg.TRAIN.CROP_W, 3), dtype=np.uint8)
        new_im[:h, :w, :] = im.copy()
        cls_dets_all = [[] for i in range(len(thresh_set))]
        for h_id in range(h_patch):
            for w_id in range(w_patch):
                h_delta = h_id * cfg.TRAIN.CROP_H
                w_delta = w_id * cfg.TRAIN.CROP_W
                cropped_im = new_im[h_delta:h_delta+cfg.TRAIN.CROP_H, w_delta:w_delta+cfg.TRAIN.CROP_W,:].copy()
                scores, boxes = im_detect(net, cropped_im)

                for j, thresh in enumerate(thresh_set):
                    inds = np.where(scores[:, 1] >= thresh)[0]
                    cls_scores = scores[inds, 1]
                    cls_boxes = boxes[inds, 4:8]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep, :]
                    cls_dets_all[j].extend(cls_dets + np.array([w_delta, h_delta, w_delta, h_delta, 0])[None, :]) 
                
        index = imdb.image_index[i]
        annotation = imdb._load_optha_annotation(index)
        where_gt = np.where(annotation['gt_classes'] == 1)
        gt_boxes = annotation['boxes'][where_gt]
        gt_dets = np.ones((len(gt_boxes), 5), dtype=np.float32)
        for g_id, gt_box in enumerate(gt_boxes):
            gt_dets[g_id][:4] = gt_box
        
        detected_bboxes.append(cls_dets_all)
        gt_bboxes.append(gt_dets)
    with open(os.path.join(output_dir, 'result.pkl'), 'wb') as f:
        pickle.dump([detected_bboxes, gt_bboxes], f)

if __name__ == '__main__':
    # load data
    imdb = get_imdb(imdb_name, imdb_dataset_name, imdb_ratio, imdb_task_name)
    imdb.competition_mode(on=True)

    # load net
    net = FasterRCNN(classes=imdb.classes, debug=False)
    #network.load_net(trained_model, net)
    model = torch.load(trained_model)
    net.load_state_dict(model['state_dict'])

    net.cuda()
    net.eval()
    test_net(save_name, net, imdb, max_per_image)
