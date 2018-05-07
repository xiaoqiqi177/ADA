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
parser.add_argument('--epochs', default='10000', type=str)
parser.add_argument('--ymlname', default='optha_half', type=str)
parser.add_argument('--trained-model', default='', type=str)
parser.add_argument('--thresh', default=0.5, type=float)

args = parser.parse_args()

imdb_name = 'optha_ma_part'
imdb_dataset_name = args.datasetname
imdb_ratio = args.ratio_name
imdb_task_name = args.task_name

cfg_file = 'experiments/cfgs/'+args.ymlname+'.yml'
#output_dir = 'models/saved_model_optha_part_'+args.ratio_name+'_'+args.task_name
#trained_model = os.path.join(output_dir, 'faster_rcnn_'+args.epochs+'.pth.tar')
trained_model = args.trained_model

save_name = 'faster_rcnn_'+args.ratio_name +'_'+args.task_name + '_' + args.datasetname

max_per_image = 300
vis = True

# ------------

# load config
cfg_from_file(cfg_file)

def vis_detections(im, class_name, dets, showcolor, thresh, gt = False):
    """Visual debugging of detections."""
    for i in range(dets.shape[0]):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], showcolor, 1)
            if gt is False:
                cv2.putText(im, '%.3f' % (score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, showcolor, thickness=1)
    return im


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

def test_net(name, net, imdb, max_per_image=300, thresh=0.5, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)

    output_dir = './output/'+ args.datasetname + '_'+ args.ratio_name+'_'+ args.task_name + '_' + args.epochs + '_'+str(args.thresh)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    detected_bboxes = []
    gt_bboxes = []
    for i in range(num_images):
        img_path = imdb.image_path_at(i)
        ori_name, step_id, h_id, w_id = img_path.split('/')[-1].split('.')[0].split('_')
        ori_name = ori_name + '_' + step_id
        w_id, h_id = int(w_id), int(h_id)
        im = cv2.imread(img_path)
        h, w = im.shape[:2]
        
        _t['im_detect'].tic()
        w_patch = (w - 1) // cfg.TRAIN.CROP_W + 1
        h_patch = (h - 1) // cfg.TRAIN.CROP_H + 1
        new_im = np.zeros((h_patch * cfg.TRAIN.CROP_H, w_patch * cfg.TRAIN.CROP_W, 3), dtype=np.uint8)
        new_im[:h, :w, :] = im.copy()
        cls_dets_all = []
        for h_id in range(h_patch):
            for w_id in range(w_patch):
                h_delta = h_id * cfg.TRAIN.CROP_H
                w_delta = w_id * cfg.TRAIN.CROP_W
                cropped_im = new_im[h_delta:h_delta+cfg.TRAIN.CROP_H, w_delta:w_delta+cfg.TRAIN.CROP_W,:].copy()
                scores, boxes = im_detect(net, cropped_im)
                detect_time = _t['im_detect'].toc(average=False)

                _t['misc'].tic()

                # skip j = 0, because it's the background class
                for j in range(1, imdb.num_classes):
                    inds = np.where(scores[:, j] >= thresh)[0]
                    cls_scores = scores[inds, j]
                    cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
                    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep, :]
                    cls_dets_all.extend(cls_dets + np.array([w_delta, h_delta, w_delta, h_delta, 0])[None, :]) 
                
        cls_dets_all = np.array(cls_dets_all, dtype='float32')
        index = imdb.image_index[i]
        annotation = imdb._load_optha_annotation(index)
        where_gt = np.where(annotation['gt_classes'] == j)
        gt_boxes = annotation['boxes'][where_gt]
        gt_dets = np.ones((len(gt_boxes), 5), dtype=np.float32)
        for g_id, gt_box in enumerate(gt_boxes):
            gt_dets[g_id][:4] = gt_box
        
        #im2show = im.copy()
        im2show = new_im.copy()
        im2show = vis_detections(im2show, imdb.classes[j], gt_dets, (0, 255, 0), thresh=thresh, gt=True)
        im2show = vis_detections(im2show, imdb.classes[j], cls_dets_all, (255, 0, 0), thresh=thresh)
        for h_id in range(h_patch):
            for w_id in range(w_patch):
                h_delta = h_id * cfg.TRAIN.CROP_H
                w_delta = w_id * cfg.TRAIN.CROP_W
                cropped_im = im2show[h_delta:h_delta+cfg.TRAIN.CROP_H, w_delta:w_delta+cfg.TRAIN.CROP_W,:].copy()
                cropped_im_ori= new_im[h_delta:h_delta+cfg.TRAIN.CROP_H, w_delta:w_delta+cfg.TRAIN.CROP_W,:].copy()
                #cv2.imwrite(os.path.join(output_dir, str(h_id)+'_'+str(w_id)+'_'+ori_name+'_vis.png'), cropped_im)
                #cv2.imwrite(os.path.join(output_dir, str(h_id)+'_'+str(w_id)+'_'+ori_name+'_ori.png'), cropped_im_ori)

        cv2.imwrite(os.path.join(output_dir, ori_name+'_output.png'), im2show)
        detected_bboxes.append(cls_dets_all)
        gt_bboxes.append(gt_dets)
        #print('im_detect: {:d}/{:d} {:.3f}s' \
        #    .format(i + 1, num_images, detect_time))

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
    #print('load model {} successfully!'.format(trained_model))

    net.cuda()
    net.eval()
    # evaluation
    print('thresh: ', args.thresh)
    test_net(save_name, net, imdb, max_per_image, thresh=args.thresh, vis=vis)
