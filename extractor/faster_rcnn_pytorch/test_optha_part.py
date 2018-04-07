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


# hyper-parameters
# ------------
cfg_file = 'experiments/cfgs/optha.yml'
if sys.argv[1] == 'healthy':
    imdb_name = 'optha_ma_part_'+sys.argv[1]
    trained_model = 'models/saved_model_optha_part'+sys.argv[2]+'/faster_rcnn_100000.h5'
else:
    imdb_name = 'optha_ma_part_test'+sys.argv[1]
    trained_model = 'models/saved_model_optha_part'+sys.argv[1][:-4]+'/faster_rcnn_100000.h5'
    #trained_model = 'models/saved_model_optha_partdouble/faster_rcnn_100000.h5'

rand_seed = 1024

save_name = 'faster_rcnn_100000'
max_per_image = 300
thresh = 0.05
vis = True

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)

def vis_detections(im, class_name, dets, showcolor, thresh=0.5):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], showcolor, 1)
            #cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.0, showcolor, thickness=1)
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


def test_net(name, net, imdb, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    if sys.argv[1] == 'healthy':
        output_dir = get_output_dir(imdb, name)+'_'+sys.argv[2]
    else:
        output_dir = get_output_dir(imdb, name)
    print(output_dir)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    ori_img_names = {}
    img_hasbb = {}
    padding = 0
    for i in range(num_images):
        img_path = imdb.image_path_at(i)
        ori_name, h_id, w_id = img_path.split('/')[-1].split('.')[0].split('_')
        w_id, h_id = int(w_id), int(h_id)
        im = cv2.imread(img_path)
        h_patch, w_patch = im.shape[:2]
        
        if ori_name in ori_img_names:
            #imshow = ori_img_names[ori_name]
            pass
        else:
            ori_img_names[ori_name] = np.zeros((h_patch*10+padding*9, w_patch*10+padding*9, 3), dtype='uint8')

        
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if vis:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im)

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            
            if len(keep) > 0:
                img_hasbb[ori_name] = True
            cls_dets = cls_dets[keep, :]
            
            index = imdb.image_index[i]
            annotation = imdb._load_optha_annotation(index)
            gt_boxes = annotation['boxes']

            gt_dets = np.ones((len(gt_boxes), 5), dtype=np.float32)
            for g_id, gt_box in enumerate(gt_boxes):
                gt_dets[g_id][:4] = gt_box
            if vis:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets, (255, 0, 0))
                im2show = vis_detections(im2show, imdb.classes[j], gt_dets, (0, 255, 0))
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc(average=False)

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, detect_time, nms_time))

        if vis:
            #cv2.imshow('test', im2show)
            #cv2.waitKey(1)
            ori_img_names[ori_name][h_id*(h_patch+padding):h_id*(h_patch+padding)+h_patch, w_id*(w_patch+padding):w_id*(w_patch+padding)+w_patch, :] = im2show.copy()
    for ori_name, showimg in ori_img_names.items():
        cv2.imwrite(os.path.join(output_dir, ori_name+'_output.png'), showimg)
    print('image level accuray: {} / {} = {}'.format(len(img_hasbb), len(ori_img_names), len(img_hasbb) / len(ori_img_names)))

if __name__ == '__main__':
    # load data
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)

    # load net
    net = FasterRCNN(classes=imdb.classes, debug=False)
    network.load_net(trained_model, net)
    print('load model successfully!'+trained_model)

    net.cuda()
    net.eval()

    # evaluation
    test_net(save_name, net, imdb, max_per_image, thresh=thresh, vis=vis)
