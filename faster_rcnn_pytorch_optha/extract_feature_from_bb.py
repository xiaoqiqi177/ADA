import cv2
import numpy as np
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
import os
def build_extractor(model_file, classes=None):
    if classes is None:
        extractor = FasterRCNN()
    else:
        extractor = FasterRCNN(classes)
    extractor.cuda()
    extractor.eval()
    network.load_net(model_file, extractor)
    print('load model successfully!')
    return extractor

def extractfeatures(im_file, extractor, dets):    
    image = cv2.imread(im_file)
    DEBUG = False
    if DEBUG:
        for det in dets:
            cv2.rectangle(image, (int(det[1]), int(det[2])), (int(det[3]), int(det[4])), (0, 204, 0), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)
    features = extractor.extract(image, dets)
    return features

if __name__ == '__main__':
    im_file = 'demo/004545.jpg'
    model_file = 'models/VGGnet_fast_rcnn_iter_70000.h5'
    dets = np.asarray([[0., 261.32, 8.64, 354.23, 228.82],
        [0., 426.56, 119.70, 454.20, 183.62],
        [0., 140.35, 207.47, 207.22, 354.97]])
    extractor = build_extractor(model_file)
    features = extractfeatures(im_file, extractor, dets)
    print(features)
