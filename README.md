# ADA: A Game-Theoretic Perspective on Data Augmentation for Object Detection

This is an implementation of [ADA](https://arxiv.org/pdf/1710.07735.pdf) using Python, Pytorch and Matlab.

The repository includes:
* Library of edge_boxes and faster_rcnn for bounding box proposal and feature extracting.
* Python code of the ADA algorithm.
* Part results of experiment on VOC 2007 dataset.

# Getting Started
## 1. Download gurobi and activate it.
## 2. Compile [faster_rcnn](extractor/faster_rcnn_pytorch/faster_rcnn)

# Steps

## 1. Preprocess steps to extract features: 
### obtain and store bounding box proposals information.
run [get_box_proposals.py](extractor/get_box_proposals.py).
### obtain and store features of proposals and groundtruths.
run [feature_extractor_bbslist.py](extractor/feature_extractor_bbslist.py) and [feature_extractor_gt.py](extractor/feature_extractor_gt.py).

## 2. Train
run [train_ada.py](ada/train_ada.py).
## 3. Test
run [test_ada.py](ada/test_ada.py).
