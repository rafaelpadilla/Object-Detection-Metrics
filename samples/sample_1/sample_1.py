###########################################################################################
#                                                                                         #
# This sample demonstrates:                                                               #
# * How to create your own bounding boxes (detections and ground truth) manually;         #
# * Ground truth bounding boxes are drawn in green and detected boxes are drawn in red;   #
# * Create objects of the class BoundingBoxes with your bounding boxes;                   #
# * Create images with detections and ground truth;                                       #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import os

import _init_paths
import cv2
from lib.BoundingBox import BoundingBox
from lib.BoundingBoxes import BoundingBoxes
from lib.utils import *

###########################
# Defining bounding boxes #
###########################
# Ground truth bounding boxes of 000001.jpg
gt_boundingBox_1 = BoundingBox(
    imageName='000001',
    classId='dog',
    x=0.34419263456090654,
    y=0.611,
    w=0.4164305949008499,
    h=0.262,
    typeCoordinates=CoordinatesType.Relative,
    bbType=BBType.GroundTruth,
    format=BBFormat.XYWH,
    imgSize=(353, 500))
gt_boundingBox_2 = BoundingBox(
    imageName='000001',
    classId='person',
    x=0.509915014164306,
    y=0.51,
    w=0.9745042492917847,
    h=0.972,
    typeCoordinates=CoordinatesType.Relative,
    bbType=BBType.GroundTruth,
    format=BBFormat.XYWH,
    imgSize=(353, 500))
# Ground truth bounding boxes of 000002.jpg
gt_boundingBox_3 = BoundingBox(
    imageName='000002',
    classId='train',
    x=0.5164179104477612,
    y=0.501,
    w=0.20298507462686569,
    h=0.202,
    typeCoordinates=CoordinatesType.Relative,
    bbType=BBType.GroundTruth,
    format=BBFormat.XYWH,
    imgSize=(335, 500))
# Ground truth bounding boxes of 000003.jpg
gt_boundingBox_4 = BoundingBox(
    imageName='000003',
    classId='bench',
    x=0.338,
    y=0.4666666666666667,
    w=0.184,
    h=0.10666666666666666,
    typeCoordinates=CoordinatesType.Relative,
    bbType=BBType.GroundTruth,
    format=BBFormat.XYWH,
    imgSize=(500, 375))
gt_boundingBox_5 = BoundingBox(
    imageName='000003',
    classId='bench',
    x=0.546,
    y=0.48133333333333334,
    w=0.136,
    h=0.13066666666666665,
    typeCoordinates=CoordinatesType.Relative,
    bbType=BBType.GroundTruth,
    format=BBFormat.XYWH,
    imgSize=(500, 375))
# Detected bounding boxes of 000001.jpg
detected_boundingBox_1 = BoundingBox(
    imageName='000001',
    classId='person',
    classConfidence=0.893202,
    x=52,
    y=4,
    w=352,
    h=442,
    typeCoordinates=CoordinatesType.Absolute,
    bbType=BBType.Detected,
    format=BBFormat.XYX2Y2,
    imgSize=(353, 500))
# Detected bounding boxes of 000002.jpg
detected_boundingBox_2 = BoundingBox(
    imageName='000002',
    classId='train',
    classConfidence=0.863700,
    x=140,
    y=195,
    w=209,
    h=293,
    typeCoordinates=CoordinatesType.Absolute,
    bbType=BBType.Detected,
    format=BBFormat.XYX2Y2,
    imgSize=(335, 500))
# Detected bounding boxes of 000003.jpg
detected_boundingBox_3 = BoundingBox(
    imageName='000003',
    classId='bench',
    classConfidence=0.278000,
    x=388,
    y=288,
    w=493,
    h=331,
    typeCoordinates=CoordinatesType.Absolute,
    bbType=BBType.Detected,
    format=BBFormat.XYX2Y2,
    imgSize=(500, 375))
# Creating the object of the class BoundingBoxes
myBoundingBoxes = BoundingBoxes()
# Add all bounding boxes to the BoundingBoxes object:
myBoundingBoxes.addBoundingBox(gt_boundingBox_1)
myBoundingBoxes.addBoundingBox(gt_boundingBox_2)
myBoundingBoxes.addBoundingBox(gt_boundingBox_3)
myBoundingBoxes.addBoundingBox(gt_boundingBox_4)
myBoundingBoxes.addBoundingBox(gt_boundingBox_5)
myBoundingBoxes.addBoundingBox(detected_boundingBox_1)
myBoundingBoxes.addBoundingBox(detected_boundingBox_2)
myBoundingBoxes.addBoundingBox(detected_boundingBox_3)

###################
# Creating images #
###################
currentPath = os.path.dirname(os.path.realpath(__file__))
gtImages = ['000001', '000002', '000003']
for imageName in gtImages:
    im = cv2.imread(os.path.join(currentPath, 'images', 'groundtruths', imageName) + '.jpg')
    # Add bounding boxes
    im = myBoundingBoxes.drawAllBoundingBoxes(im, imageName)
    # cv2.imshow(imageName+'.jpg', im)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(currentPath, 'images', imageName + '.jpg'), im)
    print('Image %s created successfully!' % imageName)
