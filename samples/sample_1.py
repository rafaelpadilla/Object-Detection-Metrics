###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections appling the following metrics:      #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla                                                            #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
import matplotlib.pyplot as plt

def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    import glob
    import os
    # Dictionary containing ground truth bounding boxes
    dictGroundTruth = {}
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath,'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    for f in files:
        nameOfImage = f.replace(".txt","")
        # Create Detections object
        GT_boudingBoxes = BoundingBoxes()
        # Read GT detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n","")
            if line.replace(' ','') == '':
                continue
            splitLine = line.split(" ")
            idClass = int(splitLine[0]) #class
            x = float(splitLine[1])
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(idClass,x,y,w,h,CoordinatesType.Absolute, (200,200), BBType.GroundTruth, format=BBFormat.XYWH)
            GT_boudingBoxes.addBoundingBox(bb)
        fh1.close()
        dictGroundTruth[nameOfImage] = GT_boudingBoxes
    # Dictionary containing detected bounding boxes
    dictDetected = {}
    # Read detections
    folderGT = os.path.join(currentPath,'detections')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    for f in files:
        nameOfImage = f.replace("_det.txt","")
        # Create Detections object
        detected_boundingBoxes = BoundingBoxes()
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n","")
            if line.replace(' ','') == '':
                continue            
            splitLine = line.split(" ")
            idClass = int(splitLine[0]) #class
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(idClass,x,y,w,h,CoordinatesType.Absolute, (200,200), BBType.Detected, confidence, format=BBFormat.XYWH)
            detected_boundingBoxes.addBoundingBox(bb)
        fh1.close()
        dictDetected[nameOfImage] = detected_boundingBoxes
    return [dictGroundTruth, dictDetected]

def createImages(dictGroundTruth, dictDetected):
    """Create representative images with bounding boxes."""
    import numpy as np
    import cv2
    # Define image size
    width = 200
    height = 200
    # Create an empty image
    _dict = {}
    # Loop through the dictionary with ground truth detections
    for key in dictGroundTruth:
        image = np.zeros((height,width,3), np.uint8)
        gt_boundingboxes = dictGroundTruth[key]
        image = gt_boundingboxes.drawAllBoundingBoxes(image)
        detection_boundingboxes = dictDetected[key]
        image = detection_boundingboxes.drawAllBoundingBoxes(image)
        # Show detection and its GT
        cv2.imshow(key, image)
        cv2.waitKey()

# Read txt files containing bounding boxes (ground truth and detections)
[dictGroundTruth, dictDetected] = getBoundingBoxes()
# Generates images based on the bounding boxes
# createImages(dictGroundTruth, dictDetected)
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()

##############################################################
# VOC PASCAL Metrics
##############################################################
# Plot Precision x Recall curve
evaluator.PlotPrecisionRecallCurve(0, # Class to show
                                   dictGroundTruth, # Dictionary with ground truth bounding boxes
                                   dictDetected, # Dictionary with detected bounding boxes
                                   IOUThreshold=0.3, # IOU threshold
                                   showAP=True, # Show Average Precision in the title of the plot
                                   showInterpolatedPrecision=False) # Don't plot the interpolated precision curve
# Get metrics with PASCAL VOC metrics
metricsPerClass = evaluator.GetPascalVOCMetrics(dictGroundTruth, # Dictionary with ground truth bounding boxes
                                        dictDetected, # Dictionary with detected bounding boxes
                                        IOUThreshold=.3) # IOU threshold
# Loop through classes to obtain their metrics
for mc in metricsPerClass:
    # Get metric values per each class
    c = mc['class']
    precision = mc['precision']
    recall = mc['recall']
    average_precision = mc['AP']
    ipre = mc['interpolated precision']
    irec = mc['interpolated recall']    

