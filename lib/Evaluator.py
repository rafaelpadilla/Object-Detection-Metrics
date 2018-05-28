from BoundingBox import *
from BoundingBoxes import *
import matplotlib.pyplot as plt
from collections import Counter
from utils import *
import numpy as np
import sys

class Evaluator:
    
    def GetPascalVOCMetrics(self, dictGroundTruth, dictDetected, IOUThreshold=0.5):
        ret = [] # list containing metrics (precision, recall, average precision) of each class
        # dictionary with classId, precision array, recall array, average precision
        # List with all ground truths (Ex: [imageName, class, confidence=1, (bb coordinates XYX2Y2)])
        groundTruths = [] 
        # List with all detections (Ex: [imageName, class, confidence, (bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through dictionary of ground truth bounding boxes
        for key in dictGroundTruth.keys(): 
            boundingboxes = dictGroundTruth[key].getBoundingBoxes()
            for bb in boundingboxes:
                # [imageName, class, confidence, (bb coordinates XYX2Y2)]
                groundTruths.append([key, bb.getClassId(), 1, bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
                # get class
                if bb.getClassId() not in classes:
                    classes.append(bb.getClassId())
        # Loop through dictionary of ground truth bounding boxes
        for key in dictDetected.keys(): 
            boundingboxes = dictDetected[key].getBoundingBoxes()
            for bb in boundingboxes:
                # [imageName, class, confidence, (bb coordinates XYX2Y2)]
                detections.append([key, bb.getClassId(), bb.getConfidence(), bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
                # get class
                if bb.getClassId() not in classes:
                    classes.append(bb.getClassId())
        classes = sorted(classes)
        ## Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in groundTruths if g[1] == c]
            npos = len(gts)
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key,val in det.items():
                det[key] = np.zeros(val)
            # print("Evaluating class: %d (%d detections)" % (c, len(dects)))
            # Loop through detections
            for d in range(len(dects)):
                # print('dect %s => %s' % (dects[d][0], dects[d][3],))
                # Find ground truth image
                gt = [gt for gt in gts if gt[0] == dects[d][0]]
                iouMax = sys.float_info.min
                for j in range(len(gt)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou>iouMax:
                        iouMax=iou
                        jmax=j
                # Assign detection as true positive/don't care/false positive
                if iouMax>=IOUThreshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d]=1  # count as true positive
                    det[dects[d][0]][jmax]=1 # flag as already 'seen'
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d]=1 # count as false positive
            # compute precision, recall and average precision
            acc_FP=np.cumsum(FP)
            acc_TP=np.cumsum(TP)
            rec=acc_TP/npos
            prec=np.divide(acc_TP,(acc_FP+acc_TP))
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision' : prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP)                
                }
            ret.append(r)
        return ret
    
    def PlotPrecisionRecallCurve(self, classId, dictGroundTruth, dictDetected, IOUThreshold=0.5, showAP=False, showInterpolatedPrecision=False, savePath=None):
        results = self.GetPascalVOCMetrics(dictGroundTruth, dictDetected, IOUThreshold)

        result = None
        for res in results:
            if res['class'] == classId:
                result = res
                break
        if result == None:
            raise IOError('Error: Class %d could not be found.' % classId)

        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        plt.plot(recall, precision, label='Precision')
        if showInterpolatedPrecision:
            plt.plot(mrec, mpre, '--r' , label='Interpolated precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        if showAP:
            plt.title('Precision x Recall curve \nClass: %d, AP: %.4f' % (classId, average_precision))
        else:
            plt.title('Precision x Recall curve \nClass: %d' % classId)
        plt.legend(shadow=True)
        plt.grid()
        if savePath != None:
            plt.savefig(savePath)            
        plt.show()
        
        ret ={}
        ret['class'] = classId
        ret['precision'] = precision
        ret['recall'] = recall
        ret['AP'] = average_precision
        ret['interpolated precision'] = mpre
        ret['interpolated recall'] = mrec
        ret['total positives'] = npos
        ret['total TP'] = total_tp
        ret['total FP'] = total_fp
        return ret

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre)-1, 0, -1):
            mpre[i-1]=max(mpre[i-1],mpre[i])
        ii = []
        for i in range(len(mrec)-1):
            if mrec[1:][i]!=mrec[0:-1][i]:
                ii.append(i+1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i]-mrec[i-1])*mpre[i])
        return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]


    # For each detections, calculate IOU with reference
    @staticmethod
    def _getAllIOUs(reference, detections):
        ret = []
        bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        # img = np.zeros((200,200,3), np.uint8)
        for d in detections:
            bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            iou = Evaluator.iou(bbReference,bb)
            # Show blank image with the bounding boxes
            # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
            # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
            ret.append((iou,reference,d)) # iou, reference, detection
        # cv2.imshow("comparing",img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("comparing")
        return sorted(ret, key=lambda i: i[0], reverse=True)# sort by iou (from highest to lowest)

    @staticmethod
    def Teste(gts, detections):
        img = np.zeros((200,200,3), np.uint8)
        for d in gts:
            img = add_bb_into_image(img, d, color=(0,255,0), thickness=2)
        for d in detections:
            img = add_bb_into_image(img, d, color=(255,0,0), thickness=2)
        cv2.imshow("comparing",img)
        cv2.waitKey(0)
        cv2.destroyWindow("comparing")

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) == False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA,boxB)
        union = Evaluator._getUnionAreas(boxA,boxB,interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]: 
            return False # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False # boxA is below boxB
        return True
    
    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)
    
    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea == None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)