###########################################################################################
#                                                                                         #
# Evaluator class: Implements the most popular metrics for object detection               #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: May 24th 2018                                                 #
###########################################################################################

from BoundingBox import *
from BoundingBoxes import *
import matplotlib.pyplot as plt
from collections import Counter
from utils import *
import numpy as np
import sys

class Evaluator:
    
    def GetPascalVOCMetrics(self, boundingboxes, IOUThreshold=0.5):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Get
        Args:
            boundingboxes: Object of the class BoundingBoxes representing ground truth and detected bounding boxes;
            IOUThreshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5).
        Returns:
            A list of dictionaries. Each dictionary contains information and metrics of each class. 
            The keys of each dictionary are: 
            dict['class']: class representing the current dictionary; 
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        ret = [] # list containing metrics (precision, recall, average precision) of each class
        # List with all ground truths (Ex: [imageName, class, confidence=1, (bb coordinates XYX2Y2)])
        groundTruths = [] 
        # List with all detections (Ex: [imageName, class, confidence, (bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in boundingboxes.getBoundingBoxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([bb.getImageName(), bb.getClassId(), 1, bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
            else:
                detections.append([bb.getImageName(), bb.getClassId(), bb.getConfidence(), bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
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
            # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
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
                        # print("TP")
                    det[dects[d][0]][jmax]=1 # flag as already 'seen'
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d]=1 # count as false positive
                    # print("FP")
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

    def PlotPrecisionRecallCurve(self, classId, boundingBoxes, IOUThreshold=0.5, showAP=False, showInterpolatedPrecision=False, savePath=None, showGraphic=True):
        """PlotPrecisionRecallCurve
        Plot the Precision x Recall curve for a given class.
        Args:
            classId: The class that will be plot; 
            boundingBoxes: Object of the class BoundingBoxes representing ground truth and detected bounding boxes; 
            IOUThreshold (optional): IOU threshold indicating which detections will be considered TP or FP (default value = 0.5); 
            showAP (optional): if True, the average precision value will be shown in the title of the graph (default = False); 
            showInterpolatedPrecision (optional): if True, it will show in the plot the interpolated precision (default = False); 
            savePath (optional): if informed, the plot will be saved as an image in this path (ex: /home/mywork/ap.png) (default = None); 
            showGraphic (optional): if True, the plot will be shown (default = True)	
        Returns:
            A dictionary containing information and metric about the class. The keys of the dictionary are:
            dict['class']: class representing the current dictionary; 
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        results = self.GetPascalVOCMetrics(boundingBoxes, IOUThreshold)
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

        if showInterpolatedPrecision:
            plt.plot(mrec, mpre, '--r' , label='Interpolated precision')
        plt.plot(recall, precision, label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        if showAP:
            ap_str = "{0:.2f}%".format(average_precision*100)
            plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
            # plt.title('Precision x Recall curve \nClass: %s, AP: %.4f' % (str(classId), average_precision))
        else:
            plt.title('Precision x Recall curve \nClass: %d' % classId)
        plt.legend(shadow=True)
        plt.grid()

        ############################################################
        # Uncomment the following block to create plot with points #
        ############################################################
        # plt.plot(recall, precision, 'bo')
        # labels = ['R', 'Y', 'J', 'A', 'U', 'C', 'M', 'F', 'D', 'B', 'H', 'P', 'E', 'X', 'N', 'T', 'K', 'Q', 'V', 'I', 'L', 'S', 'G', 'O']
        # dicPosition = {}
        # dicPosition['left_zero'] = (-30,0)
        # dicPosition['left_zero_slight'] = (-30,-10)
        # dicPosition['right_zero'] = (30,0)
        # dicPosition['left_up'] = (-30,20)
        # dicPosition['left_down'] = (-30,-25)
        # dicPosition['right_up'] = (20,20)
        # dicPosition['right_down'] = (20,-20)
        # dicPosition['up_zero'] = (0,30)
        # dicPosition['up_right'] = (0,30)
        # dicPosition['left_zero_long'] = (-60,-2)
        # dicPosition['down_zero'] = (-2,-30)
        # vecPositions = [dicPosition['left_down'], dicPosition['left_zero'], dicPosition['right_zero'], dicPosition['right_zero'], #'R', 'Y', 'J', 'A', 
        #                 dicPosition['left_up'], dicPosition['left_up'], dicPosition['right_up'], dicPosition['left_up'], # 'U', 'C', 'M', 'F', 
        #                 dicPosition['left_zero'], dicPosition['right_up'], dicPosition['right_down'], dicPosition['down_zero'], #'D', 'B', 'H', 'P'
        #                 dicPosition['left_up'], dicPosition['up_zero'], dicPosition['right_up'], dicPosition['left_up'], # 'E', 'X', 'N', 'T',
        #                 dicPosition['left_zero'], dicPosition['right_zero'], dicPosition['left_zero_long'], dicPosition['left_zero_slight'], # 'K', 'Q', 'V', 'I',
        #                 dicPosition['right_down'], dicPosition['left_down'], dicPosition['right_up'], dicPosition['down_zero']] # 'L', 'S', 'G', 'O'
        # for idx in range(len(labels)):
        #     box = dict(boxstyle='round,pad=.5',facecolor='yellow',alpha=0.5)
        #     plt.annotate(labels[idx], 
        #                 xy=(recall[idx],precision[idx]), xycoords='data',
        #                 xytext=vecPositions[idx], textcoords='offset points',
        #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
        #                 bbox=box)


        if savePath != None:
            plt.savefig(savePath)
        if showGraphic == True:
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
        ii = []
        for i,j in zip(range(len(mpre)-1, 0, -1),range(len(mrec)-1)):
            mpre[i-1]=max(mpre[i-1],mpre[i])
            if mrec[j]!=mrec[j+1]:
                ii.append(j+1)

   
            
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i]-mrec[i-1])*mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

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