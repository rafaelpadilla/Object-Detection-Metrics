###########################################################################################
#                                                                                         #
# Evaluator class: Implements the most popular metrics for object detection               #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: Oct 9th 2018                                                 #
###########################################################################################

import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from .bounding_box import BoundingBox
from .bounding_boxes import BoundingBoxes
from .utils import BBFormat, BBType, MethodAveragePrecision


class Evaluator:
    def get_pascal_voc_metrics(self,
                               bounding_boxes,
                               IOU_threshold=0.5,
                               method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Get
        Args:
            bounding_boxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOU_threshold: IOU threshold indicating which detections will be considered TP or FP
            (default value = 0.5);
            method (default = EVERY_POINT_INTERPOLATION): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EVERY_POINT_INTERPOLATION), or applying the 11-point
            interpolatiom as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge";
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
            dict['total FP']: total number of False Positive detections;
        """
        ret = []  # list containing metrics (precision, recall, average precision) of each class
        # List with all ground truths (Ex: [imageName,class,confidence=1, (bb coordinates XYX2Y2)])
        ground_truths = []
        # List with all detections (Ex: [imageName,class,confidence,(bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in bounding_boxes.get_bounding_boxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.get_bb_type() == BBType.GROUND_TRUTH:
                ground_truths.append([
                    bb.get_image_name(),
                    bb.get_class_id(), 1,
                    bb.get_absolute_bounding_box(BBFormat.XYX2Y2)
                ])
            else:
                detections.append([
                    bb.get_image_name(),
                    bb.get_class_id(),
                    bb.get_confidence(),
                    bb.get_absolute_bounding_box(BBFormat.XYX2Y2)
                ])
            # get class
            if bb.get_class_id() not in classes:
                classes.append(bb.get_class_id())
        classes = sorted(classes)
        # Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in ground_truths if g[1] == c]
            npos = len(gts)
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key, val in det.items():
                det[key] = np.zeros(val)
            print("Evaluating class: %s (%d detections and %d groundtruths)" %
                  (str(c), len(dects), len(gts)))
            # Loop through detections
            for d in range(len(dects)):
                print('dect %s => %s' % (
                    dects[d][0],
                    dects[d][3],
                ))
                # Find ground truth image
                gt = [gt for gt in gts if gt[0] == dects[d][0]]
                iou_max = sys.float_info.min
                for j in range(len(gt)):
                    print('Ground truth gt => %s' % (gt[j][3], ))
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou > iou_max:
                        iou_max = iou
                        jmax = j
                # Assign detection as true positive/don't care/false positive
                if iou_max >= IOU_threshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d] = 1  # count as true positive
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                        # print("TP")
                    else:
                        FP[d] = 1  # count as false positive
                        # print("FP")
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOU_threshold.
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # compute precision, recall and average precision
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                # [ap, mpre, mrec, ii] = Evaluator.calculate_average_precision(rec, prec)
                result = Evaluator.calculate_average_precision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.eleven_point_interpolated_ap(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': result['ap'],
                'interpolated precision': result['mpre'],
                'interpolated recall': result['mrec'],
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP)
            }
            ret.append(r)
        return ret

    def plot_precision_recall_curve(self,
                                    bounding_boxes,
                                    IOU_threshold=0.5,
                                    method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                                    show_AP=False,
                                    show_interpolated_precision=False,
                                    save_path=None,
                                    show_graphic=True):
        """ Plot the Precision x Recall curve for a given class.
        Args:
            bounding_boxes: Object of the class BoundingBoxes representing ground truth and detected
            bounding boxes;
            IOU_threshold (optional): IOU threshold indicating which detections will be considered
            TP or FP (default value = 0.5);
            method (default = EVERY_POINT_INTERPOLATION): It can be calculated as the implementation
            in the official PASCAL VOC toolkit (EVERY_POINT_INTERPOLATION), or applying the 11-point
            interpolatio as described in the paper "The PASCAL Visual Object Classes(VOC) Challenge"
            or EVERY_POINT_INTERPOLATION";
            show_AP (optional): if True, the average precision value will be shown in the title of
            the graph (default = False);
            show_interpolated_precision (optional): if True, it will show in the plot the interpolated
             precision (default = False);
            save_path (optional): if informed, the plot will be saved as an image in this path
            (ex: /home/mywork/ap.png) (default = None);
            show_graphic (optional): if True, the plot will be shown (default = True)
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
        results = self.get_pascal_voc_metrics(bounding_boxes, IOU_threshold, method)
        result = None
        # Each resut represents a class
        for result in results:
            if result is None:
                raise IOError('Error: Class could not be found.')

            class_id = result['class']
            precision = result['precision']
            recall = result['recall']
            average_precision = result['AP']
            mpre = result['interpolated precision']
            mrec = result['interpolated recall']
            plt.close()
            if show_interpolated_precision:
                if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
                    plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
                elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
                    # Remove duplicates, getting only the highest precision of each recall value
                    nrec = []
                    nprec = []
                    for idx in range(len(mrec)):
                        r = mrec[idx]
                        if r not in nrec:
                            idxEq = np.argwhere(mrec == r)
                            nrec.append(r)
                            nprec.append(max([mpre[int(id)] for id in idxEq]))
                    plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
            plt.plot(recall, precision, label='Precision')
            plt.xlabel('recall')
            plt.ylabel('precision')
            if show_AP:
                ap_str = "{0:.2f}%".format(average_precision * 100)
                # ap_str = "{0:.4f}%".format(average_precision * 100)
                plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(class_id), ap_str))
            else:
                plt.title('Precision x Recall curve \nClass: %s' % str(class_id))
            plt.legend(shadow=True)
            plt.grid()
            ############################################################
            # Uncomment the following block to create plot with points #
            ############################################################
            # plt.plot(recall, precision, 'bo')
            # labels = ['R', 'Y', 'J', 'A', 'U', 'C', 'M', 'F', 'D', 'B', 'H', 'P', 'E', 'X', 'N', 'T',
            # 'K', 'Q', 'V', 'I', 'L', 'S', 'G', 'O']
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
            # vecPositions = [
            #     dicPosition['left_down'],
            #     dicPosition['left_zero'],
            #     dicPosition['right_zero'],
            #     dicPosition['right_zero'],  #'R', 'Y', 'J', 'A',
            #     dicPosition['left_up'],
            #     dicPosition['left_up'],
            #     dicPosition['right_up'],
            #     dicPosition['left_up'],  # 'U', 'C', 'M', 'F',
            #     dicPosition['left_zero'],
            #     dicPosition['right_up'],
            #     dicPosition['right_down'],
            #     dicPosition['down_zero'],  #'D', 'B', 'H', 'P'
            #     dicPosition['left_up'],
            #     dicPosition['up_zero'],
            #     dicPosition['right_up'],
            #     dicPosition['left_up'],  # 'E', 'X', 'N', 'T',
            #     dicPosition['left_zero'],
            #     dicPosition['right_zero'],
            #     dicPosition['left_zero_long'],
            #     dicPosition['left_zero_slight'],  # 'K', 'Q', 'V', 'I',
            #     dicPosition['right_down'],
            #     dicPosition['left_down'],
            #     dicPosition['right_up'],
            #     dicPosition['down_zero']
            # ]  # 'L', 'S', 'G', 'O'
            # for idx in range(len(labels)):
            #     box = dict(boxstyle='round,pad=.5',facecolor='yellow',alpha=0.5)
            #     plt.annotate(labels[idx],
            #                 xy=(recall[idx],precision[idx]), xycoords='data',
            #                 xytext=vecPositions[idx], textcoords='offset points',
            #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            #                 bbox=box)
            if save_path is not None:
                plt.savefig(os.path.join(save_path, class_id + '.png'))
            if show_graphic is True:
                plt.show()
                plt.pause(0.05)
        return results

    @staticmethod
    def calculate_average_precision(rec, prec):
        """ Static function that calculates the Average-Precision (AP) given a list of recalls
        and precisions.

        Parameters
        ----------
        rec : list
            List with recall values.
        prec : list
            List with precisions.

        Returns
        -------
        bool
        """
        #AQUI TODO

        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return {'ap': ap, 'mpre': mpre[0:len(mpre) - 1], 'mrec': mrec[0:len(mpre) - 1], 'ii': ii}
        # [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    def eleven_point_interpolated_ap(rec, prec):
        """ Calculate the 11-point Average-Precision (AP) given a list of recalls
        and precisions.

        Parameters
        ----------
        rec : list
            List with recall values.
        prec : list
            List with precisions.

        Returns
        -------
        bool
        """
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recall_values = np.linspace(0, 1, 11)
        recall_values = list(recall_values[::-1])
        rho_interp = []
        recallValid = []
        # For each recall_values (0, 0.1, 0.2, ... , 1)
        for r in recall_values:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rho_interp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rho_interp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rho_interp]
        pvals.append(0)
        # rho_interp = rho_interp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recall_values = [i[0] for i in cc]
        rho_interp = [i[1] for i in cc]
        return {'ap': ap, 'mpre': mpre, 'mrec': mrec}
        #  [ap, rho_interp, recall_values, None]

    @staticmethod
    def _get_all_IOUs(reference, detections):
        """ Static function that calculates IOUs between a given reference and all detections found
            within an image.

        Parameters
        ----------
        reference : BoundingBox
            BoundingBox object representing the reference box.
        detections : list
            List of BoundingBox objects representing all detected boxes within an image.

        Returns
        -------
        list
            List with all IOUs sorted from the highest to lowest values.
        """
        ret = []
        bbReference = reference.get_absolute_bounding_box(BBFormat.XYX2Y2)
        # img = np.zeros((200,200,3), np.uint8)
        for d in detections:
            bb = d.get_absolute_bounding_box(BBFormat.XYX2Y2)
            iou = Evaluator.iou(bbReference, bb)
            # Show blank image with the bounding boxes
            # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
            # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
            ret.append((iou, reference, d))  # iou, reference, detection
        # cv2.imshow("comparing",img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("comparing")
        return sorted(ret, key=lambda i: i[0], reverse=True)  # sort by iou (from highest to lowest)

    @staticmethod
    def iou(boxA, boxB):
        """ Static function that calculates Intersection Over Union (IOU) between two bonding boxes.

        IOU is a value between 0 and 1 that represents how well two bounding boxes are overlapped.
        IOU reaches 1 if there is a perfect overlap between the two boxes, which means they cover
        exactly the same area of the image. IOU is 0 if their areas do not have any pixel in common.

        ----------
        boxA : BoundingBox
            BoundingBox object representing a box.
        boxB : BoundingBox
            BoundingBox object representing another box.

        Returns
        -------
        float
            Value between 0 and 1 representing the Intersection Over Union (IOU) between the two
            boxes.
        """
        # if boxes do not intersect
        if Evaluator._boxes_intersect(boxA, boxB) is False:
            return 0
        inter_area = Evaluator._get_intersection_area(boxA, boxB)
        union = Evaluator._get_union_areas(boxA, boxB, inter_area=inter_area)
        # intersection over union
        iou = inter_area / union
        print(iou)
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxes_intersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _get_intersection_area(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _get_union_areas(boxA, boxB, inter_area=None):
        area_A = Evaluator._get_area(boxA)
        area_B = Evaluator._get_area(boxB)
        if inter_area is None:
            inter_area = Evaluator._get_intersection_area(boxA, boxB)
        return float(area_A + area_B - inter_area)

    @staticmethod
    def _get_area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
