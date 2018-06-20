from BoundingBox import *
from utils import *

class BoundingBoxes:

    def __init__(self):
        self._boundingBoxes = []

    def addBoundingBox(self, bb):
        self._boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if BoundingBox.compare(d,_boundingBox):
                del self._boundingBoxes[d]
                return
    
    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []
    
    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId: # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType] 

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName] 

    def count(self, bbType=None):
        if bbType == None: # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType: # get only specified bb type
                count += 1
        return count
    
    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes

    def drawAllBoundingBoxes(self, image, imageName):
        bbxes = self.getBoundingBoxesByImageName(imageName)
        for bb in bbxes:
            if bb.getBBType() == BBType.GroundTruth: #if ground truth
                image = add_bb_into_image(image, bb ,color=(0,255,0)) #green
            else: #if detection
                image = add_bb_into_image(image, bb ,color=(255,0,0)) #red
        return image   

    # def drawAllBoundingBoxes(self, image):
    #     for gt in self.getBoundingBoxesByType(BBType.GroundTruth):
    #         image = add_bb_into_image(image, gt ,color=(0,255,0))
    #     for det in self.getBoundingBoxesByType(BBType.Detected):
    #         image = add_bb_into_image(image, det ,color=(255,0,0))
    #     return image    

#     @staticmethod
#     def evaluateDetections(gtDetection, evalDetection, minIoUTruePos=0.0):
#         #####################################################################
#         # Evaluate Intersection over Union (IoU):
#         #
#         # - Only detected objects that are overlapped with the same class 
#         #       groundtruth objects will be taken into account (e.g. a "cat"
#         #       can only be compared to another "cat").
#         # - Among multiple detections for a unique groundtruth bounding box,
#         #       only the one with the highest IoU will be taken into consid-
#         #       eration.
#         # - As the highest IoU is taken into consideration, the bounding box
#         #       pair that is considered a match (one from groundtruth and the 
#         #       other is the detected one), is removed. This way these bound-
#         #       ing boxes won't be considered in further checking for this 
#         #       image. Then only maximum IoU will be added to IoU_sum for 
#         #       the image.
#         # - The IoU of an image pair is the average of its IoU.
#         # - IoU of the image = IoU_sum / (True_positives + False_positives) which
#         #       is the same as average(IoU_sum)
#         #
#         #####################################################################
#         # Evaluate True Positive (TP) and False Positive (FP):
#         #
#         # - If detected "cat" isn't overlaped with "cat" (or overlaped with "dog"),
#         #       then this is false_positive and its IoU = 0
#         # - All non-maximum IoUs = 0 and they are false_positives.
#         # - If a detected "cat" is somehow overlapped with a "cat", then it is 
#         #       accounted as a True Positive and no other detection will be 
#         #       considered for those bounding boxes.
#         #####################################################################
#         IoUs = []
#         # Initiate True Positives and False Positive counts
#         TP = 0
#         FP = 0
#         FN = 0
#         # Detections.Teste(gtDetection, evalDetection)
#         dets = Detections.SeparateClasses(gtDetection, evalDetection)


# # Multiple detections of the same object in an image are considered
# # false detections e.g. 5 detections of a single object is counted as 1 correct detec-
# # tion and 4 false detections – it is the responsibility of the participant’s system
# # to filter multiple detections from its output

#         # For each class, get its GTs and detections
#         for d in dets:
#             gtDetections = d[1]
#             evalDetections = d[2]

#             # for each evalDetection, find the best (lowest IOU) gtDetection
#             # note: the eval detection must be the same class
#             while len(gtDetections) > 0:
#                 for detEval in evalDetections:
#                     bb = detEval.getAbsoluteBoundingBox()
#                     bestIoU = 0
#                     bestGT = None
#                     # find the bb with lowest IOU
#                     for detGT in gtDetections:
#                         iou = YOLOHelper.iou(bb, detGT.getAbsoluteBoundingBox())

#                         # Show blank image with the bounding boxes
#                         img = np.zeros((detGT.height_img,detGT.width_img,3), np.uint8)
#                         aa = detEval.getAbsoluteBoundingBox()
#                         img = cv2.rectangle(img, (aa[0],aa[1]), (aa[2],aa[3]), (0,0,255), 6)
#                         for de in evalDetection.detections:
#                             aaa = de.getAbsoluteBoundingBox()
#                             img = cv2.rectangle(img, (aaa[0],aaa[1]), (aaa[2],aaa[3]), (0,0,255), 2)
#                         bbb = detGT.getAbsoluteBoundingBox()
#                         img = cv2.rectangle(img, (bbb[0],bbb[1]), (bbb[2],bbb[3]), (0,255,0), 6)
#                         for gt in gtDetection.detections:
#                             bbb = gt.getAbsoluteBoundingBox()
#                             img = cv2.rectangle(img, (bbb[0],bbb[1]), (bbb[2],bbb[3]), (0,255,0), 2)
#                         cv2.imshow("IoU %.2f" % iou,img)
#                         cv2.waitKey(0)
#                         cv2.destroyWindow("IoU %.2f" % iou)

#                         if iou > 1 or iou < 0:
#                             raise ValueError('IOU value out of limits: %f' % iou)

#                         if iou > bestIoU:
#                             bestGT = detGT
#                             bestIoU = iou
                        
#                     IoUs.append(bestIoU)
#                     # Increment False Positives or True Positives
#                     if bestIoU == 0: # Detection makes no overlap with any object: it is a false positive
#                         FP = FP + 1
#                     elif (bestIoU >= minIoUTruePos): 
#                         TP = TP + 1
#                     else: #bestIoU < minIoUTruePos : The overlapped IOU is below the threshold
#                         FP = FP + 1 # Detection is a false positive

#                     # Now remove this detected BB and go to the next                
#                     evalDetection.detections.remove(detEval)
#                     continue
            
#                 FN = FN + len(gtDetection.detections)
#         # Return average IoU among all detected bounding boxes, True Positives and False Positives
#         return FN, TP, FP