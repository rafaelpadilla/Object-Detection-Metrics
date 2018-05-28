from utils import *

class BoundingBox:
    def __init__(self, classId, x, y, w, h, typeCoordinates = CoordinatesType.Relative, imgSize = None, bbType=BBType.GroundTruth, classConfidence=None, format=BBFormat.XYWH):
        """Constructor.
        Args:
            classId: Integer value representing class id.
            x: Float value representing the X upper-left coordinate of the bounding box.
            y: Float value representing the Y upper-left coordinate of the bounding box.
            w: Float value representing the width bounding box.
            h: Float value representing the height bounding box.
            typeCoordinates: (optional) Enum (Relative or Absolute) represents if the bounding box coordinates (x,y,w,h) are absolute or relative to size of the image. Default: 'Relative'.
            imgSize: (optional) 2D vector (width, height)=>(int, int) represents the size of the image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
            bbType: (optional) Enum (Groundtruth or Detection) identifies if the bounding box represents a ground truth or a detection. If it is a detection, the classConfidence has to be informed.
            classConfidence: (optional) Float value representing the confidence of the detected class. If detectionType is Detection, classConfidence needs to be informed.
        """
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize == None:
            raise IOError('Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if bbType == BBType.Detected and classConfidence == None:
            raise IOError('For bbType=\'Detection\', it is necessary to inform the classConfidence value.')
        if classConfidence != None and (classConfidence < 0 or classConfidence > 1):
            raise IOError('classConfidence value must be a real value between 0 and 1.')

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        # If relative coordinates, convert to absolute values
        if (typeCoordinates == CoordinatesType.Relative):
            (self._x,self._y,self._w,self._h) = convertToAbsoluteValues(imgSize, (x,y,w,h))
            self._width_img = imgSize[0]
            self._height_img =  imgSize[1]
            if format==BBFormat.XYWH:
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2-self._x
                self._h = self._y2-self._y
            else:
                # Needed to implement
                raise IOError('To implement')
        else:
            self._x = x
            self._y = y
            if format==BBFormat.XYWH:
                self._w = w
                self._h = h
                self._x2 = self._x+self._w
                self._y2 = self._y+self._h
            else:
                self._x2 = w
                self._y2 = h
                self._w = self._x2-self._x
                self._h = self._y2+self._y
        if imgSize == None:
            self._width_img = None
            self._height_img =  None

    def getAbsoluteBoundingBox(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x,self._y,self._w,self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x,self._y,self._x2,self._y2)

    def getRelativeBoundingBox(self, imgSize=None):
        if imgSize==None and self._width_img==None and self._height_img==None:
            raise IOError('Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if imgSize==None:
            return convertToRelativeValues((imgSize[0], imgSize[1]), (self._x,self._y,self._w,self._h))
        else:
            return convertToRelativeValues((self._width_img, self._height_img), (self._x,self._y,self._w,self._h))
    
    def getConfidence(self):
        return self._classConfidence
    
    def getClassId(self):
        return self._classId

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates
    
    def getBBType(self):
        return self._bbType
        
    @staticmethod
    def compare(det1, det2):
        det1BB = det1.getAbsoluteBoundingBox()
        det1ImgSize = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2ImgSize = det2.getImageSize()
        
        if det1.getClassId() == det2.getClassId() and \
           det1.classConfidence == det2.classConfidenc() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1ImgSize[0] == det1ImgSize[0] and \
           det2ImgSize[1] == det2ImgSize[1]:
           return True
        return False     
    
    @staticmethod
    def clone(boundingBox):
        newBoundingBox = BoundingBox(boundingBox.classId, boundingBox.classConfidence, \
                                boundingBox.x, boundingBox.y,  boundingBox.w,  boundingBox.h,\
                                (boundingBox.width_img, boundingBox.height_img))
        return newBoundingBox