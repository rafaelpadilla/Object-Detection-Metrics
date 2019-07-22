from utils import (BBFormat, BBType, CoordinatesType,
                   convert_to_absolute_values, convert_to_relative_values)


class BoundingBox:
    """ Class representing a bounding box. """

    def __init__(self,
                 image_name,
                 class_id,
                 x,
                 y,
                 w,
                 h,
                 type_coordinates=CoordinatesType.ABSOLUTE,
                 img_size=None,
                 bb_type=BBType.GROUND_TRUTH,
                 class_confidence=None,
                 format=BBFormat.XYWH):
        """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the image name.
            class_id : str
                String value representing class id.
            x : float
                Float value representing the X upper-left coordinate of the bounding box.
            y : float
                Float value representing the Y upper-left coordinate of the bounding box.
            w : float
                Float value representing the width bounding box.
            h : float
                Float value representing the height bounding box.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the class_confidence must be informed.
            class_confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, class_confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        self._image_name = image_name
        self._type_coordinates = type_coordinates
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        if bb_type == BBType.DETECTED and class_confidence is None:
            raise IOError(
                'For bb_type=\'Detected\', it is necessary to inform the class_confidence value.')

        self._class_confidence = class_confidence
        self._bb_type = bb_type
        self._class_id = class_id
        self._format = format

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (type_coordinates == CoordinatesType.RELATIVE):
            (self._x, self._y, self._w,
             self._h) = convert_to_absolute_values(img_size, (x, y, w, h))
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if format == BBFormat.XYWH:
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = x
            self._y = y
            if format == BBFormat.XYWH:
                self._w = w
                self._h = h
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = w
                self._y2 = h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        """ Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def get_relative_bounding_box(self, img_size=None):
        """ Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        if img_size is None:
            return convert_to_relative_values((img_size[0], img_size[1]),
                                              (self._x, self._y, self._w, self._h))
        else:
            return convert_to_relative_values((self._width_img, self._height_img),
                                              (self._x, self._y, self._w, self._h))

    def get_image_name(self):
        """ Get the string that represents the image.

        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """ Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.

        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._class_confidence

    def get_format(self):
        """ Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).

        Returns
        -------
        Enum
            Format of the bounding box. It can be either:
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        return self._format

    def get_class_id(self):
        """ Get the class of the object the bounding box represents.

        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """ Get the size of the image where the bounding box is represented.

        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return (self._width_img, self._height_img)

    def get_coordinates_type(self):
        """ Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).

        Returns
        -------
        Enum
            Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        """
        return self._type_coordinates

    def get_bb_type(self):
        """ Get type of the bounding box that represents if it is a ground-truth or detected box.

        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    @staticmethod
    def compare(det1, det2):
        """ Static function to compare if two bounding boxes represent the same area in the image,
            regardless the format of their boxes.

        Parameters
        ----------
        det1 : BoundingBox
            BoundingBox object representing one bounding box.
        dete2 : BoundingBox
            BoundingBox object representing another bounding box.

        Returns
        -------
        bool
            True if both bounding boxes have the same coordinates, otherwise False.
        """
        det1BB = det1.getAbsoluteBoundingBox()
        det1img_size = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2img_size = det2.getImageSize()

        if det1.get_class_id() == det2.get_class_id() and \
           det1.class_confidence == det2.classConfidenc() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1img_size[0] == det1img_size[0] and \
           det2img_size[1] == det2img_size[1]:
            return True
        return False

    @staticmethod
    def clone(bounding_box):
        """ Static function to clone a given bounding box.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box object to be cloned.

        Returns
        -------
        BoundingBox
            Cloned BoundingBox object.
        """
        absBB = bounding_box.get_absolute_bounding_box(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        new_bounding_box = BoundingBox(bounding_box.get_image_name(),
                                       bounding_box.get_class_id(),
                                       absBB[0],
                                       absBB[1],
                                       absBB[2],
                                       absBB[3],
                                       type_coordinates=bounding_box.getCoordinatesType(),
                                       img_size=bounding_box.getImageSize(),
                                       bb_type=bounding_box.getbb_type(),
                                       class_confidence=bounding_box.getConfidence(),
                                       format=BBFormat.XYWH)
        return new_bounding_box
