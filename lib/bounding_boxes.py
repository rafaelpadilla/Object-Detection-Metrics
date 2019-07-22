from bounding_box import BoundingBox
from utils import BBType, add_bb_into_image


class BoundingBoxes:
    """ Class representing a group of bounding boxes. """

    def __init__(self):
        self._bounding_boxes = []

    def add_bounding_box(self, bb):
        """ Adds a bounding box to the list.

        Parameters
        ----------
            bb : BoundingBox
                BoundingBox object.
        """
        self._bounding_boxes.append(bb)

    def remove_bounding_box(self, bb):
        """ Remove a bounding box from the list.

        Parameters
        ----------
            bb : BoundingBox
                BoundingBox object to be removed.
        """
        for d in self._bounding_boxes:
            if BoundingBox.compare(d, bb):
                del self._bounding_boxes[d]
                return

    def get_bounding_boxes(self):
        """ Get list of bounding boxes.

        Returns
        -------
            list
                List containing the BoundingBox objects.
        """
        return self._bounding_boxes

    def get_bounding_box_by_class(self, class_id):
        """ Returns all bounding boxes of a given class.

        Parameters
        ----------
            class_id : str
                Class of the object to be returned.
        Returns
        -------
            list
                List containing the BoundingBox objects of the specified class.
        """
        boundingBoxes = []
        for d in self._bounding_boxes:
            if d.get_class_id() == class_id:  # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def get_classes(self):
        """ Returns a list of classes in the bounding boxes.

        Returns
        -------
            list
                List containing all classes represented by the BoundingBoxes object.
        """
        classes = []
        for d in self._bounding_boxes:
            c = d.get_class_id()
            if c not in classes:
                classes.append(c)
        return classes

    def get_bounding_boxes_by_type(self, bb_type):
        """ Returns all bounding boxes of a given type (GroundTruth or Detected).

        Parameters
        ----------
            bb_type : Enum
                Enum representing the type of the bounding box (GroundTruth or Detected).
        Returns
        -------
            list
                List containing the BoundingBox objects of the specified type.
        """
        # get only specified bb type
        return [d for d in self._bounding_boxes if d.get_bb_type() == bb_type]

    def get_bounding_boxes_by_image_name(self, image_name):
        """ Returns all bounding boxes of a given image.

        Parameters
        ----------
            image_name : str
                string representing the image.
        Returns
        -------
            list
                List containing the BoundingBox objects within the image.
        """
        # get only specified bb type
        return [d for d in self._bounding_boxes if d.get_image_name() == image_name]

    def count(self, bb_type=None):
        """ Counts the amount of bounding boxes of a given type.

        Parameters
        ----------
            bb_type : Enum
                Enum representing the type of the bounding box (GroundTruth or Detected).
        Returns
        -------
            int
                Amount of bounding boxes of the specified type.
        """
        if bb_type is None:  # Return all bounding boxes
            return len(self._bounding_boxes)
        count = 0
        for d in self._bounding_boxes:
            if d.get_bb_type() == bb_type:  # get only specified bb type
                count += 1
        return count

    def clone(self):
        """ Clone the BoundingBoxes object.

        Returns
        -------
            BoundingBoxes
                Cloned boundingBoxes object.
        """
        newBoundingBoxes = BoundingBoxes()
        for d in self._bounding_boxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.add_bounding_box(det)
        return newBoundingBoxes

    def draw_all_bounding_boxes(self, image, image_name):
        """ Draws a bounding box in the image given its name.
        If the bounding box type is GroundTruth, the color of the bounding box is green.
        If the bounding box type is a detection, the color of the bounding box is red.

        Parameters
        ----------
            image : opencv image
                Image to be added the bounding box.
            image_name : str
                Name of the image inserterd in the list of bounding boxes to be drawn.
        Returns
        -------
            opencv image
                Image with the bounding box drawn.
        """
        bbxes = self.get_bounding_boxes_by_image_name(image_name)
        for bb in bbxes:
            if bb.get_bb_type() == BBType.GROUND_TRUTH:  # if ground truth
                image = add_bb_into_image(image, bb, color=(0, 255, 0))  # green
            else:  # if detection
                image = add_bb_into_image(image, bb, color=(255, 0, 0))  # red
        return image
