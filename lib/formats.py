import os
import xml.etree.ElementTree as ET

from .bounding_box import BoundingBox
from .bounding_boxes import BoundingBoxes
from .utils import BBFormat, BBType, CoordinatesType, get_files_dir


class PASCAL_VOC_BoundingBox(BoundingBox):
    # Herdar de bounding box, mas adicionar o parametro difficult
    def __init__(self, xml_file_path, bb_type, class_id, coordinates, img_size, difficult=False):
        file_name = os.path.basename(xml_file_path).replace('.xml', '')
        BoundingBox.__init__(self,
                             image_name=file_name,
                             class_id=class_id,
                             coordinates=coordinates,
                             img_size=img_size,
                             type_coordinates=CoordinatesType.ABSOLUTE,
                             format=BBFormat.XYX2Y2,
                             bb_type=bb_type)
        # For PASCAL VOC dataset, the tag difficult is important
        self._difficult = difficult
        self._file_path = xml_file_path


class Datasets_Parser():
    @staticmethod
    def parse_file_PASCAL_VOC(file_path, bounding_boxes=None):
        assert file_path.lower().endswith('.xml')
        root = ET.parse(file_path).getroot()
        # Get only important attibutes
        img_width = int(root.findall('./size/width')[0].text)
        img_height = int(root.findall('./size/height')[0].text)
        if bounding_boxes is None:
            bounding_boxes = BoundingBoxes()
        for obj in root.findall('./object'):
            bb = PASCAL_VOC_BoundingBox(xml_file_path=file_path,
                                        class_id=obj.findall('name')[0].text,
                                        coordinates=(float(obj.findall('bndbox/xmin')[0].text),
                                                     float(obj.findall('bndbox/ymin')[0].text),
                                                     float(obj.findall('bndbox/xmax')[0].text),
                                                     float(obj.findall('bndbox/ymax')[0].text)),
                                        difficult=obj.findall('difficult')[0].text,
                                        img_size=(img_width, img_height),
                                        bb_type=BBType.GROUND_TRUTH)
            bounding_boxes.add_bounding_box(bb)
        return bounding_boxes

    @staticmethod
    def get_annotations_PASCAL_VOC(directory):
        files = [os.path.join(directory, f) for f in get_files_dir(directory, extensions=['xml'])]
        bounding_boxes = BoundingBoxes()
        for file_path in files:
            bounding_boxes = Datasets_Parser.parse_file_PASCAL_VOC(file_path, bounding_boxes)
        return True, bounding_boxes

    @staticmethod
    def parse_file_xyx2y2(file_path,
                          bounding_boxes=None,
                          img_size=None,
                          type_coordinates=CoordinatesType.ABSOLUTE,
                          bb_type=BBType.DETECTED):
        name_of_image = os.path.basename(file_path).replace(".txt", "")
        if bounding_boxes is None:
            bounding_boxes = BoundingBoxes()
        # Read bounding boxes from txt file
        with open(file_path, "r") as f:
            for line in f:
                line = line.replace("\n", "")
                if line.replace(' ', '') == '':
                    continue
                splitLine = line.split(" ")
                class_id = splitLine[0]  # class
                # if file contains confidence, there will be 6 values
                if len(splitLine) == 6:
                    confidence = float(splitLine[1])
                    x = float(splitLine[2])
                    y = float(splitLine[3])
                    w = float(splitLine[4])
                    h = float(splitLine[5])
                else:
                    confidence = None
                    x = float(splitLine[1])
                    y = float(splitLine[2])
                    w = float(splitLine[3])
                    h = float(splitLine[4])
                bb = BoundingBox(image_name=name_of_image,
                                 class_id=class_id,
                                 coordinates=(x, y, w, h),
                                 type_coordinates=type_coordinates,
                                 img_size=img_size,
                                 bb_type=bb_type,
                                 class_confidence=confidence,
                                 format=BBFormat.XYX2Y2)
                bounding_boxes.add_bounding_box(bb)
        return bounding_boxes

    @staticmethod
    def get_annotations_xyx2y2(directory, type_coordinates, bb_type, img_directory=None):
        txt_files = get_files_dir(directory, extensions=['txt', None])
        files_annotations = [os.path.join(directory, f) for f in txt_files]
        # If relative coordinates are used, get size of the images
        img_files = []
        if type_coordinates is CoordinatesType.RELATIVE:
            for file_annotation in txt_files:
                f = os.path.join(img_directory, file_annotation)
                # Make sure image file exists
                if os.path.isfile(f) is False:
                    return False, f'Image file {f} not found.'
                img_files.append(f)
        else:
            img_files = [None] * len(txt_files)
        bounding_boxes = BoundingBoxes()
        for i, annot_file_path in enumerate(files_annotations):
            bounding_boxes = Datasets_Parser.parse_file_xyx2y2(annot_file_path,
                                                               bounding_boxes=bounding_boxes,
                                                               img_size=img_files[i],
                                                               type_coordinates=type_coordinates,
                                                               bb_type=bb_type)
        return True, bounding_boxes
