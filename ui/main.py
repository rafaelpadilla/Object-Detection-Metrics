# TODO:
# 1)
# Gerar histograma com quantidade de bounding boxes por classe
# Neste histograma colocar um traço com a média de bounding boxes por classe
# (eixo x: classes; eixo y: quantidades de bounding boxes)

# 2)
# Gerar histograma com quantidade de classes por imagem
# (eixo x: classes; eixo y: quantidade de imagens presente)

import os

from PyQt5.QtWidgets import QFileDialog, QMainWindow

from design.main_ui import Ui_Dialog as Main_UI
from details import Details_Dialog
from lib.bounding_boxes import BoundingBoxes
from lib.formats import Datasets_Parser
from lib.utils import BBFormat, BBType, CoordinatesType


class Main_Dialog(QMainWindow, Main_UI):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.dialog_statistics = Details_Dialog()
        # initialize variables
        self.dir_annotations_gt = ''
        self.dir_images_gt = ''
        self.dir_annotations_det = ''

        # Deleteme
        directory = '/home/rafael/thesis/Object-Detection-Metrics/samples/sample_3/detections'
        self.txb_det_dir.setText(directory)
        self.dir_annotations_det = directory
        directory = '/home/rafael/Downloads/Annotations'
        directory = '/home/rafael/thesis/Object-Detection-Metrics/samples/sample_3/groundtruths'
        self.txb_gt_dir.setText(directory)
        self.dir_annotations_gt = directory
        directory = '/home/rafael/thesis/Object-Detection-Metrics/samples/sample_3/images'
        self.txb_gt_images_dir.setText(directory)
        self.dir_images_gt = directory

    def btn_gt_dir_clicked(self):
        directory = QFileDialog.getExistingDirectory(
            self, 'Choose directory with ground truth annotations', os.getcwd())
        if directory != '':
            self.txb_gt_dir.setText(directory)
            self.dir_annotations_gt = directory

    def btn_gt_images_dir_clicked(self):
        directory = QFileDialog.getExistingDirectory(self,
                                                     'Choose directory with ground truth images',
                                                     os.getcwd())
        if directory != '':
            self.txb_gt_images_dir.setText(directory)
            self.dir_images_gt = directory

    def btn_det_dir_clicked(self):
        directory = QFileDialog.getExistingDirectory(self, 'Choose directory with detections',
                                                     os.getcwd())
        if directory != '':
            self.txb_det_dir.setText(directory)
            self.dir_annotations_det = directory

    def get_annotations(self, type_bb):
        if type_bb == BBType.GROUND_TRUTH:
            name_bb = 'ground-truth'
            directory = self.dir_annotations_gt
            rad_type_absolute = self.rad_gt_type_absolute
            rad_type_relative = self.rad_gt_type_relative
            rad_format_xyx2y2 = self.rad_gt_format_xyx2y2
            rad_format_xywh = self.rad_gt_format_xywh
            rad_format_pascal = self.rad_gt_format_pascal
        if type_bb == BBType.DETECTED:
            name_bb = 'detection'
            directory = self.dir_annotations_det
            rad_type_absolute = self.rad_det_type_absolute
            rad_type_relative = self.rad_det_type_relative
            rad_format_xyx2y2 = self.rad_det_format_xyx2y2
            rad_format_xywh = self.rad_det_format_xywh
            rad_format_pascal = self.rad_det_format_pascal

        if directory == '':
            return True, BoundingBoxes()

        if not os.path.isdir(directory):
            return False, f'Directory with {name_bb} bounding boxes is not valid.'
        if rad_type_absolute.isChecked():
            type_coords = CoordinatesType.ABSOLUTE
        elif rad_type_relative.isChecked():
            type_coords = CoordinatesType.RELATIVE

        # Format <class_name> <left> <top> <right> <bottom>
        if rad_format_xyx2y2.isChecked():
            res, annotations = Datasets_Parser.get_annotations_xyx2y2(
                directory=directory,
                type_coordinates=type_coords,
                bb_type=type_bb,
                img_directory=self.dir_images_gt)
        # Format <class_name> <left> <top> <width> <height>
        elif rad_format_xywh.isChecked():
            a = 123
        # Format PASCAL VOC format (XML)
        elif rad_format_pascal.isChecked():
            res, annotations = Datasets_Parser.get_annotations_PASCAL_VOC(directory)
        return res, annotations

    def call_dialog_statistics(self, bb_type):
        ret_gt, annotations_gt = self.get_annotations(BBType.GROUND_TRUTH)
        ret_det, annotations_det = self.get_annotations(BBType.DETECTED)
        if ret_gt is False or ret_det is False:
            pass  # Mostrar message box
        if self.dir_images_gt != '':
            if not os.path.isdir(self.dir_images_gt):
                # messagebox com erro
                return
        self.dialog_statistics.show_dialog(bb_type, annotations_gt, annotations_det,
                                           self.dir_images_gt)

    def btn_statistics_gt_clicked(self):
        self.call_dialog_statistics(BBType.GROUND_TRUTH)

    def btn_statistics_det_clicked(self):
        self.call_dialog_statistics(BBType.DETECTED)
