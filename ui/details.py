import os
import random

import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QFileDialog, QMainWindow

import _init_paths
import lib.utils as utils
from design.details_ui import Ui_Dialog as Details_UI


class Details_Dialog(QMainWindow, Details_UI):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)
        # initialize variables
        self.dir_images = ''
        self.gt_annotations = None
        self.det_annotations = None
        self.text_statistics = '<b>#TYPE_BB#:</b><br>'
        self.text_statistics += '<br>* A total of <b>#TOTAL_BB#</b> bounding boxes were found in <b>#TOTAL_IMAGES#</b> images.'
        self.text_statistics += '<br>* The average area of the bounding boxes is <b>#AVERAGE_AREA_BB#</b> pixels.'
        self.text_statistics += '<br>* The amount of bounding boxes per class is:'
        self.text_statistics += '<br>#AMOUNT_BB_PER_CLASS#'
        self.lbl_sample_image.setScaledContents(True)

    def initialize_ui(self):
        # clear all information
        self.txb_statistics.setText('')
        self.lbl_sample_image.setText('')
        self.btn_random_image.setEnabled(False)
        self.btn_save_image.setEnabled(False)

        # Create text with ground truth statistics
        if self.type_bb == utils.BBType.GROUND_TRUTH:
            stats_gt = self.text_statistics.replace('#TYPE_BB#', 'Ground Truth')
            self.annot_obj = self.gt_annotations
        elif self.type_bb == utils.BBType.DETECTED:
            stats_gt = self.text_statistics.replace('#TYPE_BB#', 'Detections')
            self.annot_obj = self.det_annotations
        stats_gt = stats_gt.replace('#TOTAL_BB#', str(self.annot_obj.count()))
        stats_gt = stats_gt.replace('#TOTAL_IMAGES#', str(self.annot_obj.get_total_images()))
        stats_gt = stats_gt.replace('#AVERAGE_AREA_BB#', '%.2f' % self.annot_obj.get_average_area())
        # Get amount of bounding boxes per class
        self.bb_per_class = self.annot_obj.get_bounding_box_all_classes()
        amount_bb_per_class = 'No class found'
        if len(self.bb_per_class) > 0:
            amount_bb_per_class = ''
            longest_class_name = len(max(self.bb_per_class.keys(), key=len))
            for c, amount in self.bb_per_class.items():
                c = c.ljust(longest_class_name, ' ')
                amount_bb_per_class += f'   {c} : {amount}<br>'
        stats_gt = stats_gt.replace('#AMOUNT_BB_PER_CLASS#', amount_bb_per_class)
        self.txb_statistics.setText(stats_gt)

        # get random image file
        if os.path.isdir(self.dir_images):
            self.image_files = utils.get_files_dir(
                self.dir_images, extensions=['jpg', 'jpge', 'png', 'bmp', 'tiff', 'tif'])
            self.load_random_image()
        else:
            self.image_files = []

    def load_random_image(self):
        random_image_file = random.choice(self.image_files)
        self.lbl_image_file_name.setText(random_image_file)
        # Get all annotations anddetections from this file
        if self.gt_annotations is not None:
            random_gt_annotation_files = self.gt_annotations.get_bounding_boxes_by_image_name(
                utils.remove_file_extension(random_image_file))
        if self.det_annotations is not None:
            self.btn_random_image.setEnabled(True)
            self.btn_save_image.setEnabled(True)
            random_det_annotation_files = self.det_annotations.get_bounding_boxes_by_image_name(
                utils.remove_file_extension(random_image_file))
        # Load image and draw bounding boxes on it
        rand_image = cv2.imread(os.path.join(self.dir_images, random_image_file))
        rand_image = cv2.cvtColor(rand_image, cv2.COLOR_BGR2RGB)
        for bb in random_gt_annotation_files:
            rand_image = utils.add_bb_into_image(rand_image,
                                                 bb,
                                                 color=(0, 255, 0),
                                                 thickness=2,
                                                 label=bb.get_class_id())
        utils.show_image_in_qt_component(rand_image, self.lbl_sample_image)

    def show_dialog(self, type_bb, gt_annotations=None, det_annotations=None, dir_images=None):
        self.type_bb = type_bb
        self.gt_annotations = gt_annotations
        self.det_annotations = det_annotations
        self.dir_images = dir_images
        self.initialize_ui()
        self.show()

    def btn_plot_bb_per_classes_clicked(self):
        plt.close()
        plt.bar(self.bb_per_class.keys(), self.bb_per_class.values())
        plt.xlabel('classes')
        plt.ylabel('amount of bounding boxes')
        plt.xticks(rotation=45)
        plt.title('Bounding boxes per class')
        plt.show()

    def btn_load_random_image_clicked(self):
        self.load_random_image()

    def btn_save_image_clicked(self):
        pass
