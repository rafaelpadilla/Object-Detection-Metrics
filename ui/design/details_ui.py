# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'details_ui.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(892, 442)
        self.lbl_sample_image = QtWidgets.QLabel(Dialog)
        self.lbl_sample_image.setGeometry(QtCore.QRect(520, 30, 361, 341))
        self.lbl_sample_image.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_sample_image.setTextFormat(QtCore.Qt.AutoText)
        self.lbl_sample_image.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_sample_image.setObjectName("lbl_sample_image")
        self.btn_random_image = QtWidgets.QPushButton(Dialog)
        self.btn_random_image.setGeometry(QtCore.QRect(520, 410, 211, 27))
        self.btn_random_image.setObjectName("btn_random_image")
        self.txb_statistics = QtWidgets.QTextEdit(Dialog)
        self.txb_statistics.setGeometry(QtCore.QRect(10, 30, 501, 371))
        self.txb_statistics.setReadOnly(True)
        self.txb_statistics.setObjectName("txb_statistics")
        self.lbl_groundtruth_dir_5 = QtWidgets.QLabel(Dialog)
        self.lbl_groundtruth_dir_5.setGeometry(QtCore.QRect(10, 10, 161, 17))
        self.lbl_groundtruth_dir_5.setObjectName("lbl_groundtruth_dir_5")
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 410, 231, 27))
        self.pushButton_3.setObjectName("pushButton_3")
        self.btn_save_image = QtWidgets.QPushButton(Dialog)
        self.btn_save_image.setGeometry(QtCore.QRect(740, 410, 141, 27))
        self.btn_save_image.setObjectName("btn_save_image")
        self.lbl_groundtruth_dir_6 = QtWidgets.QLabel(Dialog)
        self.lbl_groundtruth_dir_6.setGeometry(QtCore.QRect(520, 10, 161, 17))
        self.lbl_groundtruth_dir_6.setObjectName("lbl_groundtruth_dir_6")
        self.lbl_image_file_name = QtWidgets.QLabel(Dialog)
        self.lbl_image_file_name.setGeometry(QtCore.QRect(520, 380, 361, 20))
        self.lbl_image_file_name.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lbl_image_file_name.setObjectName("lbl_image_file_name")

        self.retranslateUi(Dialog)
        self.pushButton_3.clicked.connect(Dialog.btn_plot_bb_per_classes_clicked)
        self.btn_random_image.clicked.connect(Dialog.btn_load_random_image_clicked)
        self.btn_save_image.clicked.connect(Dialog.btn_save_image_clicked)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Bounding-boxes statistics"))
        self.lbl_sample_image.setText(_translate("Dialog", "[image]"))
        self.btn_random_image.setToolTip(_translate("Dialog", "The configurations will be applied in a random ground truth image."))
        self.btn_random_image.setText(_translate("Dialog", "load another random image"))
        self.txb_statistics.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">* 53392 bounding boxes were found in 12345 images.</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">* In 33 images no bounding boxes were found.</p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">* The average area of the bounding boxes is 1212 pixels.</p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">* The amount of bounding boxes per class is:</p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">[car]: 123 bounding boxes in 23 images.</p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">[person]: 231 bounding boxes in 93 images.</p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">[dog]: 231 bounding boxes in 90 images.</p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">[cat]: 231 bounding boxes in 393 images.</p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">(ordenar classes por ordem alfabética)</p></body></html>"))
        self.lbl_groundtruth_dir_5.setText(_translate("Dialog", "Statistics:"))
        self.pushButton_3.setToolTip(_translate("Dialog", "The configurations will be applied in a random ground truth image."))
        self.pushButton_3.setText(_translate("Dialog", "plot bounding boxes per class"))
        self.btn_save_image.setToolTip(_translate("Dialog", "The configurations will be applied in a random ground truth image."))
        self.btn_save_image.setText(_translate("Dialog", "save"))
        self.lbl_groundtruth_dir_6.setText(_translate("Dialog", "Sample image:"))
        self.lbl_image_file_name.setText(_translate("Dialog", "no image to show"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

