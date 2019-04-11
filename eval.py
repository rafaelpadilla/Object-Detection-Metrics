"""
Copyright (C) 2018, AIMLedge Pte, Ltd.
All rights reserved.

"""
import argparse
import glob
import os
import shutil
import sys
import tqdm
from external.object_detection_metrics.lib.BoundingBox import BoundingBox
from external.object_detection_metrics.lib.BoundingBoxes import BoundingBoxes
from external.object_detection_metrics.lib.Evaluator import *
from external.object_detection_metrics.lib.utils import BBFormat
import xml.etree.ElementTree as ET


def get_bounding_box_format(format_string):
  """

  :param format_string:
  :return:
  """
  if format_string == 'xywh':
    return BBFormat.XYWH
  elif format_string == 'xyrb':
    return BBFormat.XYX2Y2
  else:
    raise ValueError('Invalid bounding box format string')


def get_box_coord_type(type_str):
  """

  :param type_str:
  :return:
  """
  if type_str == 'abs':
    return CoordinatesType.Absolute
  elif type_str == 'rel':
    return CoordinatesType.Relative
  else:
    raise ValueError('Invalid coordinate type string')


def file_to_list(input_file):
  """

  :param input_file: Creates a list from lines in a file
  :return:
  """
  lines = []
  with open(input_file, 'r') as f:
    for line in f:
      lines.append(line.strip())
  return lines


def get_bounding_boxes_from_xml_file(xml_path, coord_type, box_format,
                                     all_boxes, all_classes, is_ground_truth):
  assert coord_type == CoordinatesType.Absolute
  assert box_format == BBFormat.XYX2Y2
  assert is_ground_truth is True

  tree = ET.parse(xml_path)
  root = tree.getroot()
  size = root.find('size')
  w = int(size.find('width').text)
  h = int(size.find('height').text)
  image_name = os.path.basename(xml_path).replace('.xml', '')

  for obj in root.iter('object'):
    difficult = obj.find('difficult').text
    cls_name = obj.find('name').text
    if int(difficult) == 1:
      continue
    xmlbox = obj.find('bndbox')
    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
         float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
    bb = BoundingBox(
      image_name,
      cls_name,
      float(xmlbox.find('xmin').text),
      float(xmlbox.find('ymin').text),
      float(xmlbox.find('xmax').text),
      float(xmlbox.find('ymax').text),
      coord_type,
      (w, h),
      BBType.GroundTruth,
      format=box_format)
    all_boxes.addBoundingBox(bb)
    if cls_name not in all_classes:
      all_classes.append(cls_name)


def get_bounding_boxes_from_txt_file(txt_path, coord_type, box_format,
                                     all_boxes, all_classes, is_ground_truth,
                                     image_size=(0, 0)):
  """

  :param txt_path:
  :param coord_type:
  :param box_format:
  :param all_boxes:
  :param all_classes:
  :param is_ground_truth:
  :return:
  """
  assert coord_type == CoordinatesType.Absolute
  image_name = os.path.basename(txt_path).split('.')[0]

  with open(txt_path, 'r') as f:
    for line in f:
      line = line.replace("\n", "")
      if line.replace(' ', '') == '':
        continue
      split_line = line.split(" ")
      if is_ground_truth:
        class_name = (split_line[0])
        x = float(split_line[1])
        y = float(split_line[2])
        w = float(split_line[3])
        h = float(split_line[4])
        bb = BoundingBox(image_name, class_name, x, y, w, h, coord_type,
          image_size, BBType.GroundTruth, format=box_format)
      else:
        class_name = (split_line[0])  # class
        confidence = float(split_line[1])
        x = float(split_line[2])
        y = float(split_line[3])
        w = float(split_line[4])
        h = float(split_line[5])
        bb = BoundingBox(image_name, class_name, x, y, w, h, coord_type,
                         image_size, BBType.Detected, confidence,
                         format=box_format)
      all_boxes.addBoundingBox(bb)
      if class_name not in all_classes:
        all_classes.append(class_name)


def create_bounding_boxes(label_dir_or_filelist, is_ground_truth, box_format,
                          coord_type, all_boxes=None, all_classes=None,
                          image_size=(0, 0), is_xml=False):
  if all_classes is None:
    all_classes = []
  if all_boxes is None:
    all_boxes = BoundingBoxes()
  if os.path.isfile(label_dir_or_filelist):
    label_files = file_to_list(label_dir_or_filelist)
  else:
    os.chdir(label_dir_or_filelist)
    if is_xml:
      label_files = glob.glob('*.xml')
    else:
      label_files = glob.glob('*.txt')
  label_files.sort()
  print('Generating boxes from {} files'.format(len(label_files)))
  for file_idx in tqdm.tqdm(range(len(label_files))):
    if is_xml:
      get_bounding_boxes_from_xml_file(label_files[file_idx], coord_type,
                                       box_format, all_boxes, all_classes,
                                       is_ground_truth)
    else:
      get_bounding_boxes_from_txt_file(label_files[file_idx],
                                       coord_type, box_format, all_boxes,
                                       all_classes, is_ground_truth,
                                       image_size
                                       )
  return all_boxes, all_classes


def main(args):
  iou_threshold = args.threshold

  gt_box_format = get_bounding_box_format(args.gt_format)
  gt_box_type = get_box_coord_type(args.gt_coords)
  is_gt_xml = True if args.gt_file_format == 'xml' else False

  det_box_format = get_bounding_box_format(args.det_format)
  det_box_type = get_box_coord_type(args.det_coords)
  is_det_xml = True if args.det_file_format == 'xml' else False

  # Create directory to save results
  shutil.rmtree(args.save_path, ignore_errors=True)  # Clear folder
  os.makedirs(args.save_path)

  # Get groundtruth boxes
  all_boxes, all_classes = create_bounding_boxes(args.gt_dir, True,
                                                 gt_box_format, gt_box_type,
                                                 None, None, is_xml=is_gt_xml)
  print(all_boxes.count())
  all_boxes, all_classes = create_bounding_boxes(args.det_dir, False,
                                                 det_box_format, det_box_type,
                                                 all_boxes, all_classes,
                                                 is_xml=is_det_xml)
  print(all_boxes.count(BBType.Detected))
  all_classes.sort()
  evaluator = Evaluator()
  acc_AP = 0
  valid_classes = 0

  # Plot Precision x Recall curve
  detections = evaluator.PlotPrecisionRecallCurve(
    all_boxes,
    IOUThreshold=iou_threshold,
    method=MethodAveragePrecision.EveryPointInterpolation,
    showAP=True,
    showInterpolatedPrecision=False,
    # Don't plot the interpolated precision curve
    savePath=args.save_path,
    showGraphic=args.show_plot)

  f = open(os.path.join(args.save_path, 'results.txt'), 'w')
  f.write('Object Detection Metrics\n')
  f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
  f.write('Average Precision (AP), Precision and Recall per class:')

  # each detection is a class
  for metrics_per_class in detections:

    # Get metric values per each class
    cl = metrics_per_class['class']
    ap = metrics_per_class['AP']
    precision = metrics_per_class['precision']
    recall = metrics_per_class['recall']
    totalPositives = metrics_per_class['total positives']
    total_TP = metrics_per_class['total TP']
    total_FP = metrics_per_class['total FP']

    if totalPositives > 0:
      valid_classes = valid_classes + 1
      acc_AP = acc_AP + ap
      prec = ['%.2f' % p for p in precision]
      rec = ['%.2f' % r for r in recall]
      ap_str = "{0:.2f}%".format(ap * 100)
      # ap_str = "{0:.4f}%".format(ap * 100)
      print('AP: %s (%s)' % (ap_str, cl))
      f.write('\n\nClass: %s' % cl)
      f.write('\nAP: %s' % ap_str)
      f.write('\nPrecision: %s' % prec)
      f.write('\nRecall: %s' % rec)

  mAP = acc_AP / valid_classes
  mAP_str = "{0:.2f}%".format(mAP * 100)
  print('mAP: %s' % mAP_str)
  f.write('\n\n\nmAP: %s' % mAP_str)


if __name__ == '__main__':

  # Get current path to set default folders
  current_path = os.path.dirname(os.path.abspath(__file__))
  default_save_dir = os.path.join(current_path, 'results')

  parser = argparse.ArgumentParser(
    description='Object Detection Metrics - Pascal VOC')

  parser.add_argument('-gt', '--gt_dir', dest='gt_dir', required=True,
                      help='Directory containing ground truth box files or a'
                           'file containing absolute paths of ground truth'
                           'annotation files.')
  parser.add_argument('-det', '--det_dir', dest='det_dir', required=True,
                      help='Directory containing detection files or a'
                           'file containing absolute paths of detection files.')
  parser.add_argument('-t', '--threshold', dest='threshold', type=float,
                      default=0.5, help='IOU threshold to match the '
                                        'detections.')

  parser.add_argument('-gt_format', dest='gt_format', default='xywh',
                      help='Format of the coordinates of the ground truth'
                           'bounding boxes: '
                           '(\'xywh\': <left> <top> <width> <height>)'
                           ' or (\'xyrb\': <left> <top> <right> <bottom>)')
  parser.add_argument('--gt_file_format', dest='gt_file_format', type=str,
                      default='txt', help='Ground truth annotation file format')
  parser.add_argument('-det_format', dest='det_format', default='xywh',
                      help='format of the coordinates of the detected bounding'
                           'boxes (\'xywh\': <left> <top> <width> <height>) '
                           'or (\'xyrb\': <left> <top> <right> <bottom>)')

  parser.add_argument('--det_file_format', dest='det_file_format', type=str,
                      default='txt', help='Detection file format')
  parser.add_argument('-gt_coords', dest='gt_coords', default='abs',
                      help='reference of the ground truth bounding box'
                           'coordinates: absolute values (\'abs\') or relative'
                           'to its image size (\'rel\')')
  parser.add_argument('-det_coords', default='abs', dest='det_coords',
                      help='reference of the ground truth bounding box '
                           'coordinates: absolute values (\'abs\') or relative'
                           'to its image size (\'rel\')')
  parser.add_argument('-sp', '--save_path', dest='save_path',
                      default=default_save_dir,
                      help='folder where the plots are saved')
  parser.add_argument('-np', '--noplot', dest='show_plot', action='store_false',
                      help='no plot is shown during execution')
  args = parser.parse_args()
  if args.det_coords == 'rel' or args.gt_coords == 'rel':
    print('Relative coordinates are not supported')
    sys.exit(-1)
  main(args)


