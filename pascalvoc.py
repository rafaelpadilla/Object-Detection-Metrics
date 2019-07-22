# TODO:
# Ao passar comando python pascalvoc.py -gtformat xyrb dá um erro!

###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #
# Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)                               #
#        SMT - Signal Multimedia and Telecommunications Lab                               #
#        COPPE - Universidade Federal do Rio de Janeiro                                   #
#        Last modification: June 19th 2019                                                #
###########################################################################################

import argparse
import glob
import os
import shutil
import sys

import _init_paths
from bounding_box import BoundingBox
from bounding_boxes import BoundingBoxes
from evaluator import Evaluator
from utils import BBFormat, BBType, CoordinatesType, MethodAveragePrecision


def validate_formats(arg_format, arg_name, errors):
    """ Verify if string format that represents the bounding box format is valid.

        Parameters
        ----------
        arg_format : str
            Received argument with the format to be validated.
        arg_name : str
            Argument name that represents the bounding box format.
        errors : list
            List with error messages to be appended with error message in case an error occurs.

        Returns
        -------
        BBFormat : Enum
            If arg_format is valid, it will return the enum representing the correct format. If
            format is not valid, return None.
    """

    if arg_format == 'xywh':
        return BBFormat.XYWH
    elif arg_format == 'xyrb':
        return BBFormat.XYX2Y2
    elif arg_format is None:
        return BBFormat.XYWH  # default when nothing is passed
    else:
        errors.append(f'argument {arg_name}: invalid value. It must be either \'xywh\' or \'xyrb\'')
        return None


def validate_mandatory_args(arg, arg_name, errors):
    """ Verify if a given mandatory argument is present.

        Parameters
        ----------
        arg : str
            Received argument to be validated.
        arg_name : str
            Argument name that represents the bounding box format.
        errors : list
            List with error messages to be appended with error message in case an error occurs.

        Returns
        -------
        bool
            True if argument is valid, False otherwise.
    """

    if arg is None:
        errors.append(f'argument {arg_name}: required argument')
    else:
        return True


def validate_image_size(arg, arg_name, arg_informed, errors):
    """ Verify if the argument representing the image size is valid.

        Parameters
        ----------
        arg : str
            Received image size argument to be validated.
        arg_informed : str
            Necessary argument names that represents the coordinates (-gtCoordinates or
            -detCoordinates).
        errors : list
            List with error messages to be appended with error message in case an error occurs.

        Returns
        -------
        tuple or None
            If valid, a tuple (width, height). None if not valid.
    """
    errorMsg = f'argument {arg_name}: required argument if {arg_informed} is relative'
    ret = None
    if arg is None:
        errors.append(errorMsg)
    else:
        arg = arg.replace('(', '').replace(')', '')
        args = arg.split(',')
        if len(args) != 2:
            errors.append(
                f'{errorMsg}. It must be in the format \'width,height\' (e.g. \'600,400\')')
        else:
            if not args[0].isdigit() or not args[1].isdigit():
                errors.append(
                    f'{errorMsg}. It must be in INdiaTEGER the format \'width,height\' (e.g. \'600,400\')'
                )
            else:
                ret = (int(args[0]), int(args[1]))
    return ret


def validate_coordinates_types(arg, arg_name, errors):
    """ Verify if the argument representing the type of a bounding box coordinate is valid.
    Use \'rel\' if the annotated coordinates are relative to the image size (as used in YOLO).
    Use \'abs\' if the coordinates are represented in absolute values.

        Parameters
        ----------
        arg : str
            Received coordinate type to be validated.
        arg_name : str
            Argument name that represents the bounding box format.
        errors : list
            List with error messages to be appended with error message in case an error occurs.

        Returns
        -------
        CoordinatesType : Enum
            If arg is valid, it will return the enum representing the correct format
            (CoordinatesType.Absolute or CoordinatesType.Relative). If format is not valid,
            return None. If nothing is passed, default is CoordinatesType.Absolute.
    """
    if arg == 'abs':
        return CoordinatesType.ABSOLUTE
    elif arg == 'rel':
        return CoordinatesType.RELATIVE
    elif arg is None:
        return CoordinatesType.ABSOLUTE  # default when nothing is passed
    errors.append('argument %s: invalid value. It must be either \'rel\' or \'abs\'' % arg_name)
    return None


def validate_paths(arg, arg_name, errors):
    """ Verify if the argument representing a path is valid.

        Parameters
        ----------
        arg : str
            Received path to be validated.
        arg_name : str
            Argument name that represents the path.
        errors : list
            List with error messages to be appended with error message in case an error occurs.

        Returns
        -------
        str
            If valid, it returns absolute path of the path. If not valid, it returns None.
    """

    if arg is None:
        errors.append(f'argument {arg_name}: invalid directory')
        return None
    elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(current_path, arg)) is False:
        errors.append(f'argument {arg_name}: directory does not exist \'{arg}\'')
        return None
    elif os.path.isdir(arg) is True:
        arg = os.path.join(current_path, arg)
        return arg
    else:
        return None


def read_bounding_boxes(directory,
                        is_GT,
                        bb_format,
                        coord_type,
                        all_bounding_boxes=None,
                        all_classes=None,
                        img_size=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections).

        Parameters
        ----------
        directory : str
            Directory containing the files with the bounding boxes coordinates.
        is_GT : bool
            True if files represent ground-truth bounding boxes. False if they are detected
            bounding boxes.
        bb_format : enum
            Enum BBFormat(BBFormat.XYWH or BBFormat.XYX2Y2) representing the format of the bounding
            boxes coordinates.
        coord_type : enum
            Type of the bounding box coordinates (CoordinatesType.Absolute or
            CoordinatesType.Relative)
        all_bounding_boxes : BoundingBoxes
            BoundingBoxes object containing bounding boxes to be appended. If you want to create a
            new bounding box group, set it to None. Default is None.
        all_classes : list
            List containing all classes of objects represented by the bounding boxes. You can pass
            an existing list to be appended with new classes or set it to None to start a new list.
        img_size : tuple
            Tuple representing width and height of the image size. Use the format (width, height).
            If coord_type is CoordinatesType.Relative, img_size is required.

        Returns
        -------
        BoundingBoxes
            Object containing all bounding boxes found in the directory.
        all_classes
            List containing all classes of objects represented by the bounding boxes.
    """
    if all_bounding_boxes is None:
        all_bounding_boxes = BoundingBoxes()
    if all_classes is None:
        all_classes = []
    # Read ground truths
    os.chdir(directory)
    files = glob.glob("*.txt")
    files.sort()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        image_name = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            split_line = line.split(" ")
            if is_GT:
                # class_id = int(split_line[0]) #class
                class_id = (split_line[0])  # class
                x = float(split_line[1])
                y = float(split_line[2])
                w = float(split_line[3])
                h = float(split_line[4])
                bb = BoundingBox(image_name,
                                 class_id,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coord_type,
                                 img_size,
                                 BBType.GROUND_TRUTH,
                                 format=bb_format)
            else:
                # class_id = int(split_line[0]) #class
                class_id = (split_line[0])  # class
                confidence = float(split_line[1])
                x = float(split_line[2])
                y = float(split_line[3])
                w = float(split_line[4])
                h = float(split_line[5])
                bb = BoundingBox(image_name,
                                 class_id,
                                 x,
                                 y,
                                 w,
                                 h,
                                 coord_type,
                                 img_size,
                                 BBType.DETECTED,
                                 confidence,
                                 format=bb_format)
            all_bounding_boxes.add_bounding_box(bb)
            if class_id not in all_classes:
                all_classes.append(class_id)
        fh1.close()
    return all_bounding_boxes, all_classes


# Get current path to set default folders
current_path = os.path.dirname(os.path.abspath(__file__))

VERSION = '0.1 (beta)'

parser = argparse.ArgumentParser(
    prog='Object Detection Metrics - Pascal VOC',
    description='This project applies the most popular metrics used to evaluate object detection '
    'algorithms.\nThe current implemention runs the Pascal VOC metrics.\nFor further references, '
    'please check:\nhttps://github.com/rafaelpadilla/Object-Detection-Metrics',
    epilog="Developed by: Rafael Padilla (rafael.padilla@smt.ufrj.br)")
# formatter_class=RawTextHelpFormatter)
parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + VERSION)
# Positional arguments: Mandatory
parser.add_argument('-gt',
                    '--gtfolder',
                    dest='gt_folder',
                    default=os.path.join(current_path, 'groundtruths'),
                    metavar='',
                    help='folder containing your ground truth bounding boxes')
parser.add_argument('-det',
                    '--detfolder',
                    dest='det_folder',
                    default=os.path.join(current_path, 'detections'),
                    metavar='',
                    help='folder containing your detected bounding boxes')
# Positional arguments:  Optional
parser.add_argument('-t',
                    '--threshold',
                    dest='iou_threshold',
                    type=float,
                    default=0.5,
                    metavar='',
                    help='IOU threshold. Default 0.5')
parser.add_argument('-gtformat',
                    dest='gt_format',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the ground truth bounding boxes: '
                    '(\'xywh\': <left> <top> <width> <height>)'
                    ' or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument('-detformat',
                    dest='det_format',
                    metavar='',
                    default='xywh',
                    help='format of the coordinates of the detected bounding boxes '
                    '(\'xywh\': <left> <top> <width> <height>) '
                    'or (\'xyrb\': <left> <top> <right> <bottom>)')
parser.add_argument('-gtcoords',
                    dest='gt_coordinates',
                    default='abs',
                    metavar='',
                    help='reference of the ground truth bounding box coordinates: absolute '
                    'values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument('-detcoords',
                    default='abs',
                    dest='det_coordinates',
                    metavar='',
                    help='reference of the ground truth bounding box coordinates: '
                    'absolute values (\'abs\') or relative to its image size (\'rel\')')
parser.add_argument('-img_size',
                    dest='img_size',
                    metavar='',
                    help='image size. Required if -gtcoords or -detcoords are \'rel\'')
parser.add_argument('-sp',
                    '--savepath',
                    dest='save_path',
                    default=os.path.join(current_path, 'results'),
                    metavar='',
                    help='folder where the plots are saved')
parser.add_argument('-np',
                    '--noplot',
                    dest='show_plot',
                    action='store_false',
                    help='no plot is shown during execution')
args = parser.parse_args()

iou_threshold = args.iou_threshold

# AQUI TODO
args.gt_format = 'xyrb'
print(args.gt_format)

# Arguments validation
errors = []
# Validate formats
gt_format = validate_formats(args.gt_format, '-gtformat', errors)
det_format = validate_formats(args.det_format, '-detformat', errors)
# Groundtruth folder
if validate_mandatory_args(args.gt_folder, '-gt/--gtfolder', errors):
    gt_folder = validate_paths(args.gt_folder, '-gt/--gtfolder', errors)
else:
    gt_folder = os.path.join(current_path, 'groundtruths')
    if os.path.isdir(gt_folder) is False:
        errors.append('folder %s not found' % gt_folder)
# Coordinates types
gt_coord_type = validate_coordinates_types(args.gt_coordinates, '-gtCoordinates', errors)
det_coord_type = validate_coordinates_types(args.det_coordinates, '-detCoordinates', errors)
img_size = None
if gt_coord_type == CoordinatesType.RELATIVE:  # Image size is required
    img_size = validate_image_size(args.img_size, '-img_size', '-gtCoordinates', errors)
if det_coord_type == CoordinatesType.RELATIVE:  # Image size is required
    img_size = validate_image_size(args.img_size, '-img_size', '-detCoordinates', errors)
# Detection folder
if validate_mandatory_args(args.det_folder, '-det/--detfolder', errors):
    det_folder = validate_paths(args.det_folder, '-det/--detfolder', errors)
else:
    det_folder = os.path.join(current_path, 'detections')
    if os.path.isdir(det_folder) is False:
        errors.append('folder %s not found' % det_folder)
if args.save_path is not None:
    save_path = validate_paths(args.save_path, '-sp/--savepath', errors)
# Validate save_path
# If error, show error messages
if len(errors) != 0:
    print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                [-detformat] [-save]""")
    print('Object Detection Metrics: error(s): ')
    [print(e) for e in errors]
    sys.exit()

# Create directory to save results
shutil.rmtree(save_path, ignore_errors=True)  # Clear folder
os.makedirs(save_path)
# Show plot during execution
show_plot = args.show_plot

# Uncomment the lines below to print out the parameters
# print('iou_threshold= %f' % iou_threshold)
# print('save_path = %s' % save_path)
# print('gt_format = %s' % gt_format)
# print('det_format = %s' % det_format)
# print('gt_folder = %s' % gt_folder)
# print('det_folder = %s' % det_folder)
# print('gt_coord_type = %s' % gt_coord_type)
# print('det_coord_type = %s' % det_coord_type)
# print('show_plot %s' % show_plot)

# Get groundtruth boxes
all_bounding_boxes, all_classes = read_bounding_boxes(gt_folder,
                                                      True,
                                                      gt_format,
                                                      gt_coord_type,
                                                      img_size=img_size)
# Get detected boxes
all_bounding_boxes, all_classes = read_bounding_boxes(det_folder,
                                                      False,
                                                      det_format,
                                                      det_coord_type,
                                                      all_bounding_boxes,
                                                      all_classes,
                                                      img_size=img_size)
all_classes.sort()

evaluator = Evaluator()
acc_AP = 0
count_validated_classes = 0

# Plot Precision x Recall curve
detections = evaluator.plot_precision_recall_curve(
    all_bounding_boxes,  # Object containing all bounding boxes (ground truths and detections)
    IOU_threshold=iou_threshold,  # IOU threshold
    method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
    show_AP=True,  # Show Average Precision in the title of the plot
    show_interpolated_precision=False,  # Don't plot the interpolated precision curve
    save_path=save_path,
    show_graphic=show_plot)

f = open(os.path.join(save_path, 'results.txt'), 'w')
f.write('Object Detection Metrics\n')
f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
f.write('Average Precision (AP), Precision and Recall per class:')

# Each detection is a class
for metrics_per_class in detections:

    # Get metric values per each class
    cl = metrics_per_class['class']
    ap = metrics_per_class['AP']
    precision = metrics_per_class['precision']
    recall = metrics_per_class['recall']
    total_positives = metrics_per_class['total positives']
    total_TP = metrics_per_class['total TP']
    total_FP = metrics_per_class['total FP']

    if total_positives > 0:
        count_validated_classes += 1
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

mAP = acc_AP / count_validated_classes
mAP_str = "{0:.2f}%".format(mAP * 100)
print('mAP: %s' % mAP_str)
f.write('\n\n\nmAP: %s' % mAP_str)
