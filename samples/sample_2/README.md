# Sample 2

This sample was created for those who want to understand more about the metric functions of this project. If you just want to evaluate your detections dealing with a high level interface, just check the instructions [here](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md#how-to-use-this-project).

### Evaluation Metrics

First we need to represent each bounding box with the class `BoundingBox`. The function `getBoundingBoxes` reads .txt files containing the coordinates of the [detected](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2/detections) and [ground truth](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2/groundtruths) bounding boxes:

```python
def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath,'groundtruths')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt","")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n","")
            if line.replace(' ','') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0] #class
            x = float(splitLine[1]) #confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(nameOfImage,idClass,x,y,w,h,CoordinatesType.Absolute, (200,200), BBType.GroundTruth, format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath,'detections')
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents the confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt","")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n","")
            if line.replace(' ','') == '':
                continue            
            splitLine = line.split(" ")
            idClass = splitLine[0] #class
            confidence = float(splitLine[1]) #confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(nameOfImage, idClass,x,y,w,h,CoordinatesType.Absolute, (200,200), BBType.Detected, confidence, format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes
```

Note that the text files contain one bounding box per line in the format:

 **\<class of the object> \<left> \<top> \<width> \<height>**: For ground truth files.
  
  **\<class of the object> \<confidence> \<left> \<top> \<width> \<height>**: For detection files.

#### Pascal VOC

The example below shows how to evaluate object detections using Pascal VOC metrics creating **manually** ground truth and detected bounding box coordinates.  

First, you need to import the `Evaluator` package and create the object `Evaluator()`:

```python
from Evaluator import *

# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()
```
Don't forget to put the content of the folder `\lib` in the same folder of your code. You could also put it in a different folder and add it in your project as done by the `_init_paths.py` [file](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/_init_paths.py) in the sample code.

With the ```evaluator``` object, you will have access to methods that retrieve the metrics:

| Method | Description | Parameters | Returns |
|------|-----------|----------|-------|
|  GetPascalVOCMetrics | Get the metrics used by the VOC Pascal 2012 challenge | `boundingboxes`: Object of the class `BoundingBoxes` representing ground truth and detected bounding boxes; `IOUThreshold`: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5); | List of dictionaries. Each dictionary contains information and metrics of each class. The keys of each dictionary are:  `dict['class']`: class representing the current dictionary; `dict['precision']`: array with the precision values; `dict['recall']`: array with the recall values; `dict['AP']`: **average precision**; `dict['interpolated precision']`: interpolated precision values; `dict['interpolated recall']`: interpolated recall values; `dict['total positives']`: total number of ground truth positives; `dict['total TP']`: total number of True Positive detections; `dict['total FP']`: total number of False Negative detections; |
PlotPrecisionRecallCurve |	Plot the Precision x Recall curve for a given class | `classId`: The class that will be plot; `boundingBoxes`: Object of the class `BoundingBoxes` representing ground truth and detected bounding boxes; `IOUThreshold`: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5); `showAP`: if True, the average precision value will be shown in the title of the graph (default = False); `showInterpolatedPrecision`: if True, it will show in the plot the interpolated precision (default = False); `savePath`: if informed, the plot will be saved as an image in this path (ex: `/home/mywork/ap.png`) (default = None); `showGraphic`: if True, the plot will be shown (default = True) | The dictionary containing information and metric about the class. The keys of the dictionary are: `dict['class']`: class representing the current dictionary; `dict['precision']`: array with the precision values; `dict['recall']`: array with the recall values; `dict['AP']`: **average precision**; `dict['interpolated precision']`: interpolated precision values; `dict['interpolated recall']`: interpolated recall values; `dict['total positives']`: total number of ground truth positives; `dict['total TP']`: total number of True Positive detections; `dict['total FP']`: total number of False Negative detections |

All methods that retreive metrics need you to inform the bounding boxes (ground truth and detected). Those bounding boxes are represented by an object of the class `BoundingBoxes` . Each bounding box is defined by the class `BoundingBox`. **The snippet below shows the creation of the bounding boxes** of two images (img_0001 and img_0002). In this example there are 6 ground truth bounding boxes (4 belonging to img_0001 and 2 belonging to img_0002) and 3 detections (2 belonging to img_0001 and 2 belonging to img_0002). Img_0001 ground truths contain bounding boxes of 3 classes (classes 0, 1 and 2). Img_0002 ground truths contain bounding boxes of 2 classes (classes 0 and 1):

```python
# Defining bounding boxes
# Ground truth bounding boxes of img_0001.jpg
gt_boundingBox_1 = BoundingBox(imageName='img_0001', idClass=0, 25, 16, 38, 56, bbType=BBType.GroundTruth, format=BBFormat.XYWH)
gt_boundingBox_2 = BoundingBox(imageName='img_0001', idClass=0, 129, 123, 41, 62, bbType=BBType.GroundTruth, format=BBFormat.XYWH)
gt_boundingBox_3 = BoundingBox(imageName='img_0001', idClass=1, 30, 48, 40, 38, bbType=BBType.GroundTruth, format=BBFormat.XYWH)
gt_boundingBox_4 = BoundingBox(imageName='img_0001', idClass=2, 15, 10, 56, 70, bbType=BBType.GroundTruth, format=BBFormat.XYWH)
# Ground truth bounding boxes of img_0002.jpg
gt_boundingBox_5 = BoundingBox(imageName='img_0002', idClass=0, 25, 16, 38, 56, bbType=BBType.GroundTruth, format=BBFormat.XYWH)
gt_boundingBox_8 = BoundingBox(imageName='img_0002', idClass=1, 15, 10, 56, 70, bbType=BBType.GroundTruth, format=BBFormat.XYWH)
# Detected bounding boxes of img_0001.jpg
detected_boundingBox_1 = BoundingBox(imageName='img_0001', idClass=0, 90, 78, 101, 58, bbType=BBType.Detected, format=BBFormat.XYWH)
detected_boundingBox_2 = BoundingBox(imageName='img_0001', idClass=1, 85, 17, 49, 60, bbType=BBType.Detected, format=BBFormat.XYWH)
# Detected bounding boxes of img_0002.jpg
detected_boundingBox_3 = BoundingBox(imageName='img_0002', idClass=1, 27, 18, 45, 60, bbType=BBType.Detected, format=BBFormat.XYWH)

# Creating the object of the class BoundingBoxes 
myBoundingBoxes = BoundingBoxes()
# Add all bounding boxes to the BoundingBoxes object:
myBoundingBoxes.add(gt_boundingBox_1)
myBoundingBoxes.add(gt_boundingBox_2)
myBoundingBoxes.add(gt_boundingBox_3)
myBoundingBoxes.add(gt_boundingBox_4)
myBoundingBoxes.add(gt_boundingBox_5)
myBoundingBoxes.add(gt_boundingBox_6)
myBoundingBoxes.add(detected_boundingBox_1)
myBoundingBoxes.add(detected_boundingBox_2)
myBoundingBoxes.add(detected_boundingBox_3)
```

Some important points:  

* Create your bounding boxes using the constructor of the `BoundingBox` class. The 3rd and 4th parameters represent the most top-left x and y coordinates  of the bounding box. The 5th and 6th parameters can be either the most bottom-right x and y coordinates of the bounding box or the width and height of the bounding box. If your bounding box is identified as x1, y1, x2, y2 coordinates, you need to pass `format=BBFormat.XYX2Y2`. If you want to identify it as x, y, width, height, you need to pass `format=BBFormat.XYWH`.
* Use the tag `bbType=BBType.GroundTruth` to identify your bounding box as being ground truth. If it is a detection, use `bbType=BBType.Detected`.
* Be consistent with the `imageName` parameter. For example: bounding boxes with `imageName='img_0001'` and `imageName='img0001'` are from two different images. 
* The code is all commented. [Here](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/lib/BoundingBox.py#L4) you can see all parameters needed by the constructor of the `BoundingBox` class.

**Of course you won't build your bounding boxes one by one as done in this example.** You should read your detections within a loop and create your bounding boxes inside of it. [sample_1.py](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_1.py) reads detections from 2 different folders, one containing .txt files with ground truths and the other containing .txt files with detections. Check this [sample code](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_1.py) as a reference.
