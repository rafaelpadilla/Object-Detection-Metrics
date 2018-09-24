# Sample 2

This sample was created for those who want to understand more about the metric functions of this project. If you just want to evaluate your detections dealing with a high level interface, just check the instructions [here](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md#how-to-use-this-project).

In order to reproduce the results of this code **using the high level interface**, just navigate to the folder where the ```pascalvoc.py``` is and run the following command: ```python pascalvoc.py -t 0.3```

or if you want to be more complete: ```python pascalvoc.py -gt groundtruths/ -det detections/ -t 0.3```

or if you want to use [relative coordinates](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/detections_rel) (like retrieved results from YOLO), you need to inform that the detected coordinates are relative, specify the image size:
```python pascalvoc.py -gt groundtruths/ -det detections_rel/ -detcoords rel  -imgsize 200,200 -t 0.3```

Or if you want to play a little bit with this project, follow the steps below:

### Evaluation Metrics

First we need to represent each bounding box with the class `BoundingBox`. The function `getBoundingBoxes` reads .txt files containing the coordinates of the [detected](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2/detections) and [ground truth](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2/groundtruths) bounding boxes and creates a `BoundingBox` object for each of them. Then, it gathers all boxes in the `BoundingBoxes` object and returns it:

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
    
# Read txt files containing bounding boxes (ground truth and detections)
boundingboxes = getBoundingBoxes()
```

Note that the text files contain one bounding box per line in the format:

 **\<class of the object> \<left> \<top> \<width> \<height>**: For ground truth files.
  
  **\<class of the object> \<confidence> \<left> \<top> \<width> \<height>**: For detection files.

The next step is to create an `Evaluator` object that provides us the metrics:

```python
# Create an evaluator object in order to obtain the metrics
evaluator = Evaluator()
```

With the ```evaluator``` object, you will have access to methods that retrieve the metrics:

| Method | Description | Parameters | Returns |
|------|-----------|----------|-------|
|  GetPascalVOCMetrics | Get the metrics used by the VOC Pascal 2012 challenge | `boundingboxes`: Object of the class `BoundingBoxes` representing ground truth and detected bounding boxes; `IOUThreshold`: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5); | List of dictionaries. Each dictionary contains information and metrics of each class. The keys of each dictionary are:  `dict['class']`: class representing the current dictionary; `dict['precision']`: array with the precision values; `dict['recall']`: array with the recall values; `dict['AP']`: **average precision**; `dict['interpolated precision']`: interpolated precision values; `dict['interpolated recall']`: interpolated recall values; `dict['total positives']`: total number of ground truth positives; `dict['total TP']`: total number of True Positive detections; `dict['total FP']`: total number of False Negative detections; |
PlotPrecisionRecallCurve |	Plot the Precision x Recall curve for a given class | `classId`: The class that will be plot; `boundingBoxes`: Object of the class `BoundingBoxes` representing ground truth and detected bounding boxes; `IOUThreshold`: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5); `showAP`: if True, the average precision value will be shown in the title of the graph (default = False); `showInterpolatedPrecision`: if True, it will show in the plot the interpolated precision (default = False); `savePath`: if informed, the plot will be saved as an image in this path (ex: `/home/mywork/ap.png`) (default = None); `showGraphic`: if True, the plot will be shown (default = True) | The dictionary containing information and metric about the class. The keys of the dictionary are: `dict['class']`: class representing the current dictionary; `dict['precision']`: array with the precision values; `dict['recall']`: array with the recall values; `dict['AP']`: **average precision**; `dict['interpolated precision']`: interpolated precision values; `dict['interpolated recall']`: interpolated recall values; `dict['total positives']`: total number of ground truth positives; `dict['total TP']`: total number of True Positive detections; `dict['total FP']`: total number of False Negative detections |

The snippet below is used to plot the Precision x Recall curve:

```python
# Plot Precision x Recall curve
evaluator.PlotPrecisionRecallCurve('object', # Class to show
                                   boundingboxes, # Object containing all bounding boxes (ground truths and detections)
                                   IOUThreshold=0.3, # IOU threshold
                                   showAP=True, # Show Average Precision in the title of the plot
                                   showInterpolatedPrecision=False) # Don't plot the interpolated precision curve
```

We can have access to the Average Precision value of each class using the method `GetPascalVocMetrics`:

```python
metricsPerClass = evaluator.GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.3)
print("Average precision values per class:\n")
# Loop through classes to obtain their metrics
for mc in metricsPerClass:
    # Get metric values per each class
    c = mc['class']
    precision = mc['precision']
    recall = mc['recall']
    average_precision = mc['AP']
    ipre = mc['interpolated precision']
    irec = mc['interpolated recall']
    # Print AP per class
    print('%s: %f' % (c, average_precision))
```

See [here](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_2/sample_2.py) the full code.
