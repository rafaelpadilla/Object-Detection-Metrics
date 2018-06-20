# Sample 1

This sample was created to illustrate the usage of the classes **BoundingBox** and **BoundingBoxes**. Objects of the class `BoundingBox` are an abstraction of the detections or the ground truth boxes. The object of the class `BoundingBoxes` is used by evaluation methods and represents a collection of bounding boxes.

The full code can be accessed [here](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_1/sample_1.py).

If you just want to evaluate your detections dealing with a high level interface, just check the instructions [here](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md#how-to-use-this-project).

### The code

The classes BoudingBox and BoundingBoxes are in the `lib/` folder. The file `_init_paths.py` imports these contents into our example. The file `utils.py` contains basically enumerators and useful functions. The code below shows how to import them:  

```python
import _init_paths
from utils import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
```
Don't forget to put the content of the folder `\lib` in the same folder of your code.

All bounding boxes (detected and ground truth) are represented by objects of the class `BoundingBox`. Each bounding box is created using the constructor. Use the parameter `bbType` to identify if the box is a ground truth or a detected one. The parameter `imageName` determines the image that the box belongs to. All the parameters used to create the object are:  

* `imageName`: String representing the image name.
* `classId`: String value representing class id (e.g. 'house', 'dog', 'person')
* `x`: Float value representing the X upper-left coordinate of the bounding box.
* `y`: Float value representing the Y upper-left coordinate of the bounding box.
* `w`: Float value representing the width bounding box. It can also be used to represent the X lower-right coordinate of the bounding box. For that, use the parameter `format=BBFormat.XYX2Y2`.
* `h`: Float value representing the height bounding box. It can also be used to represent the Y lower-right coordinate of the bounding box. For that, use the parameter `format=BBFormat.XYX2Y2`.
* `typeCoordinates`: (optional) Enum (`CoordinatesType.Relative` or `CoordinatesType.Absolute`) representing if the bounding box coordinates (x,y,w,h) are absolute or relative to size of the image. Default: 'Absolute'. Some projects like YOLO identifies the detected bounding boxes as being relative to the image size, it may be useful for cases like that. Note that if the coordinate type is relative, the `imgSize` parameter is required.
* `imgSize`: (optional) 2D vector (width, height)=>(int, int) representing the size of the image of the bounding box. If `typeCoordinates=CoordinatesType.Relative`, the parameter `imgSize` is required.
* `bbType`: (optional) Enum (`bbType=BBType.Groundtruth` or `bbType=BBType.Detection`) identifies if the bounding box represents a ground truth or a detection. Not that if it is a detection, the classConfidence has to be informed.
* `classConfidence`: (optional) Float value representing the confidence of the detected class. If detectionType is Detection, classConfidence needs to be informed.
* `format`: (optional) Enum (`BBFormat.XYWH` or `BBFormat.XYX2Y2`) indicating the format of the coordinates of the bounding boxes. If `format=BBFormat.XYWH`, the parameters `x`,`y`,`w` and `h` are: \<left>, \<top>, \<width> and \<height> respectively. If `format=BBFormat.XYX2Y2`, the parameters `x`,`y`,`w` and `h` are: \<left>, \<top>, \<right> and \<bottom> respectively.

**Attention**: The bounding boxes of the same image (detections or ground truth) must have have the same `imageName`. 

The snippet below shows the creation of bounding boxes of 3 different images (000001.jpg, 000002.jpg and 000003.jpg) containing 2, 1 and 1 ground truth objects to be detected respectively. There are 3 detected bounding boxes, one at each image. The images are available in the folder [sample_1/images/detections/](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_1/images/detections) and [sample_1/images/groundtruths/](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_1/images/groundtruths).

```python
# Ground truth bounding boxes of 000001.jpg
gt_boundingBox_1 = BoundingBox(imageName='000001', classId='dog', x=0.34419263456090654, y=0.611, 
                               w=0.4164305949008499, h=0.262, typeCoordinates=CoordinatesType.Relative,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(353,500))
gt_boundingBox_2 = BoundingBox(imageName='000001', classId='person', x=0.509915014164306, y=0.51, 
                               w=0.9745042492917847, h=0.972, typeCoordinates=CoordinatesType.Relative,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(353,500))
# Ground truth bounding boxes of 000002.jpg
gt_boundingBox_3 = BoundingBox(imageName='000002', classId='train', x=0.5164179104477612, y=0.501, 
                               w=0.20298507462686569, h=0.202, typeCoordinates=CoordinatesType.Relative,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(335,500))
# Ground truth bounding boxes of 000003.jpg
gt_boundingBox_4 = BoundingBox(imageName='000003', classId='bench', x=0.338, y=0.4666666666666667, 
                               w=0.184, h=0.10666666666666666, typeCoordinates=CoordinatesType.Relative, 
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(500,375))
gt_boundingBox_5 = BoundingBox(imageName='000003', classId='bench', x=0.546, y=0.48133333333333334,
                               w=0.136, h=0.13066666666666665, typeCoordinates=CoordinatesType.Relative,
                               bbType=BBType.GroundTruth, format=BBFormat.XYWH, imgSize=(500,375))
# Detected bounding boxes of 000001.jpg
detected_boundingBox_1 = BoundingBox(imageName='000001', classId='person', classConfidence= 0.893202, 
                                     x=52, y=4, w=352, h=442, typeCoordinates=CoordinatesType.Absolute, 
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(353,500))
# Detected bounding boxes of 000002.jpg
detected_boundingBox_2 = BoundingBox(imageName='000002', classId='train', classConfidence=0.863700, 
                                     x=140, y=195, w=209, h=293, typeCoordinates=CoordinatesType.Absolute,
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(335,500))
# Detected bounding boxes of 000003.jpg
detected_boundingBox_3 = BoundingBox(imageName='000003', classId='bench', classConfidence=0.278000, 
                                     x=388, y=288, w=493, h=331, typeCoordinates=CoordinatesType.Absolute,
                                     bbType=BBType.Detected, format=BBFormat.XYX2Y2, imgSize=(500,375))
```

The object `BoundingBoxes` represents a collection of the bounding boxes (ground truth and detected). Evaluation methods of the class `Evaluator` use the `BoundingBoxes` object to apply the metrics. The following code shows how to add the bounding boxes to the collection:

```python
# Creating the object of the class BoundingBoxes 
myBoundingBoxes = BoundingBoxes()
# Add all bounding boxes to the BoundingBoxes object:
myBoundingBoxes.addBoundingBox(gt_boundingBox_1)
myBoundingBoxes.addBoundingBox(gt_boundingBox_2)
myBoundingBoxes.addBoundingBox(gt_boundingBox_3)
myBoundingBoxes.addBoundingBox(gt_boundingBox_4)
myBoundingBoxes.addBoundingBox(gt_boundingBox_5)
myBoundingBoxes.addBoundingBox(detected_boundingBox_1)
myBoundingBoxes.addBoundingBox(detected_boundingBox_2)
myBoundingBoxes.addBoundingBox(detected_boundingBox_3)
```

You can use the method `drawAllBoundingBoxes(image, imageName)` to add ground truth bounding boxes (in green) and detected bounding boxes (in red) into your images:

```python
import cv2
import numpy as np
import os
currentPath = os.path.dirname(os.path.realpath(__file__))
gtImages = ['000001', '000002', '000003']
for imageName in gtImages:
    im = cv2.imread(os.path.join(currentPath,'images','groundtruths',imageName)+'.jpg')
    # Add bounding boxes
    im = myBoundingBoxes.drawAllBoundingBoxes(im, imageName)
    # Uncomment the lines below if you want to show the images
    #cv2.imshow(imageName+'.jpg', im)
    #cv2.waitKey(0)
    cv2.imwrite(os.path.join(currentPath,'images',imageName+'.jpg'),im)
    print('Image %s created successfully!' % imageName)
```

Results: 

<!--- Images with bounding boxes --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_1/images/000001.jpg"   width="20%" align="center"/>/>
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_1/images/000002.jpg" width="20%" align="center"/>/>
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/samples/sample_1/images/000003.jpg" width="33%" align="center"/>/>
  </p>

**Of course you won't build your bounding boxes one by one as done in this example.** You should read your detections within a loop and create your bounding boxes inside of it. [Sample_2](https://github.com/rafaelpadilla/Object-Detection-Metrics/tree/master/samples/sample_2) demonstrates how to read detections from folders containing .txt files.
