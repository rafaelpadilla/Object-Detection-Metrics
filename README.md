# Metrics for object detection
  
The motivation of this project is the lack of consensus used by different works and implementations concerning the *evaluation metrics of the object detection problem*. Although on-line competitions use their own metrics to evaluate the task of object detection, just some of them offer reference code snippets to calculate the accuracy of the detected objects.  
Researchers who want to evaluate their work using different datasets than those offered by the competitions, need to implement their own version of the metrics. Sometimes a wrong or different implementation can create different and biased results. Ideally, in order to have trustworthy benchmarking among different approaches, it is necessary to have a flexible implementation that can be used by everyone regardless the dataset used.  

This project aims to provide easy-to-use functions implementing the same metrics used by the the most popular competitions of object detection. Our implementation does not require modifications of your detection model to complicated input formats, avoiding conversions to XML or JSON files. We simplified the input data (ground truth bounding boxes and detected bounding boxes) and gathered in a single project the main metrics used by the academia and challenges. Our implementation was carefully compared against the official implementations and our results are exactly the same.   

In the topics below you can find an overview of the most popular metrics and competitions, as well as samples showing how to use our code.

## Different competitions, different metrics  

* **[PASCAL VOC challenge](http://host.robots.ox.ac.uk/pascal/VOC/)** offers a Matlab script in order to evaluate the quality of detected objects. A documentation explaining their criteria for object detection metrics can be accessed [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000). Participants of the competition can use the Matlab script to measure the accuracy of their detections before submitting their results. The current metrics used by the current VOC PASCAL object detection challenge are the **Precision/Recall curve** and **Average Precision**.  
The PASCAL VOC Matlab evaluation code reads the ground truth bounding boxes from XML files, requiring changes in the code if you want to apply it to other datasets or to your speficic cases. Even though projects such as [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) implement VOC PASCAL evaluation metrics, it is also necessary to convert the detected bounding boxes into their specific format. [Tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md) framework also has their PASCAL VOC metrics implementation.  

* **[COCO challenge](https://competitions.codalab.org/competitions/5181)** uses different metrics to evaluate the accuracy of object detection of different algorithms. [Here](http://cocodataset.org/#detection-eval) you can find a documentation explaining the 12 metrics used for characterizing the performance of an object detector on COCO. This competition offers python and matlab codes so users can verify their scores before submitting the results. It is also necessary to convert the results in a [format](http://cocodataset.org/#format-results) required by the competition.  

* **[Google Open Images Dataset V4 competition](https://storage.googleapis.com/openimages/web/challenge.html)** also uses mean Average Precision (mAP) over the 500 classes to evaluate the object detection task. 

* **[ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)** defines an error for each image, considering the class and the overlapping region between ground truth and detected boxes. The total error is computed as the average of all min errors of all the images in the test dataset. [Here](https://www.kaggle.com/c/imagenet-object-localization-challenge#evaluation) are more details about their evaluation method.  

## Important definitions  

### Intersection Over Union (IOU)

Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box ![](http://latex.codecogs.com/gif.latex?B_%7Bgt%7D) and a predicted bounding box ![](http://latex.codecogs.com/gif.latex?B_p). With it we can tell if a detection is valid (True Positive) or not (False Positive).  
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between the bounding boxes:  

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?%5Ctext%7BIOU%7D%20%3D%20%5Cfrac%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccap%20B_%7Bgt%7D%5Cright%29%7D%7B%5Ctext%7Barea%7D%20%5Cleft%28B_p%20%5Ccup%20B_%7Bgt%7D%5Cright%29%7D">
</p>

The image below shows the IOU between a ground truth bounding box (in green) and a detected bounding box (in red).

<!--- IOU --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/iou.png" align="center"/></p>

### True Positive, False Positive, False Negative and True Negative  

Some basic concepts used by the metrics:  

* **True Positive (TP)**: A correct detection. Detection with IOU ≥ _threshold_  
* **False Positive (FP)**: A wrong detection. Detection with IOU < _threshold_  
* **False Negative (FN)**: A ground truth not detected  
* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the detection object task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

_threshold_: depending on the metric, it is usually set as 50%, 75% or 95%.

### Precision

Precision is ability of a model to identify **only** the relevant objects. It is the percentage of positive predictions that are correct and is given by:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Precision%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FP%7D%3D%5Cfrac%7BTP%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

### Recall 

Recall is ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?Recall%20%3D%20%5Cfrac%7BTP%7D%7BTP&plus;FN%7D%3D%5Cfrac%7BFN%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

## Metrics

### Precision x Recall curves

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed. There is a curve for each object class. An object detector of a particular class is considered good if its prediction stays high as recall increases, which means that if you vary the confidence threshold the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall). A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases.
This kind of curve is used by the VOC PASCAL 2012 challenge and is available in our implementation.  

### Average Precision

Another way to compare the performance of object detectors is to calculate the area under the curve (AUC) of the Precision x Recall curve. As AP curves are often zigzag curves frequently going up and down, comparing different curves (different detectors) in the same plot usually is not an easy task - because the curves tend to cross each other much frequently. That's why Average Precision (AP), a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1. 

VOC PASCAL 2012 challenge uses the *interpolated average precision*. It tries to summarize the shape of the Precision x Recall curve by averaging the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%5Csum_%7Br%5Cin%5C%7B0%2C0.1%2C...%2C1%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D">
</p>

with

<p align="center"> 
<img src="http://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D%3D%5Cmax_%7B%5Cwidetilde%7Br%7D%3A%5Cwidetilde%7Br%7D%5Cgeqslant%7Br%7D%7D%20%5Crho%20%5Cleft%20%28%5Cwidetilde%7Br%7D%20%5Cright%29">
</p>

where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

Instead of using the precision observed at each point, the AP is obtained by interpolating the precision at each level ![](http://latex.codecogs.com/gif.latex?r) taking the maximum precision whose recall value is greater than ![](http://latex.codecogs.com/gif.latex?r).

#### Illustrated example 

An example helps us understand better the concept of the interpolated average precision. Look at the detections below:
  
<!--- Image samples 1 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/samples_1.png" align="center"/></p>
  
There is a total of 7 images with 15 ground truth objects representented by the green bounding boxes and 24 detected objects represented by the red bounding boxes. Each detected object has a confidence level and is identified by a letter (A,B,...,Y).  
The following table shows the bounding boxes with their corresponding confidences. The last column identifies the detections as TP or FP. In this example a TP is considered if IOU ![](http://latex.codecogs.com/gif.latex?%5Cgeq) 30%, otherwise it is a FP. By looking at the images above we can roughly tell if the detections are TP or FP.

<!--- Table 1 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/table_1.png" align="center"/></p>

<!---
| Images | Detections | Confidences | TP or FP |
|:------:|:----------:|:-----------:|:--------:|
| Image 1 | A | 88% | FP |
| Image 1 | B | 70% | TP |
| Image 1 |	C	| 80% | FP |
| Image 2 |	D	| 71% | FP |
| Image 2 |	E	| 54% | TP |
| Image 2 |	F	| 74% | FP |
| Image 3 |	G	| 18% | TP |
| Image 3 |	H	| 67% | FP |
| Image 3 |	I	| 38% | FP |
| Image 3 |	J	| 91% | TP |
| Image 3 |	K	| 44% | FP |
| Image 4 |	L	| 35% | FP |
| Image 4 |	M	| 78% | FP |
| Image 4 |	N	| 45% | FP |
| Image 4 |	O	| 14% | FP |
| Image 5 |	P	| 62% | TP |
| Image 5 |	Q	| 44% | FP |
| Image 5 |	R	| 95% | TP |
| Image 5 |	S	| 23% | FP |
| Image 6 |	T	| 45% | FP |
| Image 6 |	U	| 84% | FP |
| Image 6 |	V	| 43% | FP |
| Image 7 |	X	| 48% | TP |
| Image 7 |	Y	| 95% | FP |
--->

In some images there are more than one detection overlapping a ground truth. In those cases the detection with the highest IOU is taken, discarding the other detections. This rule is also applied by the VOC PASCAL 2012 metric: "e.g. 5 detections of a single object is counted as 1 correct detection and 4 false detections”.

The Precision x Recall curve is plotted by calculating the precision and recall values of the accumulated TP or FP detections. For this, first we need to order the detections by their confidences, then we calculate the precision and recall for each accumulated detection as shown in the table below: 

<!--- Table 2 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/table_2.png" align="center"/></p>

<!---
| Images | Detections | Confidences |  TP | FP | Acc TP | Acc FP | Precision | Recall |
|:------:|:----------:|:-----------:|:---:|:--:|:------:|:------:|:---------:|:------:|
| Image 7 |	Y	| 95% | 0 | 1 | 0 | 1 | 0       | 0      |
| Image 5 |	R	| 95% | 1 | 0 | 1 | 1 | 0.5     | 0.0666 |
| Image 3 |	J	| 91% | 1 | 0 | 2 | 1 | 0.6666  | 0.1333 |
| Image 1 | A | 88% | 0 | 1 | 2 | 2 | 0.5     | 0.1333 |
| Image 6 |	U	| 84% | 0 | 1 | 2 | 3 | 0.4     | 0.1333 |
| Image 1 |	C	| 80% | 0 | 1 | 2 | 4 | 0.3333  | 0.1333 |
| Image 4 |	M	| 78% | 0 | 1 | 2 | 5 | 0.2857  | 0.1333 |
| Image 2 |	F	| 74% | 0 | 1 | 2 | 6 | 0.25    | 0.1333 |
| Image 2 |	D	| 71% | 0 | 1 | 2 | 7 | 0.2222  | 0.1333 |
| Image 1 | B | 70% | 1 | 0 | 3 | 7 | 0.3     | 0.2    |
| Image 3 |	H	| 67% | 0 | 1 | 3 | 8 | 0.2727  | 0.2    |
| Image 5 |	P	| 62% | 1 | 0 | 4 | 8 | 0.3333  | 0.2666 |
| Image 2 |	E	| 54% | 1 | 0 | 5 | 8 | 0.3846  | 0.3333 |
| Image 7 |	X	| 48% | 1 | 0 | 6 | 8 | 0.4285  | 0.4    |
| Image 6 |	T	| 45% | 0 | 1 | 6 | 9 | 0.4     | 0.4    |
| Image 4 |	N	| 45% | 0 | 1 | 6 | 10 | 0.375  | 0.4    |
| Image 5 |	Q	| 44% | 0 | 1 | 6 | 11 | 0.3529 | 0.4    |
| Image 3 |	K	| 44% | 0 | 1 | 6 | 12 | 0.3333 | 0.4    |
| Image 6 |	V	| 43% | 0 | 1 | 6 | 13 | 0.3157 | 0.4    |
| Image 3 |	I	| 38% | 0 | 1 | 6 | 14 | 0.3    | 0.4    |
| Image 4 |	L	| 35% | 0 | 1 | 6 | 15 | 0.2857 | 0.4    |
| Image 5 |	S	| 23% | 0 | 1 | 6 | 16 | 0.2727 | 0.4    |
| Image 3 |	G	| 18% | 1 | 0 | 7 | 16 | 0.3043 | 0.4666 |
| Image 4 |	O	| 14% | 0 | 1 | 7 | 17 | 0.2916 | 0.4666 |
--->
 
 Plotting the precision and recall values we have the following *Precision x Recall curve*:
 
 <!--- Precision x Recall graph --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/precision_recall_example_1.png" align="center"/>
</p>
 
As seen before, the idea of the interpolated average precision is to average the precisions at a set of 11 recall levels (0,0.1,...,1). The interpolated precision values are obtained by taking the maximum precision whose recall value is greater than its current recall value. We can visually obtain those values by looking at the recalls starting from the highest (0.4666) to the lowest (0) and, as we decrease the recall, we annotate the precision values that are the highest as shown in the image below:

<!--- interpolated precision curve --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision.png" align="center"/>
</p>

The Average Precision (AP) is the AUC obtained by the interpolated precision. The intention is to reduce the impact of the wiggles in the Precision x Recall curve. We split the AUC into 3 areas (A1, A2 and A3) as shown below:
  
<!--- interpolated precision AUC --->
<p align="center">
<img src="https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/aux_images/interpolated_precision-AUC.png" align="center"/>
</p>

Calculating the total area, we have the AP:  

![](http://latex.codecogs.com/gif.latex?AP%20%3D%20A1%20&plus;%20A2%20&plus;%20A3)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.1333*0.6666%20&plus;%20%5Cleft%20%28%200.4-0.1333%20%5Cright%20%29*0.4285%20&plus;%20%5Cleft%20%28%200.4666-0.4%20%5Cright%20%29*0.3043)  
![](http://latex.codecogs.com/gif.latex?%5Cboldsymbol%7BAP%20%3D%200.2234%7D)


If you want to reproduce these results, see the **Sample 1**.

<!--In order to evaluate your detections, you just need a simple list of `Detection` objects. A `Detection` object is a very simple class containing the class id, class probability and bounding boxes coordinates of the detected objects. This same structure is used for the groundtruth detections.-->


## References

* The Relationship Between Precision-Recall and ROC Curves (Jesse Davis and Mark Goadrich)
Department of Computer Sciences and Department of Biostatistics and Medical Informatics, University of
Wisconsin  
http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

* The PASCAL Visual Object Classes (VOC) Challenge  
http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf

* Evaluation of ranked retrieval results (Salton and Mcgill 1986)  
https://www.amazon.com/Introduction-Information-Retrieval-COMPUTER-SCIENCE/dp/0070544840  
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
