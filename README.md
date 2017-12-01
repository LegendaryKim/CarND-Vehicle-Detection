## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/car_notcar_hog.png
[image3]: ./output_images/hog_sub_sampling_result_1.png
[image4-1]: ./output_images/hog_sub_sampling_windows_1.png
[image4-2]: ./output_images/hog_sub_sampling_windows_2.png
[image4-3]: ./output_images/hog_sub_sampling_windows_3.png
[image4-4]: ./output_images/hog_sub_sampling_windows_4.png
[image4-5]: ./output_images/hog_sub_sampling_windows_5.png
[image5-1]: ./output_images/heatmap_threshold_1.png
[image5-2]: ./output_images/heatmap_threshold_2.png
[image5-3]: ./output_images/heatmap_threshold_3.png
[image5-4]: ./output_images/heatmap_threshold_4.png
[image5-5]: ./output_images/heatmap_threshold_5.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step to load images is contained in the 1st ~ 3rd code cells of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=2`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters of HOG with ```spatial_bin = 16``` and ```history_bin = 16```, and estimate estimate featuring, training and predicting time, and their accuracies with 2000 image samples. Linear SVM classifier is selected. 

| #    | Colorspace | Orient. | Pixel/Cell | Cell/Block | HOG Chann.| Featuring+Training T.| Predicting T.| Accuracy|
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|1    |RGB  |9    |8    |2    |ALL  |23.14|0.0025|0.955|
|2    |HSV  |9    |8    |2    |ALL  |22.88|0.0024|0.983|
|3    |HLS  |9    |8    |2    |ALL  |23.72|0.0026|0.983|
|4    |YCrCb|9    |8    |2    |ALL  |22.87|0.0026|0.990|
|5    |YCrCb|9    |8    |2    |0    |9.81 |0.0021|0.963|
|6    |YCrCb|9    |8    |2    |1    |10.24|0.0020|0.958|
|7    |YCrCb|9    |8    |2    |2    |10.45|0.0023|0.953|
|8    |HSV  |9    |8    |2    |0    |10.68|0.0023|0.953|
|9    |HSV  |9    |8    |2    |1    |10.81|0.0021|0.965|
|10   |HSV  |9    |8    |2    |2    |9.98 |0.0019|0.980|
|11   |HLS  |9    |8    |2    |0    |10.32|0.0023|0.955|
|12   |HLS  |9    |8    |2    |1    |10.41|0.0021|0.973|
|13   |HLS  |9    |8    |2    |2    |10.59|0.0022|0.930|
|14   |LUV  |9    |8    |2    |0    |9.90 |0.0024|0.985|
|15   |YUV  |9    |8    |2    |0    |9.57 |0.0024|0.985|
|16   |YUV  |9    |8    |2    |1    |10.2 |0.0022|0.980|
|17   |YCrCb|12   |8    |2    |ALL  |20.23|0.0026|0.980|
|18   |YCrCb|12   |12   |2    |ALL  |14.13|0.0026|0.985|
|19   |YCrCb|12   |16   |2    |ALL  |11.51|0.0026|0.990|

I chose the last parameter set, ```colorspace = "YCrCb" ```, ```orient = 12```, ```pixel_per_cell=16```, ```cell_per_block=2```, ```channel="ALL"```becuase I considered not only the accuracy with the classifer's preictions but also the computational times. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the section titled, **HOG Sub-sampling Window Search**, the method I selected combines spatial features, color histogram features, and HOG feature extraction with a sliding window search. By the ```finds_cars``` function, I can reduce time by performing the feature extraction on entire image rather than featuring on each window. 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To perform an effective slidng window search, I made several size of the windows according to the ```y``` direction of image. Every Windows are half-overlaped. 

![alt text][image4-1]
![alt text][image4-2]
![alt text][image4-3]
![alt text][image4-4]
![alt text][image4-5]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on the input images using YCrCb all 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here is the result of the sliding window search:
![alt text][image3]



---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:
![alt text][image5-1]
![alt text][image5-2]
![alt text][image5-3]
![alt text][image5-4]
![alt text][image5-5]

To realize smooth-detection, I utilized the class `vehicle_detection` to mainitn the most recent detection results within the previous 10 frames. And, I combined the saved rectangles and draw the heatmap. A threshold is defined as `5 + len(v_detect.current_rects)//2` to neglect false postivie detections in the images. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The first problem I faced was finding the best parameters to generate HOG features and train the linear SVM strongly. The accuracy of SVM classifer was important, but, I had to consider computational time because it consumed lots of time to do simulation. For example, if I increased the sliding window overlapping rate would improved the accuracy, but I couldn't beacaused of extremely increased time cost. At last, this kind of program would be applyed for real-time simulation, the processing cost should be optimized.

Furtheremore, the generating scheme of sliding windows was hard for me. To recognize different distant cars, I changed the size of sliding windows according to the y-coordinate. However, It would be not suitable when the car is running on sharply curved or upward/downward roads. I hope to find more reasonable method to set the size of windows.

Lastly, in my view, linear SVM seems not the most powerful classifer to detect vehicle on the road. If we utilize CNN for this image classification, it could improved the result.

