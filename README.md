

##Vehicle Detection Project##

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/car_hog.png
[image21]: ./output_images/non_car_hog.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/detected_frame.png
[image51]: ./output_images/frame1_heatmap.png
[image52]: ./output_images/frame2_heatmap.png
[image53]: ./output_images/frame3_heatmap.png
[image54]: ./output_images/frame4_heatmap.png
[image6]: ./output_images/labels.png
[image7]: ./output_images/pipeline_result.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained  in function `get_hog_features()` the file called `feature_utils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image21]

####2. Explain how you settled on your final choice of HOG parameters.

I experimented with a number of different  color spaces and color channels to extract HOG features, i tried different combinations of HOG parameters, and i also considered the  spatial color and histogram features. <br>
I trained a linear SVM using combinations of HOG, spatial color and histogram features extracted from the chosen color channels, and experimented with diffenrent parameters for each kind of features. <br>
For HLS color space the L-channel appears to be most important, followed by the S channel. I discarded RGB color space, for its undesirable properties under changing light conditions. YUV and YCrCb also provided acceptable results, but performed a little bit worse than HLS. There was relatively little variation in the final accuracy when running the SVM with some of the individual channels of HSV,HLS and LUV.<br>
After several experiments with diffent feature combination and different parameters, I finally settled with HLS space and a value of pixels_per_cell=(8,8). Using larger values of than orient=9 did not have a striking effect and only increased the feature vector. Similarly, using values larger than cells_per_block=(2,2) did not improve results, which is why these values were chosen. So in a word, the parameters for hog feature extraction is as follows:<br>
```
color_channels="HLS"
pixels_per_cell=(8, 8)
cells_per_block=(2, 2)
orient=9
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I use a combination of hog, spatial color and histogram as the  training features. <br>
Firstly I load all car and not-car images and extract features from them,  and i use `sklearn.preprocessing.StandardScaler` to normalize the features, and dump the scaled features and  the feature scaler for later usage. The related code is defined in function `load_dataset()` in the file `train_data.py`.<br>
Secondly I split the training features and labels into two splits, the smaller split is 20% of the total samples and is used as validation set. I shuffled the training split, and use `sklearn.model_selection.GridSearchCV` to tune the `C` parameter for the linear SVC model. The related code is defined in function `train_classifier()` in file **classifier.py**, here is some code snip:
```python
# Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 10000)
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=rand_state)

    # shuffle the training set
    train_features, train_labels = shuffle(train_features, train_labels)

    # tuning the C parameter
    tuned_parameters = {'C': [1, 10]}
    clf = GridSearchCV(svm.LinearSVC(), tuned_parameters)
    clf.fit(train_features, train_labels)
```

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search in function `sliding_window_search()` at the top of file `sliding_window.py`. <br>
The function looks up a small region of the whole image and scale the region by a specified factor,  and then extract hog features for the whole region. After that i slide a window through the region, for each individual window, i did not calculate the hog feature from scrach, but use the hog feature values where the window overlapped with the whole region, which reduce the calculation time a lot. For spatial color and histogram features, i calculated them for each window position.
Since bigger cars appears closer than small cars in the video frame, i use different search area for different scales, here is the configuration defined in `configure.py`:
```
sliding_window_cfg = {
    "scales": (1.5, 2.0, 3.5),
    "y_start_stop": ((400, 528), (400, 560), (400, 660)),
    "x_start_stop": ((650, 1281), (650, 1281), (650, 1281), ),
}
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided an acceptable result.  Here is an image demonstrating the detection result  using the sliding window search:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's the [link to test video result](./test_video_output.mp4) and [link to project video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
The code for filtering out false positives and combining overlapping boxes was defined in member function `detect()` of  `class **VehicleDetector**` in file `vehicle_detector.py`,  after calling to `sliding_window_search()`.<br>

By calling to `sliding_window_search()` with each video frame, i got the possible car positions as bounding boxes.  From the positive detections I created a heatmap and then thresholded that map with 1 as threshold. After that I sumed up the current heatmap with previous 4 heatmaps,  and then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are 4 frames and their corresponding heatmaps:

![alt text][image51]
![alt text][image52]
![alt text][image53]
![alt text][image54]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 4 frames:
![alt text][image6]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is probably most likely to fail in cases where vehicles (or the HOG features thereof) don't resemble those in the training dataset, but lighting and environmental conditions might also play a role (e.g. a white car against a white background). from my experiments, i found that smaller window scales tended to produce more false positives, but they also did not often correctly label the smaller, distant cars.
I think following ideas may help improve the performance:<br>
(1) trying more features such as LBP, SIFT etc, different features contains different information of the vehicle, an appropriate combination of well designed features may result in better result;<br>
(2) train the model with more training data;<br>
(3) using deep learning based method, such as YoLo, Faster-RCNN. As i konw, YoLo can detect multiple objects in video in realtime and had good accuracy. Faster-RCNN is better at detection accuracy but a little bit slower than YoLo.

