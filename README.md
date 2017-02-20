
# Advanced Lane Finding

This project is about writing a software pipeline to identify the lane boundaries in a video. It is part of the Udacity self-driving car Nanodegree. Please see the links below for details and the project requirements

* [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
* [rubric](https://review.udacity.com/#!/rubrics/571/view)


Introduction
---

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Run the entire pipeline on a sample video recorded on a sunny day on the I-280.

The currect project is implemented as three steps: step1 - camera calibration pipeline, step2 - image processing/testing pipeline and step3 - video processing pipeline. All python code used in step2 and step3 are in two separate python files: tools.py and detect_lane.py files. The tools.py file contains helper functions for image processing part and the detect_lane.py file contains code used for line detection. The images for camera calibration are stored in the "camera_cal" folder. All test images, used for setup internel parameters and testing are saved in the "test_images" folder. The folder "output_images" contains several folders step1 & step2 with generated samples and results. Below, I describe all the steps in details and how I addressed each point in the implementation.

[//]: # (Image References)

[image1]: ./output_images/step1/undistort.jpg "Undistorted"
[image2]: ./output_images/step2/undistort_test5_sample.jpg "Undistorted"
[image3]: ./output_images/step2/binarized_sample.jpg "Binarization Example"
[image4]: ./output_images/step2/camera_view_to_bird_view.jpg "Warp Example"
[image5]: ./output_images/step2/roi_sample.jpg "Region of interest"
[image6]: ./output_images/step2/line_detection_sample.jpg "Line detection"
[image7]: ./output_images/step2/projected_lane_test5.jpg "Projected lines"
[video1]: ./processed_project_video.mp4 "Video"


# Camera calibration 

During detection of lane lines with measurement of curvature it is very important to work with undistorted images, that are free from radial and tangential distortion. To get undistorted images, the camera calibration procedure was used. The camera calibration was implemented as estimation of camera parameters - camera matrix and distortion coefficients - using a special set of chessboard pattern samples, recorded by the camera and saved in the `camera_cal` folder. The same camera was used for video recording.
The code of camera calibration is localized in the `./step1-camera_calibration.ipynb` file and is used to generate a special pickle file `./camera_dist_pickle.p`, containing estimated camera parameters. The parameters from this file are used to undistort all images before using them in image processing pipeline.

The camera calibration process starts by preparing `object points`, which are 3D pattern (x, y, z) coordinates of the chessboard corners in the world coordinate space (assuming that z=0). The `objp` is a generated array of coordinates of object points - calibration pattern points, which is appended to `objpoints` every time when callibration pattern points are successfully detected as chessboard corners in a test image.  `imgpoints` is a vector of accumulated coordinates of calibration pattern points in the space of pattern image, generated for each successful chessboard detection.

Then, `objpoints` and `imgpoints` are used to compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera()` function. The camera matrix and distortion coefficients are saved as a pickle file, named as `camera_dist_pickle.p`. To estimate the quality of camera calibration, I used the `cv2.undistort()` function to apply the distortion correction to the test image. The result of camera callibration is bellow: 
![Undistort][image1]


# Image Processing Pipeline

Image Processing pipeline is implemented as a separate `./step2-image_processing_pipeline.ipynb` file and was used for testing of image processing and lane lines detection with setup parameters. 

### Example of distortion corrected applied to an image

Below you can see the result of undistortion transformation, applied to a test image:
![Undistort][image2]

### Image binarization using gradient and color transforms to detect lane lines 

Image binarization is implemented in `binarize()` function, which can be found in the `tools.py` file as all other helper functions, used as for image processing as for perspective transformation. 

To binarize an image, the following approach is used with combination of information from two color spaces: HSL and HSV. Only the L and S channels are used from HSL color space, while only V channel is used from HSV color space. The L channel is used for gradient extraction and filtering along x, the S channel is used for yellow line extraction, while the V channel is used for white line extraction. Besides, the V channel with some thresholding is used to mask shadows area in binarized image. Below you can see the result as a combination of above described approaches with some thresholding.

![Binarized image][image3]

### Perspective Transform to bird's eye view

Perspective transformation functionality is implemented as a set of three functions named `birdview_corners()`, `warp_img_M()` and `warp_img()`, which can be found in the `tools.py` file. The `birdview_corners()` function defines 4 points as a destination points used for calculation of perspective transformation matrix for mapping from camera view space to "bird's eye" view space. The perspective transformation in both directions (from camera view to "bird's eye" view and back) is implemented in the function `warp_img_M()`. The function `warp_img()` is used as an additional wrapper for the fucntion `warp_img_M()`. Both functions have two parameters: an input RGB image and `tobird` boolean flag to define the direction for perspective transformation.

Below you can see a table with 8 points, used to calculate perspective transformation matrix, and two images: the first image defines 4 points in camera view space and the second image defines 4 points for mappping into  "bird's eye" view space.

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 190, 720      | 330, 720    | 
| 592, 450      | 330, 0      |
| 689, 450      | 981, 0      |
| 1120,720      | 981, 720    |

Transformation from camera view space to "bird's eye" view space is below. Red lines show quadrangles for perpective transformation.

![alt text][image4]


### ROI

Including of additional region of interest allows to reduce noise and artefacts at the bottom of image and, as a result, to improve the robustness of lane detection at the bottom of image. Two functions are used for generation of ROI: the `ROI()` function defines 4 points of ROI region and the `region_of_interest()` function is used for ROI mask generation for these points.

![alt text][image5]


# Lane line delection using sliding windows

All functions used for lane line detection are defined in the `detect_lane.py` file. The example of using helper functions for line detection can be found in the `./step2-image_processing_pipeline.ipynb` file.

The function `find_peaks()` finds a position of left or right line at the bottom of binary warped image via detection of peaks in computed histogram for bottom part of binarized image (the bottom half of binarized image). The calculated histogram is smoothed by gaussian filter and then is used for peak detection with some thresholding: one for noise filtering and other for filtering an expected distance between detected peak and expected position of line at the bottom of image. As a result, the function returns the x value of detected peak, which is used as starting point for lane detection along vertical Y direction.

The function `detect_line()` allows to find left or right line using a window area, sliding along vertical Y direction, and detected peak, returned by `find_peaks()` function. The vertical direction is splitted into 9 vertical zones, initial position of window in next vertical zone is defined by the center of window in previous bottom zone. The center of window in every next vertical zone is recalculated as a center of mass of points in the window with center of window from the previous bottom zone. The function `detect_line()`returns a line as two arrays of X coords and Y coordinates, line-fit coefficients line_fit of a second order polynomial as np.array and a list of window rectangles for debug purpose.

The function `draw_lanes_with_windows()` is added for debug purpose for drawing lane line, detected by `detect_line()` function. See function definition for more details.

To speed-up and smoothness of line detection, the `detect_line_in_roi()` function is added. This function allows to detect a lane line in the next frame, when the same line was successfully detected in previous frame just using line-fit coefficients of it with some ROI.  The function returns a detected line as two arrays of X coords and Y coordinates and line-fit coefficients line_fit of a second order polynomial as np.array. 

The function `draw_detect_line_in_roi()` is added for debug purpose for drawing lane line, detected by `detect_line_in_roi()` function. See function definition for more details.

To accumulate all parameters of detected lane line with some filtering to accept or not the lane detection, a special class `Line` is added. The `Line` class has a special `update()` method to accept result of line detection and uses a FIFO queue with the size of `n` elements for averaging of line-fit coefficients of a second order polynomial along `n` frames. Every time when a new line is detected in next frame, all parameters are updated in the class. But if no line is detected, the oldest line detection result is removed from the queue, until the queue is empty, and a new line detection is run from the scratch. For more details see functions `process_image_ex()` and `get_line_from_image()` in the `detect_lane.py` file.   

Below you can see the result of left and right lane lines detection: the center image is lines detection from scratch, the right image is lines detection with using lines detection information from previous frame. 

![line fit][image6]

### Measuring curvature of the road and vehicle localization

The radius of curvature is calculated in the method `Line.set_radius_of_curvature()`, which is called every time when the next frame is processed in the line method `Line.update()`. The method, used for estimation of Radius of Curvature, can be found [in this tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).  The radius of curvature for a second order polynomial equation `f(y)=A*y^2+B*y+C` is calculated using the following equation: `Rcurv=((1+(2*A*y+B)^2)^3/2)/abs(2*A)`

The distance between detected line and the center of lane is computed in the `Line.set_line_base_pos()` method.  To compute the distance, the following information is used for lanes in USA: the lane is about 30 meters long and 3.7 meters wide. 

### Projection of detected lane lines onto the road.

The code to project detected lane onto the road can be found in the function `project_lanes_onto_road()`, defined in the `detect_lane.py` file. The function `project_lanes_onto_road()` is called every time when the next frame is processed via function `process_image_ex()`, defined in the same python file. The sample code, how these functions are used for lane projection, can be found in last cells of the file `./step2-image_processing_pipeline.ipynb`. For debuging purpose, the `process_image_ex()` function allows to visualize the following information for every frame in real time: binarization of frame, "bird's eye" view to the road and left/right lane lines detection. Below you can see the lane projection for tested image `test_images/test5.jpg`, generated in the `./step2-image_processing_pipeline.ipynb` file.

![projected lanes][image7]

---

# Video Processing Pipeline

The video file `./processed_project_video.mp4`, demostrated below, was generated by pipeline, defined in the file `step3-video_processing_pipeline.ipynb`. This file uses functions, defined in two python files: `tools.py` and `detect_lane.py`
The processed project video can be found here:[link to video result](./processed_project_video.mp4)

---

# Discussion

I spent more time for implementation and debugging of python pipeline code, then for image processing part. The developed pipeline works quite good on project_video.mp4 sample and can be fixed very easy for challenge_video.mp4 by adding the following modifications: 
* Modification of perspective transformation, used to map to the bird's eye view space to get warped image with lane line. The current implementation generates warped images with too much blured lane lines at the top part of image. This debluring decreases robustness of lane detection and is a common reason of 'lane dansing' in generated processed_challenge_video.mp4 sample.
* Moving binarization into warped image (after fixing of perspective transformation above) allows to improve accurancy of lane detection and to use measurement in different areas of road for adaptive thresholding.

The next harder_challenge_video.mp4 sample is more complicated for current pipepline and I see two problems, connected with it: existed areas with strong shadows and moving objects like other vehicles. The first problem can be solved with local thresholding while the second problem can be solved by using information from previous frames. The filtering for large objects can be added to filter areas, covered by moving vehicles. I'll implement all notes if I have free time :)

PS: as a result of this project, I've got usefull tool for experiments with video :) 

[link to my video result](./processed_project_video.mp4)



```python

```
