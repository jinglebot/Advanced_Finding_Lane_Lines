
# **Advanced Lane Finding Project**
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[image1]: ./output_images/undistorted_chessboards/figure_1.jpg "Undistorted"
[image2]: ./output_images/undistorted_test_images/test6.jpg ""Road Transformed"
[image3]: ./examples/threshold_images/figure_1-9.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/figure_6.jpg "Fit Visual"
[image6]: ./output_images/result_test_images/test1.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. [My write up report](https://github.com/jinglebot/CarND-Advanced-Lane-Lines/blob/master/writeup_report.md) 


### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients and an example of a distortion corrected calibration image

The code for this step is contained in the cell block entitled *Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.*, cells 1 - 4 of the IPython notebook located ** [here](https://github.com/jinglebot/CarND-Advanced-Lane-Lines/blob/master/notebook/detect_lane.ipynb).**

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Undistorted sample image][image1]

### Pipeline 

#### 1. Example of a distortion-corrected image

Initially, I saved the camera calibration and distortion coefficients in a `pickle` so I can use it whenever. I created a `cal_undistort()` function that takes in an image to be undistorted and camera calibration and distortion coefficients to output the undistorted image. The code is in cell block entitled *Apply a distortion correction to raw images.*, cell 5. The images came out like this: 

![Undistorted Image][image2]

#### 2. Color transforms, gradients or other methods to create a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image. This step is under cell block entitled *Use color transforms, gradients, etc., to create a thresholded binary image.* I divided this cell block into two: the gradient thresholds (under cell block *Gradients*) and the color thresholds (under cell block *Colorspace*). In the gradients threshold, I would pre-process an image to give the grayscale version, the Sobel X and Sobel Y operators. These I need for the gradient threshold functions: the absolute Sobel, the magnitude and the direction thresholds. In the color threshold, I have the conversion to gray, RGB and HLS colorspace. I manipulated the `Red` channel in the RGB (for the white color of the lanes) and the `Lightness` and `Saturation` channels in the HLS (for the shadows on  the road and the yellow-colored lanes). Then, I combined them and, with a lot of trial and error, was able to get a `combined binary image` as well as a bonus color binary image. The code is under cell block entitled *Apply and combine thresholds*. Here's an example of my output for this step.

![Colored Binary Image and Combined Binary Image][image3]

#### 3. Perspective transform and an example of a transformed image

The code for my perspective transform is under cell block entitled *Apply a perspective transform to rectify binary image ("birds-eye view").* The function is called `corners_unwarp()`, which appears in the 3rd code cell of the said cell block. The `corners_unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I created the function `get_shapes()` to get those `src` and `dst` points beforehand, as well as the `vertices` needed in the `region_of_interest()` function that comes before `corners_unwarp()`.

```python
    img_size = (img.shape[1], img.shape[0])
    # left,top --> left,bottom --> right,bottom --> right, top
    top = img_size[1] * 0.62
    bottom = img_size[1] 
    left_top = img_size[0] * 0.47
    right_top = img_size[0] * 0.53
    left_bottom = img_size[0] * 0.155
    right_bottom = img_size[0] * 0.87
    src = np.float32 ([[left_top, top], [left_bottom,bottom], [right_bottom, bottom], [right_top, top]])
    dst = np.float32([[img_size[0] /4, 0], [img_size[0] /4, img_size[1]], 
                [img_size[0] * 3/4, img_size[1] ], [img_size[0] * 3/4, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 601, 446      | 320, 0        |
| 198, 720      | 320, 720      |
| 1113, 720     | 960, 720      |
| 678, 446      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Undistorted Image with Source Points and its Warped Version with Destination Points][image4]

#### 4. Identifying lane-line pixels and fitting their positions with a polynomial

Under the code block entitled *Detect lane pixels and fit to find the lane boundary.*, I have 4 functions. The first one is in the first cell of the block and it's called `detect_lane_boundary()`. It takes in the perspective-transformed binary image called `binary_warped` and, using the histogram and the sliding window technique, detects which part of the image has the highest binary values on its left and right half. Those binary values are converted to 2nd order polynomial line equations which can then be plotted to give you, tadah, the lane lines: values, images and all. 

The second function in the block is called `detect_lane_boundary2()`. This function takes in and uses also a binary warped image and takes the line values from a previous image (if it's a series of images, like a video) to easily detect the lane lines in the current image. The second function skips the histogram and sliding windows part of the first function.

Since I was already using the two lane detection functions with the histograms and sliding windows, I skipped the third one which is called `detect_lane_boundary3()` which employs convolutions. 

The last function in the block is where `Line class` comes in and the smoothing average is computed. The function is called `detect_smoothed_lanes()` under the code block entitled *Apply smoothing average*. The function takes in the left and right lane line values and stores them in a `Line class` array. The averages for the values for up to `iterations = 10` are computed and returned as results. If an image fed in the function is good (meaning, the lane lines were detected), the lane values of the image is included in the computation of the averages. If it's a bad image (meaning, the lines detected have unrealistic values), then it is not included and considered undetected. By filtering the images this way, the next image can have the option to use the lane lines in the previous image (if detected) to find its own lane lines using `detect_lane_boundary2()`. Nevertheless, whether good or bad image, the average values will be returned and only the ones to be used in the next series of functions. 

The sample image output of these functions are like this:

![Detected lane line boundaries][image5]

#### 5. Radius of curvature of the lane and the position of the vehicle with respect to center

I calculated the radius of curvature of the lane in first cell of the code block entitled *Determine the curvature of the lane and vehicle position with respect to center.* It also returns a boolean value that answers if the lanes are both curving in the same direction. I added two functions to supply some needed values: the `lane_space()` and the `is_lane_detected()`. The `lane_space()` takes in the x values of the left and right lanes and returns a boolean value that answers whether the difference between the distance between lanes at the top of the image and at the bottom is acceptable. The `is_lane_detected()` accepts the results of the `lane_space()` and `lane_curvature()`'s curverad_equal to determine if the lanes are detected or not.

The calculation of the lane_center is in the `detect_smoothed_lanes()` function along with the lane_detection and lane curvature functions. This is because the left and right base position values are there and the calculation involves only one line of code. The position of the vehicle with respect to center, however, is in the `show_lane()` function so as to get the offset value as well.

#### 6. Example image of my result plotted back down onto the road with the lane area is identified clearly.

I implemented this step in the code block entitled *Warp the detected lane boundaries back onto the original image.* The function is called `reverse_transform()` which accepts as arguments the undistorted version of the image, the warped version of the image, the left and right x values of the line, the inverse of perspective transform and ploty. The result is fed to another function in code block *Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.* which displays the image with an overlay of the vehicle position and mean radius of curvature values. Here is an example of my result on a test image:

![Final Image Result][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project, I used `classes` to be able to store averages that will smoothen the lanes in the video. I used `class methods` to store the line values in the `classes`' arrays as well as their mean values. By having a separate function for each step, my goal was to make the code more understandable. But with all the input and output of the same variables from one function to another, it could create a lot of chances for error. If I were going to pursue this project further, I would streamline the number of functions to an acceptable yet understandable number.

The selection of values to use for the color and gradient thresholds were time consuming. If I were going to pursue this project further, I would also create a dashboard with control buttons and threshold counters for tweaking the values as well as what the image would look like when the values are tweaked. 

And lastly, my lane_detection code depends solely on the lanespace and curve values. If there are no lanes and curve values, then the code will fail. The code can accept bad images up to a certain extent but if there are no good images to average then it'll also fail. If I were going to pursue this project further, I would find other ways of detecting the path other than lane lines and curves. 
