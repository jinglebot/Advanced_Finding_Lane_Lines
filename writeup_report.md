
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
[image2]: ./test_images/test1.jpg "Road Transformed"
[image2-1]: ./output_images/undistorted_test_images/figure_1-1.jpg "Undistorted Road Image"
[image3]: ./examples/threshold_images/figure_1-9.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"

[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. [My write up report](https://github.com/jinglebot/CarND-Advanced-Lane-Lines/blob/master/writeup_report.md) 


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cell block entitled *Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.*, cells 1 - 4 of the IPython notebook located [here] (https://github.com/jinglebot/CarND-Advanced-Lane-Lines/blob/master/notebook/detect_lane.ipynb).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![Undistorted sample image][image1]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Undistorted Transformed Image][image2]

Initially, I saved the camera calibration and distortion coefficients in a `pickle` so I can use it whenever. I created a `cal_undistort()` function that takes in an image to be undistorted and camera calibration and distortion coefficients to output the undistorted image. The code is in cell block entitled *Apply a distortion correction to raw images.*, cell 5. The images came out like this: 
![Compared Distorted and Undistorted Images][image2-1]

#### 2. Color transforms, gradients or other methods to create a thresholded binary image.  

I used a combination of color and gradient thresholds to generate a binary image. This step is under cell block entitled *Use color transforms, gradients, etc., to create a thresholded binary image.* I divided this cell block into two: the gradient thresholds (under cell block *Gradients*) and the color thresholds (under cell block *Colorspace*). In the gradients threshold, I would pre-process an image to give the grayscale version, the Sobel X and Sobel Y operators. These I need for the gradient threshold functions: the absolute Sobel, the magnitude and the direction thresholds. In the color threshold, I have the conversion to gray, RGB and HLS colorspace. I manipulated the `Red` channel in the RGB (for the white color of the lanes) and the `Lightness` and `Saturation` channels in the HLS (for the shadows on  the road and the yellow-colored lanes). Then, I combined them and, with a lot of trial and error, was able to get a `combined binary image` as well as a bonus color binary image. The code is under cell block entitled *Apply and combine thresholds*. Here's an example of my output for this step.
![Colored Binary Image and Combined Binary Image][image3]

#### 3. Perspective transform and an example of a transformed image.

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

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
