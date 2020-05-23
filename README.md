# CItyU_CS4186

This project contains the two assignments of CS4186, computer vision, and image processing, in the City University of Hong Kong.

### Assignment 1: Instance Search
1. 5,000 images, 10 example query instances, and 20 testing query instances. Given the query instance, the target is to find out the most relevant images.
2. Two algorithms are to be implemented. I choose color_histogram and SIFT.
3. **Details in SIFT**
   * ***RANSAC***: filter out outliers. Because the query instance is given, we could treat the transformation as global. Using the RANSAC algorithm directly gives great improvement.
   * ***Data Augmentation***: increase the robustness of the program. Perform 8 different perspective transformations and find out the maximal match number.
   
### Assignment 2: Disparity Map and Relative Height
1. The first part is to explore the grabcut, a semi-auto segmentation algorithm.
2. The second part is to compute the disparity map of three pairs of images and calculate the relative height.
3. **Details of second part**
   1. Step 1: Corner detection. This time, the transformation is no longer global, using RANSAC will introduce undesirable behavior. Instead, using the ratio test is better.
   2. Step 2: Fundamental Matrix. Because the intrinsic and extrinsic parameters are not given, we could only use the 8-point-algorithm to compute the fundamental matrix, which relates two images.
   3. Step 3(Optional): Compute the epipolar line. Given the fundamental matrix, we could calculate the corresponding epipolar line of each corner point.
   4. Step 4: Image Rectification. In order to perform the matching efficiently, it is necessary to re-project two images to a common plane and make the epipolar lines parallel to each other.
   5. Step 5: Disparity Map. It is noted that for ***cv2.StereoSGBM_create*** function, we have to set the ***minDisparity*** argument to a negative value in order to have good performance.
   6. Step 6: Relative Height. After computing the disparity map, we could use the triangulation to compute the relative depth information of two objects and use it to compute the relative height.
