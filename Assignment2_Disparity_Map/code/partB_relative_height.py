import numpy as np
import cv2


def corner_detection(img1, img2):
    # use SIFT to do corner detection
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def fundamental_matrix_estimation(pts1, pts2):
    # find the fundamental matrix
    # Use RANSAC algorithm
    # 2 -> the maximum distance from a point to an epipolar line in pixels
    # 0.999 -> the level of confidence
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 2, 0.9999)

    # Select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return pts1, pts2, F


def epipolar_lines(file_number, img1_ori, img2_ori, pts1, pts2, F, show_epi_result=False, write_result=False,
                   pts_number=4):
    # random sampling pts_number pairs of matches
    index = np.random.choice(pts1.shape[0], pts_number, replace=False)
    pts1 = pts1[index]
    pts2 = pts2[index]
    img1 = img1_ori.copy()
    img2 = img2_ori.copy()

    def drawlines(img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        row, col, channel = img1.shape
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [col, -(r[2] + r[0] * col) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 3)
            img1 = cv2.circle(img1, tuple(pt1), 6, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 6, color, -1)
        return img1, img2

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_epi, img2_corner = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_epi, img1_corner = drawlines(img2, img1, lines2, pts2, pts1)

    epi_result = np.hstack((img1_epi, img2_epi))
    if show_epi_result:
        cv2.imshow("epipolar_result", epi_result)
        cv2.waitKey(0)
    if write_result:
        cv2.imwrite("relative_height_result/epipolar_line/{}.jpg".format(file_number), epi_result)


def image_rectification(file_number, img1, img2, pts1, pts2, F, show_rect_result=False, write_result=False):
    height, width, channel = img1.shape
    rect_canvas_size = (width, height)
    retBool, rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, rect_canvas_size)

    img1_rect = cv2.warpPerspective(img1, rectmat1, rect_canvas_size)
    img2_rect = cv2.warpPerspective(img2, rectmat2, rect_canvas_size)

    if show_rect_result:
        rect_result = np.hstack((img1_rect, img2_rect))
        cv2.imshow("rect_result", rect_result)
        cv2.waitKey(0)

    if write_result:
        cv2.imwrite("relative_height_result/rectified_images/{}_a.jpg".format(file_number), img1_rect)
        cv2.imwrite("relative_height_result/rectified_images/{}_b.jpg".format(file_number), img2_rect)

    return img1_rect, img2_rect


def disparity_map(file_number, img1, img2, show_result=False, write_result=False):
    if file_number == 1:
        blockSize = 3
        stereo = cv2.StereoSGBM_create(minDisparity=-32,
                                       numDisparities=64,
                                       blockSize=blockSize,
                                       P1=8 * 3 * blockSize * blockSize,
                                       P2=32 * 3 * blockSize * blockSize,
                                       disp12MaxDiff=1,
                                       preFilterCap=15,
                                       uniquenessRatio=5,
                                       speckleWindowSize=50,
                                       speckleRange=2,
                                       mode=1
                                       )
    elif file_number == 2:
        blockSize = 7
        stereo = cv2.StereoSGBM_create(minDisparity=-128,
                                       numDisparities=256,
                                       blockSize=blockSize,
                                       P1=4 * 3 * blockSize * blockSize,
                                       P2=64 * 3 * blockSize * blockSize,
                                       disp12MaxDiff=1,
                                       # preFilterCap=15,
                                       uniquenessRatio=5,
                                       # speckleWindowSize=100,
                                       # speckleRange=2,
                                       mode=1
                                       )

    elif file_number == 3:
        blockSize = 3
        stereo = cv2.StereoSGBM_create(minDisparity=-16,
                                       numDisparities=32,
                                       blockSize=blockSize,
                                       P1=16 * 3 * blockSize * blockSize,
                                       P2=32 * 3 * blockSize * blockSize,
                                       disp12MaxDiff=2,
                                       preFilterCap=15,
                                       uniquenessRatio=5,
                                       speckleRange=2,
                                       mode=1
                                       )
    disp = stereo.compute(img1, img2) / 16
    disp = ((disp - np.min(disp)) * 255 / (np.max(disp) - np.min(disp))).astype('uint8')
    if show_result:
        cv2.imshow("disp", disp)
        cv2.waitKey(0)
    if write_result:
        cv2.imwrite("relative_height_result/disparity/{}.jpg".format(file_number), disp)
    return disp


def grabcut(img_ori, rect, name, manual_mask=None, show_result=False):
    img = img_ori.copy()
    # initialize mask, background model, foreground model
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    if manual_mask is not None:
        # if a manual marked mask are given to the image
        # wherever it is marked white (sure foreground), change mask=1
        # wherever it is marked black (sure background), change mask=0
        mask[manual_mask == 0] = 0
        mask[manual_mask == 255] = 1
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    # change mask=0 (sure background), mask=2 (probable background) into value 0
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask[:, :, np.newaxis]
    if show_result:
        cv2.imshow(name, img)
        cv2.waitKey(0)
    return mask


def find_mask_two_object(file_number, img, show_result=False):
    if file_number == 1:
        traffic_cone_rect_left = (110, 255, 90, 155)
        traffic_cone_rect_right = (200, 250, 75, 125)
        img_name = 'traffic_cone'
        traffic_cone_mask = cv2.imread("relative_height_result/mask/1_a.png", 0)
        mask1 = grabcut(img, traffic_cone_rect_left, img_name, traffic_cone_mask, show_result=show_result)
        mask2 = grabcut(img, traffic_cone_rect_right, img_name, show_result=show_result)
    elif file_number == 2:
        bottle_rect_left = (235, 580, 300, 350)
        bottle_rect_right = (570, 160, 400, 600)
        img_name = 'bottle'
        bottle_mask = cv2.imread("relative_height_result/mask/2_a.png", 0)
        mask1 = grabcut(img, bottle_rect_left, img_name, show_result=show_result)
        mask2 = grabcut(img, bottle_rect_right, img_name, bottle_mask, show_result=show_result)
    elif file_number == 3:
        statue_rect_left = (175, 75, 240, 350)
        stone_rect_right = (745, 320, 125, 125)
        img_name = 'statue_stone'
        statue_stone_mask = cv2.imread("relative_height_result/mask/3_a.png", 0)
        mask1 = grabcut(img, statue_rect_left, img_name, statue_stone_mask, show_result=show_result)
        mask2 = grabcut(img, stone_rect_right, img_name, show_result=show_result)
    return mask1, mask2


def relative_height(file_number, disp, mask1, mask2):
    # pixel height measured from the rectified images
    if file_number == 1:
        pixel_height1 = 130
        pixel_height2 = 105
    if file_number == 2:
        pixel_height1 = np.sqrt(70 ** 2 + 270 ** 2)
        pixel_height2 = np.sqrt(140 ** 2 + 540 ** 2)
    if file_number == 3:
        pixel_height1 = 325
        pixel_height2 = 105
    '''
    disparity = B*f/Z, where Z is the depth
    disparity1/disparity2 = Z2/Z1
    real_height = pixel_height/focal_length*Z
    real_height1/real_height2 = pixel_height1*Z1/(pixel_height2*Z2)
    '''

    def select_representative(disp):
        values = disp[disp > 0]
        # select median as the representative disparity value
        # more robust against noise
        representative = np.median(values)
        return representative

    # first find out the representative of the disparity of the left object
    disparity1 = select_representative(disp * mask1)
    disparity2 = select_representative(disp * mask2)
    '''
    # Result of the real height ratio
    # image 1, two traffic cones: 1.0146
    # image 2, two bottles: 0.3767
    # image 3, statue and stone: 2.4182
    '''
    real_height_ratio = pixel_height1 / pixel_height2 * (disparity2 / disparity1)
    print("%.4f" % real_height_ratio)


def main():
    file_number = 1
    show_result = False
    write_result = False
    test_mode = False
    if test_mode:
        img1 = cv2.imread("test/scene1.row3.col1.ppm")
        img2 = cv2.imread("test/scene1.row3.col5.ppm")
    else:
        img1 = cv2.imread("relative_height/{}_a.jpg".format(file_number))
        img2 = cv2.imread("relative_height/{}_b.jpg".format(file_number))

    pts1, pts2 = corner_detection(img1, img2)
    pts1, pts2, F = fundamental_matrix_estimation(pts1, pts2)

    epipolar_lines(file_number, img1, img2, pts1, pts2, F,
                   show_epi_result=show_result, write_result=write_result)
    img1_rect, img2_rect = image_rectification(file_number, img1, img2, pts1, pts2, F,
                                               show_rect_result=show_result, write_result=write_result)
    disp = disparity_map(file_number, img1_rect, img2_rect, show_result=show_result, write_result=write_result)
    mask1, mask2 = find_mask_two_object(file_number, img1_rect, show_result=show_result)
    relative_height(file_number, disp, mask1, mask2)


if __name__ == "__main__":
    main()
