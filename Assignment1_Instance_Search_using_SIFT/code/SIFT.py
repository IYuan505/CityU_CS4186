import numpy as np
import cv2


def detect_data(data):
    # detect corners of images inside the database
    # use SIFT to describe the feature
    cnt = 0
    # Initiate SIFT detector
    corner_detector = cv2.xfeatures2d.SIFT_create()
    for data_number in data:
        print(cnt / len(data))
        cnt += 1
        data_img = data[data_number]['img']
        # find the keypoints and descriptors with SIFT
        kp, des = corner_detector.detectAndCompute(data_img, None)
        data[data_number]['kp'] = kp
        data[data_number]['des'] = des
        del data[data_number]['img']
    return data


def data_aug(img):
    # perspective change to augment query data
    transformed_imgs = [img]

    height, width, channel = img.shape
    pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # eight different perspective transformations
    pts1 = np.float32([[0, 0], [0.8 * width, height * 0.025], [0.8 * width, 0.975 * height], [0, height]])
    pts2 = np.float32([[0.2 * width, 0.025 * height], [width, 0], [width, height], [0.2 * width, 0.975 * height]])
    pts3 = np.float32([[0, 0], [width, 0], [0.975 * width, 0.8 * height], [0.025 * width, 0.8 * height]])
    pts4 = np.float32([[0.025 * width, 0.2 * height], [0.975 * width, 0.2 * height], [width, height], [0, height]])

    pts5 = np.float32([[0, 0], [0.6 * width, height * 0.1], [0.6 * width, 0.9 * height], [0, height]])
    pts6 = np.float32([[0.4 * width, 0.1 * height], [width, 0], [width, height], [0.4 * width, 0.9 * height]])
    pts7 = np.float32([[0, 0], [width, 0], [0.9 * width, 0.6 * height], [0.1 * width, 0.6 * height]])
    pts8 = np.float32([[0.1 * width, 0.4 * height], [0.9 * width, 0.4 * height], [width, height], [0, height]])

    all_target_pts = [pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8]

    for idx, target in enumerate(all_target_pts):
        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(pts, target)
        transformed = cv2.warpPerspective(img, M, (width, height))
        transformed_imgs.append(transformed)
    return transformed_imgs


def SIFT_perspective_change(query_data, data, test_mode=False):
    # SIFT
    # Initiate SIFT detector
    corner_detector = cv2.xfeatures2d.SIFT_create()

    # use brute force matcher with crossCheck=True
    # crossCheck is an alternative of ratio test
    bf = cv2.BFMatcher(crossCheck=True)

    result = {}
    if test_mode:
        query_data = {'01': query_data['01']}
    for query_number in query_data:
        print('Processing: Q' + query_number)
        query_img = query_data[query_number]['img']
        # query image goes through the augmentation, 9 images including the original one are returned
        augmented_query_img = data_aug(query_img)
        # initialize the score of each image inside the database
        score_for_each = {}
        for data_number in data:
            score_for_each[data_number] = 0
        for img in augmented_query_img:
            kp, des = corner_detector.detectAndCompute(img, None)
            for data_number in data:
                des_data = data[data_number]['des']
                kp_data = data[data_number]['kp']
                if des_data is not None:
                    matches = bf.match(des, des_data)

                    # Use homography to filter out incorrect matches
                    src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_data[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    # The score of the similarity between each query image and each image inside the
                    # database is assigned as the maximal number of correct matches.
                    score_for_each[data_number] = max(score_for_each[data_number], np.sum(mask))

        # sort the images inside the database according to the similarity score
        sorted_match_score = {k: v for k, v in
                              sorted(score_for_each.items(), key=lambda item: item[1], reverse=True)}

        rank_list = [str(int(k)) for k in sorted_match_score]

        result[query_number] = rank_list
    return result

