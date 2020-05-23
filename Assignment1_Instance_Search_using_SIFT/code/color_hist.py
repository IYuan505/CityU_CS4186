import numpy as np


def convert_to_bin(number, range_start, range_end, bin):
    # number is the real value of the input
    # range_start is the minimal value of the range
    # range_end is the maximal value of the range
    # bin is the number of bins
    range_per_bin = (range_end - range_start) / bin
    current_nubmer = range_start
    for i in range(bin):
        current_nubmer += range_per_bin
        if current_nubmer > number:
            return i


def img2bin(img):
    # this function convert one image into color histogram
    row, col, channel = img.shape
    converted_img = np.zeros(shape=(row, col))
    for i in range(row):
        for j in range(col):
            # Swain and Ballard’s Histogram Matching
            # channel order: BGR
            # wb = R+G+B
            # rg = R-G
            # by = 2B-R-G
            wb = np.sum(img[i, j])
            converted_wb = convert_to_bin(wb, 0, 256 * 3, 8)
            rg = int(img[i, j, 2]) - int(img[i, j, 1])
            converted_rg = convert_to_bin(rg, -256, 256, 16)
            by = 2 * int(img[i, j, 0]) - int(img[i, j, 2]) - int(img[i, j, 1])
            converted_by = convert_to_bin(by, -512, 512, 16)
            converted_img[i, j] = converted_wb * 16 * 16 + converted_rg * 16 + converted_by
    bin, _ = np.histogram(converted_img, bins=8 * 16 * 16, range=[0, 8 * 16 * 16])
    return bin


def convert_data(data):
    # This function convert all images into color histograms
    cnt = 0
    for data_number in data:
        print(cnt / len(data))
        cnt += 1
        data_img = data[data_number]['img']
        data_bin = img2bin(data_img)
        data[data_number]['bin'] = data_bin
        # delete the original image matrix to save space
        del data[data_number]['img']
    return data


def color_hist(query_data, data):
    # This function compute the similarity between the query image and images in the database
    result = {}
    for query_number in query_data:
        query_img = query_data[query_number]['img']
        query_bin = img2bin(query_img)
        score_for_each = {}
        for data_number in data:
            # compute the intersection and normalize the value
            intersection = np.minimum(query_bin, data[data_number]['bin'])
            sum_intersection = np.sum(intersection)
            match_score = sum_intersection / np.sum(query_bin)
            score_for_each[data_number] = match_score
        # sort the images inside the database according to the similarity score
        sorted_match_score = {k: v for k, v in sorted(score_for_each.items(), key=lambda item: item[1], reverse=True)}
        rank_list = [str(int(k)) for k in sorted_match_score]
        result[query_number] = rank_list
    return result
