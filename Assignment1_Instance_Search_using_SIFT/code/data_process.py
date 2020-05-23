import cv2
import os
import glob
import re
import pickle


def read_from_img(data_dir, file_path=None, test_mode=False):
    # read all images and bounding boxes into "data"
    data = {}
    img_path = os.path.join(data_dir, '*jpg')
    img_files = glob.glob(img_path)
    if test_mode:
        # if test_mode is true, only read first 10 images
        img_files = img_files[:10]
    for img_file in img_files:
        img = cv2.imread(img_file)
        img_number = re.search('(\d+)', img_file).group(1)
        bouding_box_file = os.path.join(data_dir, img_number + '.txt')
        if os.path.isfile(bouding_box_file):
            # if bounding box infomation is given, read the information and crop the image
            f = open(bouding_box_file, 'r').read()
            info = re.split('\s', f)
            x = int(info[0])
            y = int(info[1])
            width = int(info[2])
            height = int(info[3])
            info_dict = {'x': x, 'y': y, 'width': width, 'height': height}
            img = img[y:y + height, x:x + width]
            data[img_number] = {'img': img, 'info': info_dict}
        else:
            data[img_number] = {'img': img}

    if (not test_mode) and file_path:
        # write the "data" dictionary to the file
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
            f.close()
    return data


def read_from_file(file_path):
    # read the "data" dictionary directly from the file
    # This is for speed up
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data


def write_result(result, file_path):
    # write the ranking result into rankList file
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[0])}
    write_str = ''
    for query_number in result:
        query_str = 'Q' + str(int(query_number)) + ': '
        result_str = ' '.join(result[query_number])
        write_str += (query_str+result_str+'\n')
    with open(file_path,'w') as f:
        f.write(write_str)
