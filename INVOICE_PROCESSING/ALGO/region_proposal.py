"""
region_proposal.py
blur the image, then propose regions
"""
import cv2
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
import imutils
import numpy as np


def blur_images(image):
    """
    Apply transformation to image
    :param image:
    :return:
    """
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray_scale.shape
    gray_resize = cv2.resize(gray_scale, (int(w * resize_ratio), int(h * resize_ratio)))
    ori_resize = cv2.resize(image, (int(w * resize_ratio), int(h * resize_ratio)))

    blank = np.zeros(ori_resize.shape, np.uint8)
    blank[:] = (255, 255, 255)
    gray_original = gray_resize.copy()
    gray_resize_second = gray_resize.copy()

    # pixel is not BGR channel, is 1D color, from 0(black) to 255(white)
    for index_r, row in enumerate(gray_resize):
        for index_p, pixel in enumerate(row):
            if int(pixel) != 255:
                # a threshold to parse it
                if int(pixel) < 200:
                    # most likely gray color to black
                    # 'blur' the surroundings
                    if 0 < index_p < len(row) - 1:
                        row[index_p - 1] = 0
                        row[index_p + 1] = 0

    # create another mask
    for index_r, row in enumerate(gray_resize_second):
        row_len = len(row) - 1
        for index_p, pixel in enumerate(reversed(row)):
            if int(pixel) != 255:
                if int(pixel) < 200:
                    temp_index = row_len - index_p
                    if 0 < temp_index < len(row) - 1:
                        row[temp_index + 1] = 0
                        row[temp_index - 1] = 0
    # if len([p for p in row if int(p) != 255]) > 1:
    #         print(len([p for p in row if int(p) != 255]))
    (thresh_one, im_bw_one) = cv2.threshold(gray_resize, 20, 255, cv2.THRESH_BINARY)
    (thresh_two, im_bw_two) = cv2.threshold(gray_resize_second, 20, 255, cv2.THRESH_BINARY)

    unique_v = set()
    for r in im_bw_one:
        for p in r:
            unique_v.add(p)

    im_bw_three = im_bw_two
    for index_r, row in enumerate(im_bw_one):
        for index_p, pixel in enumerate(row):
            if int(im_bw_one[index_r][index_p]) != int(im_bw_two[index_r][index_p]):
                im_bw_three[index_r][index_p] = 255

    return im_bw_three


def propose_regions(image_p):
    list_of_bounding_rect = list()
    # extract image parts
    cnts, hierarchy = cv2.findContours(image_p, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    height, width = image_p.shape
    for c in cnts:
        # for point in c:
        # print(point)
        x, y, w, h = cv2.boundingRect(c)
        if abs(h - y) != height and abs(w - x) != width:
            # if SHOW_IMAGE:
            # cv2.rectangle(image_p, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # cv2.drawContours(image_p, c, -1, (0, 255, 0), 1)

            list_of_bounding_rect.append([x, y, w, h])
    # sort by height

    def insertion_sort(rect_list):
        for i in range(1, len(rect_list)):
            key = rect_list[i][1]
            j = i - 1
            while j >= 0 and key < rect_list[j][1]:
                temp = rect_list[j + 1]
                rect_list[j + 1] = rect_list[j]
                rect_list[j] = temp
                j -= 1
            rect_list[j + 1][1] = key
        return rect_list
    final_list = insertion_sort(list_of_bounding_rect)
    if SHOW_IMAGE:
        cv2.imshow("clusters", image_p)
    return final_list


def region_proposal(image):
    """
    This script can be used for proposing regions.
    You can use it for machine learning, or just pure region extraction
    :param image:
    :return:
    """
    image_processed = blur_images(image)
    list_rect = propose_regions(image_processed)
    return list_rect



