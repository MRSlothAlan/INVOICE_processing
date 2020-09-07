"""
parse data sets

based on yolo-labels

data set:
    1. blur lines on invoice, crop different sub-regions
    2. area of region
    3. location of region
    4. word embedding (not sure)

target field of model:
    1. address
    2. line items
    3. header
    4. some fields as others e.g. invoice number, invoice date, etc.. (handled by RE mostly)
"""
from pathlib import Path
from os import listdir
import os
import cv2
from whitening import whiten
import PIL.Image
import numpy as np
import xml.etree.ElementTree as ET


def deginrad(degree):
    radiant = 2*np.pi/360 * degree
    return radiant


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    list_with_labels = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text
        label = boxes.find('name').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        list_with_labels.append(label)

    return filename, list_with_all_boxes, list_with_labels


image_data_set_dir = Path.cwd().parents[3] / "0_RAW_DATASETS/all_raw_images"
label_data_set_dir = Path.cwd().parents[3] / "0_RAW_DATASETS/all_raw_images_labels"
processed_dir = Path.cwd() / "processed"

# load all class labels
with open(str(label_data_set_dir / "classes.txt"), mode="+r", encoding="utf-8") as label_class_f:
    classes_labels = label_class_f.read()
    label_class_f.close()

classes_labels_list = classes_labels.split("\n")
classes_index_label = dict()

for index, label in enumerate(classes_labels_list):
    classes_index_label[index] = str()
    classes_index_label[index] = label

"""
get pre-defined labels
"""
print(classes_index_label)

for label_file_name in listdir(str(label_data_set_dir)):
    if label_file_name.split('.')[-1] == "xml" and not label_file_name.__contains__("classes"):
        # load the image and labels, which is in yolo format
        print("\nProcessing: ", label_file_name)
        processed_image_dir = "C:/processed/" + label_file_name.rsplit('.', 1)[0]
        if not os.path.exists(str(processed_image_dir)):
            os.makedirs(str(processed_image_dir))
        image_file_name = label_file_name.rsplit('.', 1)[0] + ".jpg"
        image = cv2.imread(str(image_data_set_dir / image_file_name))
        h_ori, w_ori, c_ori = image.shape
        print("Shape: ", image.shape)
        """
        pre-process images, e.g. whitening, binarisation
        """
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image_blur = cv2.medianBlur(image_gray, 5)
        kernel = np.ones((1, 1), np.uint8)
        image_blur = cv2.erode(image_gray, kernel, iterations=1)
        image_blur = cv2.cvtColor(image_blur, cv2.COLOR_GRAY2BGR)
        # whiten images
        image_foreground, image_background = whiten(image_blur, kernel_size=20, downsample=4)
        image_pil = PIL.Image.fromarray(image_foreground)
        image = np.array(image_pil)
        image = image[:, :, ::-1].copy()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        """
        filter vertical lines in images
        try out filtered gray
        """
        theta = deginrad(0)
        g_kernel = cv2.getGaborKernel((9, 9), 8, theta, 5, 0.5, 0, ktype=cv2.CV_32F)
        filtered_gray = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
        # cv2.imwrite(str(processed_image_dir) + ("/filtered_" + image_file_name), filtered_gray)

        # threshold the grayscale image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        """
        test filter out vertical lines
        """
        test_param_ratio = [30]
        image_hor = image.copy()
        for r in test_param_ratio:
            thresh_copy = thresh.copy()
            kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h_ori / r)))
            # morph_ver = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_ver)
            thresh_copy = cv2.erode(thresh_copy, kernel_ver)
            cv2.bitwise_not(image_hor, image_hor, mask=thresh_copy)

        # cv2.imwrite(str(processed_image_dir) + ("/morphology_" + image_file_name), thresh_copy)
        """
        Update the values
        """
        gray = cv2.cvtColor(image_hor, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imwrite(str(processed_image_dir) + ("/morphology_" + image_file_name), gray)

        # use morphology erode to blur horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w_ori / 30), 3))
        # just think of it as extending the line of characters
        morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

        # use morphology open to remove thin lines from dotted lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 17))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        # find contours
        cntrs = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        # find the topmost box
        ythresh = 1000000
        for c in cntrs:
            box = cv2.boundingRect(c)
            x, y, w, h = box
            if y < ythresh:
                topbox = box
                ythresh = y

        # Draw contours excluding the topmost box
        result = image.copy()
        for c in cntrs:
            box = cv2.boundingRect(c)
            if box != topbox:
                x, y, w, h = box
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.destroyAllWindows()
        """
        crop images from results
        """
        height_i, width_i, color = result.shape
        print("Result shape: ", result.shape)

        name, boxes, labels_xml = read_content(str(label_data_set_dir / label_file_name))
        for index, box in enumerate(boxes):
            if labels_xml[index] == "address":
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                print(box)
                result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                print(labels_xml[index])
            elif labels_xml[index] == "line item":
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                print(box)
                result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
                print(labels_xml[index])
            elif labels_xml[index] == "header":
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                print(box)
                result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (0, 100, 100), 3)
                print(labels_xml[index])
            elif not labels_xml[index] == "table content":
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                print(box)
                result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                print(labels_xml[index])

        print(str(processed_image_dir) + ("/gray_" + image_file_name))
        cv2.imwrite(str(processed_image_dir) + ("/gray_" + image_file_name), result)



