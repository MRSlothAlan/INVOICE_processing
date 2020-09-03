"""
parse data sets

based on yolo-labels

data set:
    1. blur lines on invoice, crop different sub-regions
    2. area of region
    3. location of region
    4. word embedding (not sure)

"""
from pathlib import Path
from os import listdir
import cv2

image_data_set_dir = Path.cwd().parents[3] / "0_RAW_DATASETS/all_raw_images"
label_data_set_dir = Path.cwd().parents[3] / "0_RAW_DATASETS/all_raw_images_labels"

# load all class labels
with open(str(label_data_set_dir / "classes.txt"), mode="+r", encoding="utf-8") as label_class_f:
    classes_labels = label_class_f.read()
    label_class_f.close()

classes_labels_list = classes_labels.split("\n")
classes_index_label = dict()

for index, label in enumerate(classes_labels_list):
    classes_index_label[index] = str()
    classes_index_label[index] = label

print(classes_index_label)

for label_file_name in listdir(str(label_data_set_dir)):
    if label_file_name.split('.')[-1] == "txt" and not label_file_name.__contains__("classes"):
        # load the image and labels, which is in yolo format
        print(label_file_name)
        image_file_name = label_file_name.rsplit('.', 1)[0] + ".jpg"
        image = cv2.imread(str(image_data_set_dir / image_file_name))
        label_content = str()
        with open(str(label_data_set_dir / label_file_name), mode="+r", encoding='utf-8') as label_f:
            label_content = label_f.read()
            label_f.close()
        print(label_content)
        # pre-process images, e.g. whitening, binarisation

