"""

basic data operation
"""


def load_class_list(label_data_set_dir):
    # load all class labels
    with open(str(label_data_set_dir / "classes.txt"), mode="+r", encoding="utf-8") as label_class_f:
        classes_labels = label_class_f.read()
        label_class_f.close()
    classes_labels_list = classes_labels.split("\n")

    return classes_labels_list

