"""
node_label_matrix.py
return node label matrix
"""
import xml.etree.ElementTree as ET
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node


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


def node_label_matrix(word_node, classes=list(), label_data_set_dir="", label_file_name=""):
    """
    given word nodes (Node type) and classes labels, and bounding boxes
    return the N * E matrix for multiplication
    :param word_node:
    :param classes:
    :return:
    """
    node_label_matrix_dict = dict()
    length = len(classes)
    print("TOTAL NUMBER OF RAW NODE: {}".format(len(word_node)))

    name, boxes, labels_xml = read_content(str(label_data_set_dir / label_file_name))
    # efficiently check whether nodes are in bounding boxes?
    for index, box in enumerate(boxes):
        # each box contains >= 1 words
        result_temp = list()
        if len(word_node) > 0:
            scanned_node = list()
            # [xmin, ymin, xmax, ymax]
            for index_n, node in enumerate(word_node):
                if box[0] < node.center_x < box[2] and box[1] < node.center_y < box[3]:

                    # print(node.word, labels_xml[index])
                    # set the label of the node
                    # using a one-hot encoding way
                    one_hot_class = [0] * length
                    one_hot_class[classes.index(labels_xml[index])] = 1
                    node_label_matrix_dict[node.id] = list()
                    node_label_matrix_dict[node.id] = one_hot_class
                else:
                    result_temp.append(node)

        word_node = result_temp
    # handle the unlabelled nodes
    for index_r, node_remaining in enumerate(word_node):
        one_hot_class = [0] * length
        one_hot_class[classes.index('')] = 1
        node_label_matrix_dict[node_remaining.id] = list()
        node_label_matrix_dict[node_remaining.id] = one_hot_class

    return node_label_matrix_dict
