"""
Word feature calculation

calculate the features of all the words in the graph
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine
from dateutil.parser import parse
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from bpemb import BPEmb
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
import numpy as np
import multiprocessing as mp
import sys


def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False
    except OverflowError:
        return False
    except TypeError:
        return False


def word_vector_generation(word):
    # 08092020
    # only deal with english for now
    # https://pypi.org/project/bpemb/
    # https://nlp.h-its.org/bpemb/
    bpemb_en = BPEmb(lang="en", vs=10000, dim=50)
    bpemb_en.encode(word)
    if type(word) is not list():
        word = [word]
    # only pick the first sub-word for simplicity
    word_check = word[0]
    ids = bpemb_en.encode_ids(word_check)
    vec_set = bpemb_en.vectors[ids]
    try:
        vec_list = vec_set[0].tolist()
    except IndexError as e:
        vec_list = [0.0] * 100
    return vec_list


def feature_of_node(node, width, height):
    """

    :param node:
    :return:
    """
    """
        features:
        boolean features
        (isdate, isalphabetic, isnumeric, iscurrency, mix)
    """
    boolean_feature = [0] * 5
    if is_date(node.word):
        boolean_feature[0] = 1
    # check is currency?
    # implement a parser later, now just detect , and . and digits
    if str(node.word).__contains__(".") or str(node.word).__contains__(","):
        if str(node.word)[-1] is not "." and str(node.word)[-1] is not ",":
            if len([c for c in str(node.word) if c.isdigit()]) > 0:
                boolean_feature[3] = 1

    # only number / mostly numbers?
    if len(node.word) > 0:

        if len([c for c in str(node.word) if c.isdigit()]) / len(str(node.word)) >= 0.8:
            boolean_feature[2] = 1
        elif len([c for c in str(node.word) if c.isalpha()]) / len(str(node.word)) >= 0.9:
            boolean_feature[1] = 1
        else:
            boolean_feature[4] = 1

    """
    numeric feature:
    relative distance to the nearest neighbors
    [left, top, right, bottom]
    """
    numeric_features = [0.0] * 4
    numeric_features[0] = node.left_node_dis / width
    numeric_features[1] = node.top_node_dis / height
    numeric_features[2] = node.right_node_dis / width
    numeric_features[3] = node.bottom_node_dis / height
    """
    text feature representation
    convert into a meaningful representation
    """
    word_vector_feature = word_vector_generation(node.word)
    # yield the final feature vector for processing
    final_feature_vector = boolean_feature + numeric_features + word_vector_feature
    return final_feature_vector


def feature_calculation(same_line=SameLine(), image=None):
    """
    calculate the word features of the line
    sorry for breaking OOP but it is okay
    :param same_line:
    :return:
    """
    # it is the best to set a dictionary with id : feature pair in order to avoid confusion
    feature_matrix = dict()

    if image is not None:

        height, width, color = image.shape

        for key in same_line.storage:
            for key_line in same_line.storage[key]:
                """
                results = [pool.apply(feature_of_node, args=(node, width, height)) for node
                                in same_line.storage[key][key_line]]

                print(results)
                """
                for node in same_line.storage[key][key_line]:
                    final_feature_vector = feature_of_node(node, width, height)
                    feature_matrix[node.id] = list()
                    feature_matrix[node.id] = final_feature_vector
    else:
        print("images not set!")
    return feature_matrix


