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
import time
from concurrent import futures


class WordFeature:
    def __init__(self):
        self.WIDTH = 0
        self.HEIGHT = 0
        self.DIM = 0

    def is_date(self, string, fuzzy=False):
        try:
            parse(string, fuzzy=fuzzy)
            return True
        except ValueError:
            return False
        except OverflowError:
            return False
        except TypeError:
            return False

    def word_vector_generation(self, word, dim):
        # 08092020
        # only deal with english for now
        # https://pypi.org/project/bpemb/
        # https://nlp.h-its.org/bpemb/
        dim = self.DIM
        bpemb_en = BPEmb(lang="en", vs=10000, dim=dim)
        bpemb_en.encode(word)
        # 09092020: update: just combine all vectors if len(vec_list) > 1
        ids = bpemb_en.encode_ids(word)
        vec_list = bpemb_en.vectors[ids].tolist()
        try:
            final = vec_list[0]
            if len(vec_list) > 1:
                for index, vec in enumerate(vec_list):
                    if index > 0:
                        for i, v in enumerate(vec):
                            final[i] += v
        except IndexError as e:
            final = [0.0] * dim
        return final

    def feature_of_node(self, node_batches):
        """

        :param node:
        :return:
        """
        """
            features:
            boolean features
            (isdate, isalphabetic, isnumeric, iscurrency, mix)
        """
        final_feature_vector_list = list()
        width = self.WIDTH
        height = self.HEIGHT
        dim = self.DIM

        for node in node_batches:
            boolean_feature = [0] * 5
            if self.is_date(node.word):
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

            for index, feat in enumerate(numeric_features):
                try:
                    numeric_features[0] = node.left_node_dis / width
                except ZeroDivisionError:
                    numeric_features[0] = 0
                try:
                    numeric_features[1] = node.top_node_dis / height
                except ZeroDivisionError:
                    numeric_features[1] = 0
                try:
                    numeric_features[2] = node.right_node_dis / width
                except ZeroDivisionError:
                    numeric_features[2] = 0
                try:
                    numeric_features[3] = node.bottom_node_dis / height
                except ZeroDivisionError:
                    numeric_features[3] = 0

            """
            text feature representation
            convert into a meaningful representation
            """
            word_vector_feature = self.word_vector_generation(node.word, dim)
            # yield the final feature vector for processing
            final_feature_vector = boolean_feature + numeric_features + word_vector_feature
            temp_data_tuple = (node.id, final_feature_vector)
            final_feature_vector_list.append(temp_data_tuple)

        return final_feature_vector_list

    def feature_calculation(self, word_node, image=None):
        """
        calculate the word features of the line
        sorry for breaking OOP but it is okay
        :param same_line:
        :return:
        """
        # it is the best to set a dictionary with id : feature pair in order to avoid confusion
        final_dict = dict()

        if image is not None:

            height, width, color = image.shape
            dim = 100
            self.HEIGHT = int(height),
            self.WIDTH = int(width),
            self.DIM = dim

            if type(self.WIDTH) is not int:
                self.WIDTH = self.WIDTH[0]
            if type(self.HEIGHT) is not int:
                self.HEIGHT = self.HEIGHT[0]
            # print(self.WIDTH, self.HEIGHT, self.DIM, width, height)
            # set the global constant in order to use process pool executor
            start = time.time()
            # try to implement parallel processing in the code
            batch_size = 10
            node_batches = [word_node[i: i + batch_size] for i in range(0, len(word_node), batch_size)]
            all_feature_vectors = list()
            with futures.ProcessPoolExecutor(max_workers=5) as pool:
                for feature_vectors in pool.map(self.feature_of_node, node_batches):
                    all_feature_vectors.append(feature_vectors)
            # final_feature_vector = feature_of_node(node, width, height, dim)
            # feature_matrix[node.id] = list()
            # feature_matrix[node.id] = final_feature_vector
            end = time.time()
            print("         feature production requires: {}".format(abs(start - end)))
            # convert batch feature vectors into dict
            for batch_features in all_feature_vectors:
                for word_tuple in batch_features:
                    final_dict[word_tuple[0]] = list()
                    final_dict[word_tuple[0]] = word_tuple[1]
        else:
            print("images not set!")
        return final_dict


