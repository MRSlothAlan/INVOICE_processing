"""
graph_construction.py

explore ways to represent the graphical connection of nodes
    -> adjacency matrix, with integer scalar input
    -> (node - number of connections) pair

feed:
    graph
"""
import numpy as np
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node


def adjacency_matrix_basic(word_nodes, height, width):
    """
    Represent the invoice layout with classic adjacency matrix
    :param word_nodes:
    :return:
    """
    # adj matrix size: from
    adj_size = word_nodes[-1].id + 1
    adj_matrix = [[0] * adj_size] * adj_size
    ratio = 0.009

    for node in word_nodes:
        # print(node.id, node.top_node_id)
        # need to consider distance
        if node.top_node_dis / height < ratio:
            adj_matrix[node.id][node.top_node_id] = 1
        if node.right_node_dis / width < ratio:
            adj_matrix[node.id][node.right_node_id] = 1
        if node.bottom_node_dis / height < ratio:
            adj_matrix[node.id][node.bottom_node_id] = 1
        if node.left_node_dis / width < ratio:
            adj_matrix[node.id][node.left_node_id] = 1
    for row in adj_matrix:
        print(row)
    return adj_matrix


