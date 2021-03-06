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
    adj_matrix = np.array(adj_matrix)
    ratio = 0.009

    for node in word_nodes:
        # need to consider distance
        # if node.top_node_dis / height < ratio:
        if node.id != node.top_node_id:
            adj_matrix[node.id][node.top_node_id] = 1
            adj_matrix[node.top_node_id][node.id] = 1
        # if node.right_node_dis / width < ratio:
        if node.id != node.right_node_id:
            adj_matrix[node.id][node.right_node_id] = 1
            adj_matrix[node.right_node_id][node.id] = 1
        # if node.bottom_node_dis / height < ratio:
        if node.id != node.bottom_node_id:
            adj_matrix[node.id][node.bottom_node_id] = 1
            adj_matrix[node.bottom_node_id][node.id] = 1
        # if node.left_node_dis / width < ratio:
        if node.id != node.left_node_id:
            adj_matrix[node.id][node.left_node_id] = 1
            adj_matrix[node.left_node_id][node.id] = 1

    adj_matrix_list = adj_matrix.tolist()

    return adj_matrix_list


