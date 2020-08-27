"""
graph_construction.py

explore ways to represent the graphical connection of nodes
    -> adjacency matrix, with integer scalar input
    -> (node - number of connections) pair

feed:
    graph
"""
import numpy as np


def adjacency_matrix_basic(word_nodes):
    """
    Represent the invoice layout with classic adjacency matrix
    :param word_nodes:
    :return:
    """
    # adj matrix size: from
    adj_size = word_nodes[-1].id + 1
    adj_matrix = [[0] * adj_size] * adj_size
    for node in word_nodes:
        # print(node.id, node.top_node_id)
        adj_matrix[node.id][node.top_node_id] += 1
        adj_matrix[node.id][node.right_node_id] += 1
        adj_matrix[node.id][node.bottom_node_id] += 1
        adj_matrix[node.id][node.left_node_id] += 1

    # print(np.array(adj_matrix))
    return adj_matrix

