"""
graph_process.py
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import resize_ratio
import cv2


def get_list_of_neighbors(words_raw, node):
    # looking for neighbors
    list_of_neighbors = [node_nei for node_nei in
                         words_raw if
                         node_nei.id == node.top_node_id or
                         node_nei.id == node.right_node_id or
                         node_nei.id == node.left_node_id or
                         node_nei.id == node.bottom_node_id]
    return list_of_neighbors



