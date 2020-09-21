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


def get_linked_rows(raw):
    """
    Get rows based on right neighbors
    Assume that pointers are set in each node
    :param raw:
    :return:
    """
    parsed_node_id = set()
    all_lines_in_block = list()
    for node in raw:
        line = list()
        line.append(node)
        temp = node
        # use id to check any recursive loop occurs?
        scanned_nodes_id = dict()
        scanned_nodes_id[temp.id] = 1

        if node.id not in parsed_node_id:
            parsed_node_id.add(node.id)

            while temp.right_node_ptr is not None:
                temp = temp.right_node_ptr
                parsed_node_id.add(temp.id)
                try:
                    scanned_nodes_id[temp.id] += 1
                except KeyError:
                    scanned_nodes_id[temp.id] = 1
                if scanned_nodes_id[temp.id] > 1:
                    # this successful solve the recursive linkage logic error
                    # by checking the unique ids
                    break
                # append the original row
                line.append(temp)
            all_lines_in_block.append(line)
    return all_lines_in_block


def merge_nearby_node_info_process(node_line, width):
    """
    Given a node_line, output a merged line
    :param node_line:
    :return:
    """
    # merge based on label, distance
    thresh_distance = int(width / 320)
    index = 0
    result = list()
    # do not skip the last one
    while index < len(node_line):
        curr_index = index
        """
        plan: for the word, if the gap is less than 5, merge them
        """
        temp_word_num = node_line[index].word_num
        temp_str = node_line[index].word + " "

        temp_width = node_line[index].width

        try:
            while index < len(node_line) and node_line[index + 1].left - \
                    node_line[index].left - node_line[index].width < \
                    thresh_distance:
                temp_width += node_line[index + 1].width + thresh_distance
                temp_word_num += node_line[index + 1].word_num + 1
                temp_str += node_line[index + 1].word + " "
                index += 1
                if index >= len(node_line) - 1:
                    break
        except IndexError as e:
            pass
        # create a new node based on this
        merged_node = Node(word=temp_str,
                           left=node_line[curr_index].left,
                           top=node_line[curr_index].top,
                           width=temp_width,
                           height=node_line[curr_index].height,
                           word_num=temp_word_num,
                           sub_word_num=0)
        result.append(merged_node)
        index += 1

    return result

