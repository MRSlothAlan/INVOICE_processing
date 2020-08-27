"""
Find specific information from the invoices,
e.g. invoice number, invoice date, total
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import resize_ratio, SHOW_IMAGE
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.graph_process import get_list_of_neighbors
import cv2
# import all the predefined rules
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.rules import *


def find_temp(words_raw_new, resize_temp, resize_ratio=resize_ratio, what=""):
    for node in words_raw_new:
        if what == "total":
            if node.word.lower() == "total" or str(node.word).lower().__contains__('total'):
                # looking for neighbors
                list_of_neighbors = get_list_of_neighbors(words_raw_new, node)
                for node_nei_con in list_of_neighbors:
                    # if is_total_amount_grand(node_nei_con, node):
                        # draw specific arrow
                        resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                            int(node.center_y * resize_ratio)),
                                                            (int(node_nei_con.center_x * resize_ratio),
                                                            int(node_nei_con.center_y * resize_ratio)),
                                                            (0, 0, 255), 2)
        elif what == "invoice_no":
            """
            possible patterns: invoice no, invoice number, No.:
            """
            string_to_watch = str(node.word).lower().strip()
            if (string_to_watch.__contains__('invoice') and string_to_watch.__contains__("no"))\
                    or (string_to_watch.__contains__("no") and not string_to_watch.__contains__("invoice"))\
                    or (string_to_watch.__contains__("invoice") and string_to_watch.__contains__("num")):
                if 0 < len(str(node.word).lower().strip().split(' ')) < 3:
                    # looking for neighbors
                    list_of_neighbors = get_list_of_neighbors(words_raw_new, node)
                    for node_nei_con in list_of_neighbors:
                        resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                            int(node.center_y * resize_ratio)),
                                                            (int(node_nei_con.center_x * resize_ratio),
                                                            int(node_nei_con.center_y * resize_ratio)),
                                                            (0, 255, 0), 2)
                """
                for node_nei_con in list_of_neighbors:
                    print(node_nei_con.word)
                    if node_nei_con.id is not node.id:
                        digit_count = 0
                        for char in str(node_nei_con.word):
                            if char.isdigit():
                                digit_count += 1
                        if digit_count > 1:
                            if str(node_nei_con.word).__contains__('$'):
                                # draw specific arrow
                                resize_temp = cv2.arrowedLine(resize_temp, (int(node.center_x * resize_ratio),
                                                                            int(node.center_y * resize_ratio)),
                                                              (int(node_nei_con.center_x * resize_ratio),
                                                               int(node_nei_con.center_y * resize_ratio)),
                                                              (255, 0, 0), 2)
                """
        elif what == "date":
            if node.word.lower() == "date" or str(node.word).lower().__contains__('date'):
                list_of_neighbors = get_list_of_neighbors(words_raw_new, node)
                for node_nei_con in list_of_neighbors:
                    resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                         int(node.center_y * resize_ratio)),
                                           (int(node_nei_con.center_x * resize_ratio),
                                            int(node_nei_con.center_y * resize_ratio)),
                                           (255, 0, 0), 2)

    return resize_temp


def display_all_word_neighors(words_raw_new):
    for node in words_raw_new:
        print("\ncurrent node: {}".format((node.word, node.POS_tag)))
        list_of_neighbors = [node_nei for node_nei in
                             words_raw_new if
                             node_nei.id != node.id and (
                                     node_nei.id == node.top_node_id or
                                     node_nei.id == node.right_node_id or
                                     node_nei.id == node.left_node_id or
                                     node_nei.id == node.bottom_node_id)]
        print("List of neighbors: {}\n".format([(n.word, n.POS_tag) for n in list_of_neighbors]))


def is_total_amount_grand(node_nei_con, node):
    if node_nei_con.id is not node.id:
        digit_count = 0
        for char in str(node_nei_con.word):
            if char.isdigit():
                digit_count += 1
        if digit_count > 1:
            # if str(node_nei_con.word).__contains__('$'):
            return True
    return False


def find_information_rule_based(words_raw_new, image, resize_ratio):
    entries_to_find = ["total", "invoice_no", "date"]
    for entry in entries_to_find:
        image = find_temp(words_raw_new, image, resize_ratio, what=entry)
    if SHOW_IMAGE:
        cv2.imshow("try find entry", image)
    return image


