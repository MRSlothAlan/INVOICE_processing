"""
Find specific information from the invoices,
e.g. invoice number, invoice date, total
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import resize_ratio, SHOW_IMAGE
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.graph_process import get_list_of_neighbors
import cv2
# import all the predefined rules
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.rules import *
import enchant
from dateutil.parser import parse


def is_date(string, fuzzy=False):
    try:
        parse(string, fuzzy=fuzzy)
        return True
    except ValueError:
        return False


def find_temp(words_raw_new, resize_temp, resize_ratio=resize_ratio, what="", dict=enchant.Dict()):
    """

    :param words_raw_new:
    :param resize_temp:
    :param resize_ratio:
    :param what:
    :param dict:
    :return:
    """
    # return results, each with [label, node, node_original]
    all_results = list()

    for node in words_raw_new:
        if what == "total":
            if node.word.lower() == "total" or str(node.word).lower().__contains__('total'):
                # looking for neighbors
                list_of_neighbors = get_list_of_neighbors(words_raw_new, node)
                # print([node.word for node in list_of_neighbors])
                for node_nei_con in list_of_neighbors:
                    # if is_total_amount_grand(node_nei_con, node):
                    # draw specific arrow
                    count_digits = 0
                    for bool in [str(n).isdigit() for n in node_nei_con.word]:
                        if bool == True: count_digits += 1
                    if count_digits > 1:
                        if str(node_nei_con.word).__contains__(",") or str(node_nei_con.word).__contains__(".") or\
                                str(node_nei_con.word).__contains__("$") or str(node_nei_con.word).__contains__("HK"):
                            resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                              int(node.center_y * resize_ratio)),
                                                              (int(node_nei_con.center_x * resize_ratio),
                                                              int(node_nei_con.center_y * resize_ratio)),
                                                              (0, 0, 255), 2)
                            all_results.append(["total amount", node_nei_con, node])

        elif what == "invoice_no":
            """
            possible patterns: invoice no, invoice number, No.:
            Weighting: add more weighting to words on right and bottom left
            If the neighbor has recognizable words, reduce the confidence
            """
            string_to_watch = str(node.word).lower().strip()
            if (string_to_watch.__contains__('invoice') and string_to_watch.__contains__("no"))\
                    or (string_to_watch.__contains__("no") and not string_to_watch.__contains__("invoice"))\
                    or (string_to_watch.__contains__("invoice") and string_to_watch.__contains__("num")):
                if 0 < len(str(node.word).lower().strip().split(' ')) < 3:
                    # looking for neighbors
                    list_of_neighbors = get_list_of_neighbors(words_raw_new, node)
                    # print([node.word for node in list_of_neighbors])
                    for node_nei_con in list_of_neighbors:
                        words = node_nei_con.word.split(" ")
                        count_valid_words = 0
                        for word in words:
                            if not str(word).__contains__(":"):
                                try:
                                    if dict.check(word):
                                        count_valid_words += 1
                                except ValueError as e:
                                    pass
                        if count_valid_words <= 1 and any(char.isdigit() for char in node_nei_con.word):

                            resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                                 int(node.center_y * resize_ratio)),
                                                                (int(node_nei_con.center_x * resize_ratio),
                                                                int(node_nei_con.center_y * resize_ratio)),
                                                               (0, 255, 0), 2)
                            all_results.append(["invoice_number", node_nei_con, node])
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
                # print([node.word for node in list_of_neighbors])
                for node_nei_con in list_of_neighbors:
                    if str(node_nei_con.word).strip(' ').__contains__(":") and len(str(node_nei_con.word).strip(' ')) <= 1:
                        # get a further neighbor
                        # print(node_nei_con.right_node_id)
                        additional_nodes = [node for node in words_raw_new if node.id == node_nei_con.right_node_id]
                        # check this neighbor
                        for node_add in additional_nodes:
                            to_parse = str(node_add.word).replace(':', '').strip(' ')
                            if is_date(to_parse):
                                resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                                     int(node.center_y * resize_ratio)),
                                                       (int(node_add.center_x * resize_ratio),
                                                        int(node_add.center_y * resize_ratio)),
                                                       (255, 0, 0), 2)
                                all_results.append(["date", node_add, node])

                    # parse the date
                    to_parse = str(node_nei_con.word).replace(':', '').strip(' ')
                    if is_date(to_parse):
                        # print(node_nei_con.word)
                        resize_temp = cv2.line(resize_temp, (int(node.center_x * resize_ratio),
                                                         int(node.center_y * resize_ratio)),
                                           (int(node_nei_con.center_x * resize_ratio),
                                            int(node_nei_con.center_y * resize_ratio)),
                                           (255, 0, 0), 2)

                        all_results.append(["date", node_nei_con, node])

    return resize_temp, all_results


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


def find_information_rule_based(words_raw_new, image, resize_ratio, dictionary):
    all_results_collect = list()
    entries_to_find = ["total", "invoice_no", "date"]
    for entry in entries_to_find:
        image, all_results = find_temp(words_raw_new, image, resize_ratio, what=entry, dict=dictionary)
        all_results_collect += all_results
    if SHOW_IMAGE:
        cv2.imshow("try find entry", image)
    return image, all_results_collect


