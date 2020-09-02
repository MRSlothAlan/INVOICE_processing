"""
Find specific information from the invoices,
e.g. invoice number, invoice date, total
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.graph_process import get_list_of_neighbors
import cv2
# import all the predefined rules
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.rules import *
import enchant
from dateutil.parser import parse
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.rules import *
import pytesseract
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.OCR_operation import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.basic_operations.basic_string_operation import \
    levenshtein_ratio_and_distance
"""
Now just for testing, will modify it soon
"""

from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_hierarchy import InvoiceHierarchy
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine, \
    CopyOfSameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock
import re


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


def extract_words(words_raw_new, rect_bounding_box, resize_r=resize_ratio):
    """
    :param words_raw_new:
    :param rect_bounding_box:
    :return:
    """
    x = rect_bounding_box[0]
    y = rect_bounding_box[1]
    w = rect_bounding_box[2]
    h = rect_bounding_box[3]
    content = list()
    min_t = 0
    max_t = 0
    word_h = words_raw_new[0].height
    counter = 0

    for index, raw_node in enumerate(words_raw_new):
        left = int(raw_node.left * resize_r)
        top = int(raw_node.top * resize_r)
        height = int(raw_node.height * resize_r)
        if left >= x - 5 and \
           top >= y - 5:
            if top + height <= (y + h):
                counter += 1
                # print("top: ", top)
                if min_t == 0:
                    min_t = top
                else:
                    min_t = min(min_t, top)
                if max_t == 0:
                    max_t = top
                else:
                    max_t = max(max_t, top)
                word_h += height
                content.append(raw_node)
    try:
        word_h /= counter
    except ZeroDivisionError as e:
        pass
    # indicate whether the layout is a table or just a line item
    # print("\n", [node.word for node in content])
    # print("average height: ", word_h)
    # print("max - min: ", abs(max_t - min_t))
    # determine whether it is a block, and whether it is a block which is big enough
    if abs(max_t - min_t) > 10:
        label = "block"
    else:
        label = "line"
    return content, label


def generate_raw_words(content):
    raw_word_list = list()
    for node in content:
        words = [str(w).lower() for w in str(node.word).split(" ")]
        raw_word_list += words
    final_raw_word_list = [w for w in raw_word_list if w is not '']
    return final_raw_word_list


def is_table_or_table_header(raw_words):
    """
    A temporary hard-code solution for english
    In the long term, a machine learning model should be made
    Method:
        blur all line items
        bounding box indicate table content

        training parameters:
            1. image cropped
            2. size
            3. position of box

    :param raw_words:
    :return:
    """
    score = 0.0
    count = 0
    word_list = list(POSSIBLE_HEADER_WORDS.keys())
    for word in raw_words:
        # need to apply Levenshtein Distance in order to extract words needed
        # also apply the weighting defined by me.
        max_score = max(levenshtein_ratio_and_distance(word.lower(), k, True)
                        for k in word_list)
        try:
            max_score *= float(POSSIBLE_HEADER_WORDS[word])
        except KeyError:
            max_score *= 0.95

        if max_score > 0.7:
            count += 1
            score += max_score
    try:
        score /= count
    except ZeroDivisionError as e:
        score = 0
    # print("Line: {}, Score: {}, No. of words: {}".format(raw_words, score, count))
    # 02092020: only rows with score > 80% or above are extracted
    if score > 0.8:
        return True, score
    else:
        return False, score
    # return len([word for word in raw_words if word in POSSIBLE_HEADER_WORDS]) > 0


def find_line_item_rule_based(words_raw_new, rect_regions, resize_r, image):
    """
    find line items based on rectangular regions and keywords
    :param words_raw_new:
    :param rect_regions:
    :return:
    """
    # print([(node.word, node.left, node.top) for node in words_raw_new])
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    for rect in rect_regions:
        # [x, y, w, h]
        # extract words from rect region
        content, label = extract_words(words_raw_new, rect, resize_r)
        # if it is a block, scan it one more time. within that region
        # reason: sometimes OCR missed the words
        if label == "block":

            x = int(rect[0] / resize_r)
            y = int(rect[1] / resize_r)
            w = int(rect[2] / resize_r)
            h = int(rect[3] / resize_r)
            crop_img = image[y:y+h, x:x+w]

            resize = cv2.resize(crop_img, (int(w * resize_r), int(h * resize_r)))
            # parse again
            info_detailed = pytesseract.image_to_data(crop_img, output_type='dict')
            words_raw, same_line, same_block = ocr_to_standard_data(info_detailed)
            same_line.generate_graph()
            resize = same_line.draw_graph(words_raw, resize, resize_r)

            # find all the lines by connecting the right neighbors
            # it is a pity that I haven't thought of this before
            # may include this in the same line class
            # a much smarter way

            if SHOW_SUB_IMAGE:
                cv2.imshow("sub region", resize)
                cv2.waitKey(0)
            # check the top row. if it is header, extract the line items
            # that's it for now
        elif label == "line":
            # generate keyword lists
            keywords_list = generate_raw_words(content)
            bool, score = is_table_or_table_header(keywords_list)
            if bool:
                # this works like a charm :-)
                print("POSSIBLE HEADER:{}, SCORE:{}".format(keywords_list, score))
                # scan the rows below it

        # info = pytesseract.image_to_data(image, output_type='dict')
        # check whether the region has line items




