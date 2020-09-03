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
import numpy as np
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.graph_process import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.word_parser.parse_words import \
    part_of_speech_label_parsing_rule_based

from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING import constants
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import LOG_LINE_ITEM
"""
Now just for testing, will modify it soon
"""

from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_hierarchy import InvoiceHierarchy
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine, \
    CopyOfSameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock
import re
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.const_labels import *


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
            # if node.word.lower() == "total" or str(node.word).lower().__contains__('total'):
            if node.label == TOTAL_GRAND:
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
            """
            if (string_to_watch.__contains__('invoice') and string_to_watch.__contains__("no"))\
                    or (string_to_watch.__contains__("no") and not string_to_watch.__contains__("invoice"))\
                    or (string_to_watch.__contains__("invoice") and string_to_watch.__contains__("num")):
            """
            if node.label == INVOICE_NUM:
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
            # if node.word.lower() == "date" or str(node.word).lower().__contains__('date'):
            if node.label == INVOICE_DATE:
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


def is_table_header(raw_words):
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
    if score >= 0.8:
        return True, score
    else:
        return False, score
    # return len([word for word in raw_words if word in POSSIBLE_HEADER_WORDS]) > 0


def parse_lines_get_items(content, keywords_list, score, index, rect_regions, words_raw_new, resize_r, is_block, width):
    table_line_items = list()
    if not is_block:
        if LOG_LINE_ITEM:
            # this works like a charm :-)
            print("\nPOSSIBLE HEADER(LINE):{}, SCORE:{}\n".format(keywords_list, score))
            table_line_items.append(content)
    # append the header first
    # since this is a LINE, that means line items are those lines below it.
    temp_ptr = index
    while temp_ptr < len(rect_regions):
        content_temp, label = extract_words(words_raw_new, rect_regions[temp_ptr], resize_r)
        line_node_merged = merge_nearby_node_info_process(content_temp, width)
        keywords_list_temp = generate_raw_words(content_temp)
        # check whether any word(s) within the row has the label 'total'
        # perform word matching for now
        table_line_items.append(line_node_merged)
        # error, sometimes the line is 'empty'
        """
         When shall I end the parsing?
         ..when there are still numbers?
         ** when there are entries without other kinds of tags
             e.g. remarks?
         ** another way: get all the line, excluding those which are extracted as other fields
         *** In the long term, machine learning is a MUST, not only needed
         """
        """
          if len(line_node_merged) > 0:
            try:
                if max(levenshtein_ratio_and_distance("total", w, ratio_calc=True)
                       for w in keywords_list_temp) >= 0.8:
                    continue
            except ValueError as e:
                # you can skip safely
                pass
        """
        temp_ptr += 1
    return table_line_items


def is_header_only(all_lines_in_block_raw):
    if LOG_LINE_ITEM:
        print("CHECK IS HEADER ONLY?")
    """
    characteristics of only header:
        not many digits e.g. no HK$2400
        length is short (not necessary) 
    """
    num_digits = 0
    for item in all_lines_in_block_raw:
        for node in item:
            parsed = [c.isdigit() for c in node.word]
            for bool in parsed:
                if bool:
                    num_digits += 1
    # I do not believe that there are more than 4 digits existed in a header
    if num_digits < 4:
        if LOG_LINE_ITEM:
            print("\nonly a header\n")
        return True
    else:
        return False


def find_line_item_rule_based(words_raw_new, rect_regions, resize_r, image):
    """
    find line items based on rectangular regions and keywords
    :param words_raw_new:
    :param rect_regions:
    :return:
    """
    # print([(node.word, node.left, node.top) for node in words_raw_new])
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    """
    Store all proposed line items, decided the final candidate based on score of header
    Format of each entry:
        [all_lines_in_block_raw, score]
    """
    ALL_SUGGESTED_LINE_ITEMS = list()

    temp_line_items = list()
    temp_block_header_score = 0

    for index, rect in enumerate(rect_regions):
        # [x, y, w, h]
        # extract words from rect region
        # words_raw_new contains words with labels
        content, label = extract_words(words_raw_new, rect, resize_r)
        # if it is a block, scan it one more time. within that region
        # reason: sometimes OCR missed the words, and it is very true
        if label == "block":
            # check this and think whether you need to parse the invoice once again or not
            # this keyword list contains the original contents
            # can be used to find linked rows again!
            keywords_list = generate_raw_words(content)
            bool, score = is_table_header(keywords_list)
            if bool:
                if LOG_LINE_ITEM:
                    print("\nPOSSIBLE HEADER(BLOCK):{}, SCORE:{}\n".format(keywords_list, score))
                # if header detected, scan the entries below and get line items
                # You can parse again if the quality of content is not satisfactory
                # e.g. line item is missing
                # you should parse again then

                # to-do:
                # determine when to parse again (hard, if you do ML then it is not a problem)
                #       temporary plan: parse if score >= 80% (valuable information)
                #                       -> so that it is worth the time to parse it again
                # parse the block into lines and scan each line again
                # problem: invoice parsed and invoice RE-parsed get different words ha ha
                # solution: filling each other, missing words? fill back in!
                if score >= 0.8:
                    score_header_block = score
                    x = int(rect[0] / resize_r)
                    y = int(rect[1] / resize_r)
                    w = int(rect[2] / resize_r)
                    h = int(rect[3] / resize_r)
                    crop_img = image[y:y + h, x:x + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                    crop_img = cv2.erode(crop_img, kernel, iterations=1)
                    if SHOW_SUB_IMAGE:
                        cv2.imshow("sub region original", crop_img)
                        cv2.waitKey(0)

                    resize = cv2.resize(crop_img, (int(w * resize_r), int(h * resize_r)))
                    # before parse again, check the words and see whether the block contains headers
                    # parsing takes time
                    # parse again
                    height, width, color = crop_img.shape
                    info_detailed = pytesseract.image_to_data(crop_img, output_type='dict')
                    words_raw, same_line, same_block = ocr_to_standard_data(info_detailed)
                    same_line.generate_graph()
                    resize = same_line.draw_graph(words_raw, resize, resize_r)

                    # find all the lines by connecting the right neighbors
                    # a much smarter way

                    # Linked list implementation
                    new_raw = same_line.return_raw_node()
                    # sometimes, parsed data is not great, compare to the entries generated previously
                    if LOG_LINE_ITEM:
                        print("Newly parsed words: ", [n.word for n in new_raw])

                    all_lines_in_block = get_linked_rows(new_raw)
                    all_lines_in_block_raw = get_linked_rows(content)
                    """
                    based on all lines, get those which are line items
                    it really depends on OCR, really

                    depends on:
                        header location
                        whether the line is NOT labeled with entries such as invoice_no
                    """
                    if LOG_LINE_ITEM:
                        print("Proposed line items (block) (without machine learning)")
                    for index, line in enumerate(all_lines_in_block):
                        # print([n.word for n in line])
                        # part_of_speech_label_parsing_rule_based(line)
                        line = merge_nearby_node_info_process(line, width)
                        all_lines_in_block[index] = line
                        if LOG_LINE_ITEM:
                            print([n.word for n in line])
                    if LOG_LINE_ITEM:
                        print("\nProposed line items (block) original:")
                        for line_raw in all_lines_in_block_raw:
                            print([n.word for n in line_raw])
                    if LOG_LINE_ITEM:
                        print("\nUPDATE THE ORIGINAL ROWS")
                    for index_r, line_raw in enumerate(all_lines_in_block_raw):
                        # for each line, compare to that in newly parsed content
                        line_raw_list = [n.word for n in line_raw]
                        for line_new in all_lines_in_block:
                            line_new_list = [nn.word for nn in line_new]
                            # main_list = np.setdiff1d([nn.word for nn in line_new], [n.word for n in line_raw])
                            max_score = 0.0
                            read_already = [0] * len(line_new_list)
                            for word_raw in line_raw_list:
                                for index_n, word_new in enumerate(line_new_list):
                                    score = levenshtein_ratio_and_distance(word_raw,
                                                                           word_new,
                                                                           ratio_calc=True)
                                    max_score = max(max_score, score)
                                    if score >= 0.8 and read_already[index_n] != 1:
                                        # marked as read already, won't append this
                                        read_already[index_n] = 1

                            if any(i in line_new_list for i in line_raw_list) or max_score >= 0.8:
                                if LOG_LINE_ITEM:
                                    print("{},{}".format(line_raw_list, line_new_list))
                                # original as the baseline
                                if len(line_raw_list) < len(line_new_list):
                                    # that means there are new info
                                    # append and sort by coordinates
                                    for index_n, word_new in enumerate(line_new_list):
                                        if read_already[index_n] == 0:
                                            # just for checking
                                            line_raw_list.append(word_new)
                                            # really append new node here
                                            all_lines_in_block_raw[index_r].append(line_new[index_n])
                                if LOG_LINE_ITEM:
                                    print("New: {}".format([node.word for node in all_lines_in_block_raw[index_r]]))
                    # after update the block content, make judgement
                    # To-do: sometimes the block is only the header itself, need to parse the lines
                    if is_header_only(all_lines_in_block_raw):
                        constants.HEADER_IS_BLOCK = True

                        temp_line_items += all_lines_in_block_raw
                        temp_block_header_score = score_header_block

                    else:
                        constants.HEADER_IS_BLOCK = False
                        # append it to the proposed line items
                        ALL_SUGGESTED_LINE_ITEMS.append([all_lines_in_block_raw, score_header_block])

                    if SHOW_SUB_IMAGE:
                        cv2.imshow("sub region", resize)
                        cv2.waitKey(0)

        elif label == "line":
            height, width, color = image.shape
            # set a boolean,
            # generate keyword lists
            keywords_list = generate_raw_words(content)
            if not constants.HEADER_IS_BLOCK:
                # check whether line is header
                bool, score = is_table_header(keywords_list)
                if bool:
                    table_line_items = parse_lines_get_items(content, keywords_list, score, index, rect_regions,
                                                             words_raw_new, resize_r, HEADER_IS_BLOCK, width)
                    ALL_SUGGESTED_LINE_ITEMS.append([table_line_items, score])

            elif constants.HEADER_IS_BLOCK:
                # the previous header is a block type
                # no need to check the header, directly find content now
                if LOG_LINE_ITEM:
                    print("Header is a block, but no info such as 'total'")
                    print("parse the lines below")
                table_line_items = parse_lines_get_items(None, None, None, index, rect_regions, words_raw_new, resize_r,
                                                         constants.HEADER_IS_BLOCK, width)
                temp_line_items += table_line_items

                ALL_SUGGESTED_LINE_ITEMS.append([temp_line_items, temp_block_header_score])

                temp_block_header_score = 0
                temp_line_items = list()
                # reset this flag
                constants.HEADER_IS_BLOCK = False

    """
    To-do:
        Change the format to json
        check scoring system
    """
    if LOG_LINE_ITEM:
        for proposed in ALL_SUGGESTED_LINE_ITEMS:
            print("\n===PROPOSED===")
            for entry in proposed[0]:
                print([(node.word, node.label) for node in entry])
            print("SCORE=", proposed[1])

    return ALL_SUGGESTED_LINE_ITEMS





