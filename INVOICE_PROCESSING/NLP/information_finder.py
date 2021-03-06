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
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.const_labels import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.depreciated import deprecated

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
        if what == TOTAL_GRAND:
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

        elif what == INVOICE_NUM:
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
        elif what == INVOICE_DATE:
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
                                node_add.label = "date"
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
        elif what == PO_NUM:
            if node.label == PO_NUM:
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
                                                   (0, 100, 100), 2)
                            all_results.append(["po_number", node_nei_con, node])
        elif what == ACCOUNT_NAME:
            # find bank account name
            pass
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
    entries_to_find = [TOTAL_GRAND, INVOICE_NUM, INVOICE_DATE, PO_NUM]
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
    raw_string = ""

    for word in raw_words:
        raw_string += word
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
    """
    21/09/2020: exclude rows if it contains page(?)
    """
    if raw_string.__contains__("page"):
        score /= 2

    if score >= 0.8:
        return True, score
    else:
        return False, score
    # return len([word for word in raw_words if word in POSSIBLE_HEADER_WORDS]) > 0


def parse_lines_get_items(keywords_list, score, index, rect_regions, words_raw, resize_r, is_block, width):
    table_line_items = list()
    if not is_block:
        if LOG_LINE_ITEM:
            # this works like a charm :-)
            print("\nPOSSIBLE HEADER(LINE):{}, SCORE:{}\n".format(keywords_list, score))
    # append the header first
    # since this is a LINE, that means line items are those lines below it.
    temp_ptr = index
    TO_CONTINUE = True

    while temp_ptr < len(rect_regions) and TO_CONTINUE:
        """
        may need to also parse the raw node in order to put them into different columns
        """
        content_temp, label = extract_words(words_raw, rect_regions[temp_ptr], resize_r)

        # line_node_merged = merge_nearby_node_info_process(content_temp, width)
        keywords_list_temp = generate_raw_words(content_temp)
        # check whether any word(s) within the row has the label 'total'
        # perform word matching for now
        # print([node.word for node in line_node_merged])
        for word in keywords_list_temp:
            if  str(word).lower().__contains__("remark") or \
                str(word).lower().__contains__("bank"):
                TO_CONTINUE = False

        table_line_items.append(content_temp)
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
    """
    :param all_lines_in_block_raw: [[node, node, ...], [node, node, ...]]
    :return:
    """
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
        return True
    else:
        return False


def segment_block_into_lines(content):
    """
    segment according to node connections instead of line number suggested by OCR
    :param content:
    :return:
    """
    index = 0
    block_segmented = list()
    block_segmented_linked_list = list()
    block_segmented_top = list()

    all_node = content
    """
    parse according to node parsing, the linked list
    """
    """
    while len(all_node) > 0:
        result = list()
        line = list()
        line.append(all_node[0])
        result.append(all_node[0])

        temp_node = all_node[0].right_node_ptr

        while temp_node is not None:
            line.append(temp_node)
            result.append(temp_node)
            print([node.word for node in result])
            temp_node = temp_node.right_node_ptr
            try:
                temp_node.word
            except Exception as e:
                break
        print("L(PTR): => ", [node.word for node in line])
        block_segmented_linked_list.append(line)

        removed_result = [node for node in all_node if node not in result]
        all_node = removed_result
        # print([node.word for node in all_node])
    """

    """
    parse the node list, according to height, with buffer of 5
    """
    while len(all_node) > 0:
        result = list()
        line = list()
        line.append(all_node[0])
        result.append(all_node[0])
        temp = all_node[0]

        for node in all_node:
            if node.id is not temp.id and \
                abs(node.top - temp.top) <= 5:
                line.append(node)
                result.append(node)
                temp = node
        if LOG_LINE_ITEM:
            print("L(TOP): => ", [node.word for node in line])
        block_segmented_top.append(line)
        removed_result = [node for node in all_node if node not in result]
        all_node = removed_result

    """
    segment based on OCR line number
    while index < len(content) - 1:
        line = list()
        line.append(content[index])
        index += 1
        while content[index].line_no == content[index - 1].line_no and index < len(content):
            line.append(content[index])
            index += 1
            if index >= len(content):
                break
        print("L: => ", [node.word for node in line])
        avg_height = 0
        # further check the line. segment it according to different top coordinate
        for h in [node.height for node in line]:
            avg_height += h
        avg_height /= len(line)
        diff_in_top = abs(line[-1].top - line[0].top)
        if diff_in_top > avg_height:
            # a buffer
            if abs(diff_in_top - avg_height) > 3:
                print(diff_in_top, " > ", avg_height)
                print("can be further splited")

        block_segmented.append(line)
    """
    return block_segmented_top


def predict_header_space(header_node_list):
    """
    input: [[left, [node string]], [left, [node string]], ...]
    return a list of coordinates for segment columns
    :return:
    """

    # preprocess header, remove space nodes
    return [node[0] for index, node in enumerate(header_node_list) if index > 0]


def table_row_partition(row_nodes, coordinate_list):
    """
    given row of nodes and coordinates_lists, partition them into N groups
    :param row_nodes:
    :param coordinate_list:
    :return:
    """
    partition_row = list()
    for i in range(0, len(coordinate_list) + 2):
        partition_row.append([])

    for index, left_cor in enumerate(coordinate_list):
        temp_part = list()
        result = list()
        for node in row_nodes:
            if node.left + node.width <= left_cor:
                temp_part.append(node.word)
                result.append(node)
        node_remove = [node for node in row_nodes if node not in result]
        row_nodes = node_remove
        partition_row[index] = temp_part
    # for remaining nodes, put in the last partition
    partition_row[-1] = [node.word for node in row_nodes]

    # convert the 2D list into a 1D list
    final_row_content = list()
    for list_content in partition_row:
        final_row_content.append(' '.join(list_content))

    return final_row_content


def table_node_segmentation(table_rows, keywords_list_in, width):
    """
    based on extracted lines, classify the header,
    then based on width of nodes between header,
    classify the columns
    :return:
    """
    ALL_PROPOSED_TABLE = list()

    len_max = len(table_rows)
    index = 0

    while index < len_max:
        keyword_list = generate_raw_words(table_rows[index])
        bool, score = is_table_header(keyword_list)

        if bool and len(keyword_list) <= 10:
            # a header detected
            table_list = list()

            def merge_nearby_col(header_nodes, width):
                """
                return a list of the following format:
                    [[left, node_merged], [left, node_merged]]
                :param header_nodes:
                :param width:
                :return:
                """

                i = 0
                merged = list()
                thresh = width / 100
                while i < len(header_nodes):
                    final = [header_nodes[i].left, [header_nodes[i].word]]
                    i += 1
                    try:
                        while header_nodes[i].left - header_nodes[i - 1].left - header_nodes[i - 1].width <= thresh:
                            final[1].append(header_nodes[i].word)
                            i += 1
                            if i > len(header_nodes):
                                break
                    except IndexError as e:
                        pass
                    merged.append(final)
                return merged
            merged = merge_nearby_col(table_rows[index], width)
            coordinate_columns = predict_header_space(merged)

            table_list.append([' '.join(n[1]) for n in merged])
            # starting from next row
            """
            extract proposed line items
            """
            index_temp = index + 1
            END_EXTRACTING = False

            while index_temp < len_max and not END_EXTRACTING:
                if ["remark", "remarks", "bank"] in [node.word.lower() for node in table_rows[index_temp]]:
                    END_EXTRACTING = True
                table_list.append(table_row_partition(table_rows[index_temp], coordinate_columns))
                index_temp += 1

            ALL_PROPOSED_TABLE.append(table_list)
            # convert this part into a csv

            # append entries according to the coordinates extracted
        index += 1


    return ALL_PROPOSED_TABLE


def find_line_item_rule_based_new(words_raw, rect_regions, resize_r, image):
    """
    Goal: get line items based on possible headers
    :param words_raw:
    :param rect_regions:
    :return:
    """
    print("==============================================")
    ALL_SUGGESTED_LINE_ITEM_LIST = list()
    """
    Loop through the rectangles, get contents inside.
    
    Possibilities:
        1. the table is a bunch of lines only
        2. the entire table / region is bounded in a rectangle
        3. the header is a block within a rectangle, the line items are lines
    Possible problems:
        1. the original OCR words do not have good quality, re-parse that region again?
            -- I DON'T THINK I NEED TO RE-PARSE SINCE IT DOESN'T MAKE A HUGE DIFFERENCE ANYWAY --
    
    Method:
        for each rect, parse the content
            if rect is 'line' and is header:
                for lines below it
                    parse lines below til some keywords are matched
            else if rect is 'block' and contains header words:
                if it is a header:
                    parse lines below til some keywords are matched
                else:
                    segment the contents inside into different rows
                    detect each row:
                        if row == header
                            parse lines below til some keywords are matched
    """
    height, width, color = image.shape

    for index_r, rect in enumerate(rect_regions):
        content, label = extract_words(words_raw, rect, resize_r)

        keywords_list = generate_raw_words(content)

        if label == "line":
            bool, score = is_table_header(keywords_list)
            if bool:
                """
                parse the lines below the header
                """
                table_line_items = parse_lines_get_items(keywords_list, score, index_r,
                                                         rect_regions, words_raw, resize_r,
                                                         False, width)
                # table_line_items.insert(0, content)
                """
                Done for line only
                """
                ALL_SUGGESTED_LINE_ITEM_LIST.append([table_node_segmentation(table_line_items, keywords_list, width), score])

        elif label == "block":
            bool, score = is_table_header(keywords_list)
            if bool:

                """
                segment the content into lines
                """
                block_segmented = segment_block_into_lines(content)

                """
                is the block a header or a complete table?
                """
                if is_header_only(block_segmented):
                    """
                    parse the lines below it, calculate a distance metric, classify entries of line items
                    """
                    table_line_items = parse_lines_get_items(keywords_list, score, index_r + 1,
                                                             rect_regions, words_raw, resize_r,
                                                             False, width)

                    for i, line in reversed(list(enumerate(block_segmented))):
                        table_line_items.insert(0, line)
                    ALL_SUGGESTED_LINE_ITEM_LIST.append([table_node_segmentation(table_line_items, None, width), score])

                else:
                    """
                    re-parse the region. The regions are suggested based on the resized image
                    x = int(rect[0] / resize_r)
                    y = int(rect[1] / resize_r)
                    w = int(rect[2] / resize_r)
                    h = int(rect[3] / resize_r)
                    score_header_block = score
                    crop_img = image[y:y + h, x:x + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                    crop_img = cv2.erode(crop_img, kernel, iterations=1)
                    height_c, width_c, color_c = crop_img.shape
                    # necessary?
                    info_detailed = pytesseract.image_to_data(crop_img, output_type='dict')
                    words_raw_temp, same_line, same_block = ocr_to_standard_data(info_detailed)

                    # block_segmented_temp = segment_block_into_lines(words_raw_temp)
                    """
                    """
                    21092020: re-parse does not have good result
                    testing only
                    """
                    ALL_SUGGESTED_LINE_ITEM_LIST.append([table_node_segmentation(block_segmented, None, width), score])
                    """
                    # Function to do insertion sort
                    def insertionSort(arr):
                        # Traverse through 1 to len(arr)
                        for i in range(1, len(arr)):
                            key = arr[i][0].top
                            # Move elements of arr[0..i-1], that are
                            # greater than key, to one position ahead
                            # of their current position
                            j = i - 1
                            while j >= 0 and key < arr[j][0].top:
                                arr[j + 1] = arr[j]
                                j -= 1
                            arr[j + 1][0].top = key
                        return arr

                    sorted_block_seg = insertionSort(block_segmented_temp)
                    final_block = list()
                    """


    return ALL_SUGGESTED_LINE_ITEM_LIST

@deprecated
def find_line_item_rule_based(words_raw_new, words_raw, rect_regions, resize_r, image):
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

        # content, label = extract_words(words_raw_new, rect, resize_r)
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
                    # parse again
                    height, width, color = crop_img.shape
                    # necessary?
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

                    """
                    new approach for getting line items in blocks:
                    get all the nodes within the rectangle
                    link them in rows
                    
                    
                    """

                    all_lines_in_block = get_linked_rows(new_raw) # contains raw nodes
                    all_lines_in_block_raw = get_linked_rows(content) #
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
                    table_line_items = parse_lines_get_items(keywords_list, score, index, rect_regions,
                                                             words_raw_new, resize_r, HEADER_IS_BLOCK, width)
                    ALL_SUGGESTED_LINE_ITEMS.append([table_line_items, score])

            elif constants.HEADER_IS_BLOCK:
                # the previous header is a block type
                # no need to check the header, directly find content now
                if LOG_LINE_ITEM:
                    print("Header is a block, but no info such as 'total'")
                    print("parse the lines below")
                table_line_items = parse_lines_get_items(None, None, index, rect_regions, words_raw_new, resize_r,
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





