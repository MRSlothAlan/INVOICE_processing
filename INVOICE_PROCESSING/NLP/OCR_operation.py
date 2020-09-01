from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_hierarchy import InvoiceHierarchy
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine, \
    CopyOfSameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock
import re
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node


def ocr_to_standard_data(info_detailed):
    segment_data = InvoiceHierarchy()
    length = len(info_detailed['level'])
    for index in range(length):
        text = info_detailed['text'][index]
        if text not in ignore_char:
            left, top, width, height, page_num, block_num, par_num, line_num, word_num = \
                get_OCR_data(info_detailed, index)
            if len(text) > 1 and str(text).__contains__(':'):
                # divided the width by the number of characters, define new nodes
                split_list = re.split('(:)', text)
                split_list_modified = [s for s in split_list if s is not '' or ' ']
                width_char_each = int(width / len(text))
                word_left = left
                for index, splited_token in enumerate(split_list_modified):
                    left_added = width_char_each * len(splited_token)
                    token = Node(splited_token, word_left, top, left_added, height, word_num,
                                 sub_word_num=index)
                    word_left += left_added
                    token.word_parse()
                    segment_data.add_new_token(token, page_num, block_num, par_num, line_num)
            else:
                token = Node(text, left, top, width, height, word_num)
                token.word_parse()
                segment_data.add_new_token(token, page_num, block_num, par_num, line_num)
    same_line = SameLine()
    same_block = SameBlock()
    words_raw, same_line, same_block = segment_data.get_lines_blocks(store_line=same_line,
                                                                     store_block=same_block)

    return words_raw, same_line, same_block