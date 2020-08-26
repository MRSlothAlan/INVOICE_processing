"""
Goal:
Use this to parse the invoice and generate data sets

Now only support english invoices, just to simplify things.
if not english: OCR slow
"""
import pytesseract
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import re

import pandas as pd
import string
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_hierarchy import InvoiceHierarchy
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.opencv_image_operations import resize_with_ratio, \
    draw_rectangle_text_with_ratio, pre_process_images_before_scanning
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.information_finder import find_information_rule_based



def main():
    """
    :return:
    """

    # LOAD MODEL
    """
    print("\n===== LOAD MODEL ======\n")
    get_pretrained_model()
    print("COMPLETE\n")
    wv.vocab
    wv.most_similar(positive=["invoice"], topn=5)
    """

    image_files = get_image_files()

    # load csv of currency, save as dictionary
    currency_dict = get_currency_csv()
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    for image_name in tqdm(image_files):
        INVOICE_DATA = list()

        segment_data = InvoiceHierarchy()
        same_line = SameLine()
        same_block = SameBlock()
        print("\n---------------------------------------------\n"
              "processing {}".format(image_name),
              "\n---------------------------------------------\n")
        image_path = str(dataset_dir / image_name)
        image = cv2.imread(image_path, 1)
        resize = resize_with_ratio(image, resize_ratio)
        resize_copy = resize.copy()
        resize_temp = resize.copy()
        resize_function = resize.copy()

        # info = pytesseract.image_to_data(image, lang="chi_tra", output_type='dict')
        # assume all english
        image_pil = pre_process_images_before_scanning(image)
        image = np.array(image_pil)
        image = image[:, :, ::-1].copy()

        info = pytesseract.image_to_data(image, output_type='dict')

        """ index defined: 
         level, page_num, block_num, par_num
         line_num, word_num, left, top, width, height, conf, text
         """
        # assumed that all lists in dict are having uniform length
        length = len(info['level'])
        for index in range(length):
            text = info['text'][index]
            if text not in ignore_char:
                left, top, width, height, page_num, block_num, par_num, line_num, word_num = get_OCR_data(info, index)
                if len(text) > 1 and str(text).__contains__(':'):
                    # divided the width by the number of characters, define new nodes
                    split_list = re.split('(:)', text)
                    split_list_modified = [s for s in split_list if s is not '' or ' ']
                    width_char_each = int(width / len(text))
                    word_left = left
                    for index, splited_token in enumerate(split_list_modified):
                        left_added = width_char_each * len(splited_token)
                        token = Node(splited_token, word_left, top, left_added, height, word_num, sub_word_num=index)
                        word_left += left_added
                        token.word_parse()
                        segment_data.add_new_token(token, page_num, block_num, par_num, line_num)
                else:
                    token = Node(text, left, top, width, height, word_num)
                    token.word_parse()
                    segment_data.add_new_token(token, page_num, block_num, par_num, line_num)
                    """
                Usable features for clustering:
                    coordinates of word
                    block number
                        within the block, may have many different words
                    page num (very rare)

                    line num (just group each line) (EASY, done already)
                        just parse each word, if word near to each other, append it
                        if:
                            too far away horizontally
                """
        words_raw, same_line, same_block = segment_data.get_lines_blocks(store_line=same_line,
                                                                         store_block=same_block)
        # same_line.print()
        # used for node connections
        same_line_copy = same_line
        print("generate graph for individual words")
        same_line_copy.generate_graph()
        resize_temp = same_line_copy.draw_graph(words_raw, resize_temp, resize_ratio)

        TEMP_LINE = same_line.return_start_last_tokens()
        # need to merge some nodes which are closed together
        # word model will be applied here
        height, width, color = image.shape
        words_raw_new = same_line.merge_nearby_token(width)
        # need to plot a graph to check
        if DEBUG_DETAILED:
            for index, line_data in enumerate(words_raw_new):
                resize = draw_rectangle_text_with_ratio(resize,
                                                        line_data.left, line_data.top,
                                                        line_data.width,
                                                        line_data.height,
                                                        ' ',
                                                        resize_ratio)
            cv2.imshow("merged nodes", resize)
            cv2.waitKey(0)
        """
        generate graph for training!! 
        """
        print("generate graph of merged nodes")
        same_line.generate_graph()
        # each node now has the connection data
        # draw the results
        # 20082020: logic error, need to get a list of raw, MERGED tokens
        resize_copy = same_line.draw_graph(words_raw_new, resize_copy, resize_ratio)

        """
        ======  Temporary testing section  ======
        """
        find_information_rule_based(words_raw_new, resize_function, resize_ratio)

        # check list of raw words and connections
        file_name = output_dir / str(image_name[:-4] + ".txt")

        # json_name = output_json_dir / str(image_name[:-4] + ".json")

        # write parsed words to a text file
        """
        same_block.write_to_file(file_name)
        same_line.write_to_file(file_name)
        """
        # generate dataset!
        if GENERATE_DATASET:
            pass

        if DEBUG:
            # cv2.imshow("image", resize)
            cv2.imshow("graph", resize_copy)
            cv2.imshow("original graph", resize_temp)
            cv2.waitKey(0)

        if PARSE:
            # with the node structure, you can tag stuff easily.
            results, currency = same_line.use_parser_re(currency_dict)
            print(results)
            total = 1 + len(results)
            if currency is not None:
                print(currency_dict[currency.upper()])
            else:
                print("currency undefined")


if __name__ == '__main__':
    main()

