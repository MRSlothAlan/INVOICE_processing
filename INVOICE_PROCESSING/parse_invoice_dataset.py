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
from copy import copy

import pandas as pd
import string
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_hierarchy import InvoiceHierarchy
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine, CopyOfSameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.opencv_image_operations import resize_with_ratio, \
    draw_rectangle_text_with_ratio, pre_process_images_before_scanning, auto_align_image
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.information_finder import find_information_rule_based, \
    find_line_item_rule_based
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.Neural_network.feature_extraction.graph_construction import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.ALGO.minimum_spanning_tree import GraphLineWeights, generate_mst_graph
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.ALGO.region_proposal import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.OCR_operation import *


def parse_main():
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

        print("\n---------------------------------------------\n"
              "processing {}".format(image_name),
              "\n---------------------------------------------\n")
        image_path = str(dataset_dir / image_name)
        image = cv2.imread(image_path, 1)

        # info = pytesseract.image_to_data(image, lang="chi_tra", output_type='dict')
        # assume all english
        image_pil = pre_process_images_before_scanning(image)
        image = np.array(image_pil)
        # to OpenCV format
        image = image[:, :, ::-1].copy()
        # align image
        print("auto align image...")
        image = auto_align_image(img=image)
        print("done")

        resize = resize_with_ratio(image, resize_ratio)
        resize_region = resize.copy()
        resize_copy = resize.copy()
        resize_temp = resize.copy()
        resize_function = resize.copy()
        resize_mst = resize.copy()

        print("propose regions")
        # try to pre-process image, generate regions
        """
        format of entry:
        [x, y, w, h]
        """
        rect_regions = region_proposal(image)

        if SHOW_IMAGE:
            for rect in rect_regions:
                cv2.rectangle(resize_region, (rect[0], rect[1]),
                              (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 3)
            cv2.imshow("regions", resize_region)
        print("finish")

        info = pytesseract.image_to_data(image, output_type='dict')
        image_copy = image.copy()

        """ index defined: 
         level, page_num, block_num, par_num
         line_num, word_num, left, top, width, height, conf, text
         """
        words_raw, same_line, same_block = ocr_to_standard_data(info)

        # same_line.print()
        # used for node connections
        same_line_copy = copy(same_line)
        # same_line_copy = same_line
        print("generate graph for individual words (detailed)")
        glw_detailed = same_line_copy.generate_graph()
        print("Done")
        # get total number of nodes
        total_number_of_nodes = len(words_raw) - 1
        """
        print("generating minimum spanning tree...")
        mst, starting_node_id = generate_mst_graph(glw_detailed, total_number_of_nodes)
        print("Done")

        if SHOW_IMAGE:
            resize_mst = mst.draw_mst_on_graph(words_raw, resize_mst, resize_ratio)
            cv2.imshow("MST", resize_mst)
        """
        resize_temp = same_line_copy.draw_graph(words_raw, resize_temp, resize_ratio)
        # need to merge some nodes which are closed together
        # word model will be applied here
        height, width, color = image.shape
        words_raw_new = same_line.merge_nearby_token(width)
        # for node in words_raw_new:
        # print(node.word)
        # need to plot a graph to check
        if DEBUG_DETAILED:
            if SHOW_IMAGE:
                for index, line_data in enumerate(words_raw_new):
                    resize = draw_rectangle_text_with_ratio(resize,
                                                            line_data.left, line_data.top,
                                                            line_data.width,
                                                            line_data.height,
                                                            ' ',
                                                            resize_ratio)
                cv2.imshow("merged nodes", resize)
                cv2.waitKey(0)
        print("generate graph of merged nodes")
        same_line.generate_graph()
        print("Done")
        # each node now has the connection data
        # draw the results
        # 20082020: logic error, need to get a list of raw, MERGED tokens
        resize_copy = same_line.draw_graph(words_raw_new, resize_copy, resize_ratio)
        import enchant
        d = enchant.Dict("en_US")
        # check specific entries only
        image, all_results = find_information_rule_based(words_raw_new, resize_function, resize_ratio, d)
        find_line_item_rule_based(words_raw_new, rect_regions, resize_ratio, image_copy)

        json_name = output_json_dir / str(image_name[:-4] + ".json")

        # write parsed words to a text file
        """
        same_block.write_to_file(file_name)
        same_line.write_to_file(file_name)
        """

        # generate data set!
        # may not needed
        if DL:
            adjacency_matrix_basic(words_raw_new)

        if DEBUG:
            if SHOW_IMAGE or SHOW_SUB_IMAGE:
                # cv2.imshow("image", resize)
                cv2.imshow("graph", resize_copy)
                cv2.imshow("original graph", resize_temp)
                cv2.waitKey(0)

        if PARSE:
            # with the node structure, you can tag stuff easily.
            results, currency = same_line_copy.use_parser_re(currency_dict)
            if currency is not None:
                print(currency_dict[currency.upper()])
            else:
                print("currency undefined")
            # return a json file of data
            results = list()
            for tagged_items in all_results:
                label = tagged_items[0]
                node_entry = tagged_items[1]
                node_origin = tagged_items[2]
                # format [[left, top, width, height, 'invoice_number', original_word, entry]]
                # format ['HONG KONG', 'Hong Kong Dollar']
                # def save_as_json(json_path, results, currency, currency_info):
                results.append([node_entry.left,
                                node_entry.top,
                                node_entry.width,
                                node_entry.height,
                                label,
                                node_origin.word,
                                node_entry.word])
                # print(label, node_entry.word, node_origin.word)
            """
            Append line items here
            """
            try:
                save_as_json(json_name, results, currency, currency_dict[currency.upper()])
            except AttributeError as e:
                save_as_json(json_name, results, currency=None, currency_info=None)


if __name__ == '__main__':
    parse_main()

