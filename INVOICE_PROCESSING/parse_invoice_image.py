"""
===============================

THIS SCRIPT IS USED FOR THE API

===============================

Goal:
Use this to parse the invoice and generate data sets

Now only support english invoices, just to simplify things.
if not english: OCR slow
"""
import pytesseract
import cv2
import re
from copy import copy

from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_hierarchy import InvoiceHierarchy
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.opencv_image_operations import resize_with_ratio, \
    draw_rectangle_text_with_ratio, pre_process_images_before_scanning, auto_align_image
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.information_finder import find_information_rule_based
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.feature_extraction.graph_construction import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.ALGO.minimum_spanning_tree import generate_mst_graph


def parse_main(img, image_name):
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

    # load csv of currency, save as dictionary
    currency_dict = get_currency_csv()
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    segment_data = InvoiceHierarchy()
    same_line = SameLine()
    same_block = SameBlock()

    # image = cv2.imread(img, 1)
    image = img
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
    resize_copy = resize.copy()
    resize_temp = resize.copy()
    resize_function = resize.copy()
    resize_mst = resize.copy()
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
    same_line_copy = copy(same_line)
    # same_line_copy = same_line
    print("generate graph for individual words (detailed)")

    glw_detailed = same_line_copy.generate_graph()
    print("Done")
    # get total number of nodes
    total_number_of_nodes = len(words_raw) - 1
    print("generating minimum spanning tree...")
    mst, starting_node_id = generate_mst_graph(glw_detailed, total_number_of_nodes)
    print("Done")
    if SHOW_IMAGE:
        resize_mst = mst.draw_mst_on_graph(words_raw, resize_mst, resize_ratio)
        cv2.imshow("MST", resize_mst)

    resize_temp = same_line_copy.draw_graph(words_raw, resize_temp, resize_ratio)

    # need to merge some nodes which are closed together
    # word model will be applied here
    height, width, color = image.shape
    words_raw_new = same_line.merge_nearby_token(width)
    # for node in words_raw_new:
    #  print(node.word)
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
    """
    generate graph for training!! 
    """
    print("generate graph of merged nodes")
    same_line.generate_graph()
    print("Done")
    # each node now has the connection data
    # draw the results
    # 20082020: logic error, need to get a list of raw, MERGED tokens
    resize_copy = same_line.draw_graph(words_raw_new, resize_copy, resize_ratio)

    """
    ======  Temporary testing section  ======
    """
    import enchant
    d = enchant.Dict("en_US")
    image, all_results = find_information_rule_based(words_raw_new, resize_function, resize_ratio, d)
    # check list of raw words and connections
    file_name = output_dir / str(image_name[:-4] + ".txt")

    json_name = output_json_dir / str(image_name[:-4] + ".json")

    # write parsed words to a text file
    """
    same_block.write_to_file(file_name)
    same_line.write_to_file(file_name)
    """
    # cluster regions

    if DEBUG:
        if SHOW_IMAGE:
            # cv2.imshow("image", resize)
            cv2.imshow("graph", resize_copy)
            cv2.imshow("original graph", resize_temp)
            cv2.waitKey(0)

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
    json_object = save_as_json(json_name, results, currency, currency_dict[currency.upper()])
    return json_object






