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
    cv2.imshow("auto-align", image)
    cv2.waitKey(0)
