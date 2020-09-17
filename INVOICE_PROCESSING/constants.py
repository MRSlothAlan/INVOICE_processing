from pathlib import Path
from os import listdir
from os.path import isfile, join
import csv
import json
import gensim
import gensim.downloader as api
import numpy as np
import re

"""
Characters allowed for data set generation
"""
CHAR_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 ,.-+:/%?$£€#()&@")
CHAR_ALLOWED_STR = "abcdefghijklmnopqrstuvwxyz0123456789 ,.-+:/%?$£€#()&@"

TESSERACT_PATH = str(Path.cwd() / "Tesseract-OCR/tesseract.exe")
dataset_dir = Path.cwd() / "test_images"
output_dir = Path.cwd() / "output"
output_json_dir = Path.cwd() / "output_json"
csv_currency_dir = Path.cwd() / "csv_currency/codes-all.csv"
# currency_dict = dict()

# 24/08/2020 svc classifier
PARSE_ENTRY_NAME = "parse_entry.joblib"
PARSE_ENTRY_CLF_DIR = Path.cwd() / "NLP/word_model/model/{}".format(PARSE_ENTRY_NAME)

resize_ratio = 0.3
ignore_char = [" ", "", None, "\n", "\t", "   ", "  "]

DEBUG = True
SHOW_IMAGE = False
SHOW_SUB_IMAGE = False
DEBUG_DETAILED = False
PARSE = True
TEMP_TEST = False
DL = False
GCN = False
AUTO_ALIGN = True
TIMER = True

LOG_LINE_ITEM = False

HEADER_IS_BLOCK = False

N_GRAM_NO = 4

wv = gensim.models.KeyedVectors(vector_size=2)


def get_pretrained_model():
    wv.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    # wv = api.load('word2vec-google-news-300')


def get_image_files():
    return [f for f in listdir(str(dataset_dir))
               if (isfile(join(str(dataset_dir), f)) and f.split(".")[-1] == 'jpg')]


def get_OCR_data(info, index):
    left = int(info['left'][index])
    top = int(info['top'][index])
    width = int(info['width'][index])
    height = int(info['height'][index])
    page_num = int(info['page_num'][index])
    block_num = int(info['block_num'][index])
    par_num = int(info['par_num'][index])
    line_num = int(info['line_num'][index])
    word_num = int(info['word_num'][index])
    return left, top, width, height, page_num, block_num, par_num, line_num, word_num


def get_currency_csv():
    with open(str(csv_currency_dir), mode='r', encoding='utf-8') as csv_file:
        reader_csv = csv.reader(csv_file)
        header = next(reader_csv)
        # format of each row:
        # ['Entity', 'Currency', 'AlphabeticCode', 'NumericCode', 'MinorUnit', 'WithdrawalDate']
        currency_dict = {row[2]: [row[0], row[1]] for row in reader_csv}
        csv_file.close()
    return currency_dict


# format [[left, top, width, height, 'invoice_number', original_word, entry]]
# format ['HONG KONG', 'Hong Kong Dollar']
def save_as_json(json_path, results, line_items, currency, currency_info, raw_colon_separated_entries):
    dictionary_json = dict()
    for index, result in enumerate(results):
        name = result[4] + "_" + str(index)
        # just create a new dict since it must be a new element
        dictionary_json[name] = dict()
        dictionary_json[name]["left"] = result[0]
        dictionary_json[name]["top"] = result[1]
        dictionary_json[name]["width"] = result[2]
        dictionary_json[name]["height"] = result[3]
        dictionary_json[name]["indication"] = result[5]
        dictionary_json[name]["entry"] = result[6]
    dictionary_json["currency"] = dict()
    """
    Format of proposed line items (each entry)
        [<line items>, <score>] 
    """

    for index, proposed in enumerate(line_items):
        name_table = "table_proposed_" + str(index)
        dictionary_json[name_table] = dict()
        dictionary_json[name_table]["confidence"] = proposed[1]
        dictionary_json[name_table]["raw_content"] = dict()

        for index_line, line in enumerate(proposed[0]):
            # check whether the list is 1D first
            if len(np.array(line).shape) > 1:
                line = list(np.concatenate(line).flat)
            if len(line) > 0:
                dictionary_json[name_table]["raw_content"][index_line] = dict()
                dictionary_json[name_table]["raw_content"][index_line]["starting_left"] = line[0].left
                dictionary_json[name_table]["raw_content"][index_line]["starting_top"] = line[0].top
                dictionary_json[name_table]["raw_content"][index_line]["raw_sentence"] = [n.word for n in line]
    if currency is not None:
        dictionary_json["currency"]["type"] = currency_info[1]
        dictionary_json["currency"]["country"] = currency_info[0]
        dictionary_json["currency"]["abbreviated"] = str(currency)
    else:
        dictionary_json["currency"]["type"] = "undefined"
        dictionary_json["currency"]["country"] = "undefined"
        dictionary_json["currency"]["abbreviated"] = "undefined"

    """
    16092020: also append the raw colon entries
    """
    dictionary_json["raw_colon_entries"] = dict()
    for index_r, entry_row in enumerate(raw_colon_separated_entries):
        if len(entry_row) > 0:
            entry = list()
            value = list()
            for node in entry_row:
                if str(node.word).__contains__(':'):
                    value.append(node)
                else:
                    entry.append(node)
            for index_i, node_e in enumerate(entry):
                dictionary_json["raw_colon_entries"][node_e.word] = re.sub(':', '', str(value[index_i].word))


        # dictionary_json["raw_colon_entries"]

    json_object = json.dumps(dictionary_json, indent=4)
    with open(json_path, "w", encoding='utf-8') as json_f:
        json_f.write(json_object)
    return json_object
