"""
Implement parsing functions in this file
"""
import re
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from collections import Counter
import operator


def label_words(store_temp_results, string, raw_word_list, label):
    store_temp_results[string][0] = raw_word_list[0].left
    store_temp_results[string][1] = raw_word_list[0].top
    store_temp_results[string][2] = raw_word_list[-1].left + raw_word_list[-1].width - raw_word_list[0].left
    store_temp_results[string][3] = raw_word_list[-1].height
    store_temp_results[string][4] = "invoice_number"


def parse_using_re(raw_word_list, currency_dict):
    """
    Regular expression parsing, try to identify:
    invoice number, invoice date, currency, amount, supplier
    :param raw_word_list: a list of Node type data
    :return: a list, with content: (rect coordinates, label)
    """
    store_temp_results = dict()
    currency_store = dict()
    string = ' '.join([str(elem.word).lower() for elem in raw_word_list])

    if len(raw_word_list) <= 5:
        # check whether it is invoice number
        if re.match(r'invoice number|invoice no', string):
            # store the coordinates of the phrase
            # print("Match, {}".format(string))
            if string not in store_temp_results:
                store_temp_results[string] = [None] * 5
                # print(raw_word_list[-1].word)
                # print(raw_word_list[0].word)
                # list format: [left, top, width, height, label]
                label_words(store_temp_results, string, raw_word_list, label="invoice_number")

    # matching currency?
    string_split_set = set(list(string.split()))
    intersection_currency = list(string_split_set.intersection(list(str(key).lower() for key in currency_dict.keys())))
    # one exceptional case: hk$?
    for element in string_split_set:
        for special_hk_name in ["hk$", "hkd"]:
            if special_hk_name in element:
                intersection_currency.append("hkd")

    results = list()
    for key in store_temp_results:
        result_temp = store_temp_results[key]
        result_temp.append(str(key))
        results.append(result_temp)

    return results, intersection_currency


def decided_currency(currency_list):
    currency_list_count = Counter(currency_list)
    return max(currency_list_count.items(), key=operator.itemgetter(1))[0]



