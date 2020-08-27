"""
One of the rule-based operations, mainly uses RE and string matching
"""
"""
known words (uni-gram):
    No --> invoice no
    Total --> total amount
    Vendor --> supplier
    Date --> invoice date

known bigram: (check with POS tags)
    invoice no --> invoice no
    invoice number --> invoice no
    invoice date --> date
    Amount Total --> total
    Total HKD --> total
    ayable to --> supplier (payable to someone)
    
known trigram:
    total amount hkd --> total
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.const_labels import *


def regex_parse_unigram(unigram_node_list):
    """
    set label to the node
    :param unigram_node_list:
    :return:
    """
    for node in unigram_node_list:
        raw_word = str(node[0].word).strip().lower()
        length = len(raw_word)
        if raw_word.__contains__("no") and length <= 3:
            if str(node[0].POS_tag).__contains__("N"):
                node[0].label = INVOICE_NUM
        elif raw_word.__contains__("total") and length <= 6:
            node[0].label = TOTAL_GRAND
        elif raw_word.__contains__("date") and length <= 4:
            node[0].label = INVOICE_DATE


def regex_parse_bigram(bigram_node_list):
    """
    a temporary method to parse bi-gram
    make use of POS and word matching
    :return:
    """
    for node in bigram_node_list:
        # print(node[0].POS_tag, node[0].word, " | ", node[1].POS_tag, node[1].word)
        first_n_pos = str(node[0].POS_tag)
        second_n_pos = str(node[1].POS_tag)
        first_n_word = str(node[0].word).strip().lower()
        second_n_word = str(node[1].word).strip().lower()

        if first_n_pos.__contains__("N"):
            if second_n_pos.__contains__("DT") or second_n_pos.__contains__("N"):
                if first_n_word.__contains__("vend") and second_n_word.__contains__("no"):
                    node[0].label = VENDOR
                    node[1].label = VENDOR
                # print("Bigram: {} and {}".format(first_n_word, second_n_word))
                if first_n_word.__contains__("invoice") and second_n_word.__contains__("n"):
                    node[0].label = INVOICE_NUM
                    node[1].label = INVOICE_NUM
                # part of speech are both noun and contains total?
                if first_n_word.__contains__("total") and second_n_word.__contains__("hkd"):
                    node[0].label = TOTAL_GRAND
                    node[1].label = TOTAL_GRAND

        # sometimes, total can be used as adjective. E.g. total amount
        # JJ: adjective or numeral, ordinal
        if first_n_pos.__contains__("JJ"):
            if second_n_pos.__contains__("N"):
                if first_n_word.__contains__("total") or first_n_word.__contains__("otal"):
                    if second_n_word.__contains__("amount"):
                        node[0].label = TOTAL_GRAND
                        node[1].label = TOTAL_GRAND


def regex_parse_trigram(trigram_node_list):
    for node in trigram_node_list:
        first_n_pos = str(node[0].POS_tag)
        second_n_pos = str(node[1].POS_tag)
        third_n_pos = str(node[2].POS_tag)

        first_n_word = str(node[0].word).strip().lower()
        second_n_word = str(node[1].word).strip().lower()
        third_n_word = str(node[2].word).strip().lower()

        if first_n_pos.__contains__("JJ"):
            if second_n_pos.__contains__("N"):
                if third_n_pos.__contains__("N"):
                    if first_n_word.__contains__("total"):
                        if second_n_word.__contains__("amount"):
                            if third_n_word.__contains__("due"):
                                node[0].label = TOTAL_GRAND
                                node[1].label = TOTAL_GRAND
                                node[2].label = TOTAL_GRAND
