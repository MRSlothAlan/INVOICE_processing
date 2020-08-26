"""
parse_words.py
parse nodes in a list of n-grams. return a list of split entries
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.basic_operations.generate_n_gram import *
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
from nltk import pos_tag
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
import nltk
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.word_parser.RE.rules_regex import *


def parse_word_nodes_n_gram_cumulative(node_list, N=4):
    """
    parse nodes
    :param node_list: a list of nodes
    :return:
    """
    # generate N grams, N value: from 1 to N
    n = 1
    n_grams_list = list()
    while n <= N:
        n_grams_list.append(generate_n_gram(node_list, N=n))
        n += 1

    # work to do: data set generation for the nodes
    if DEBUG_DETAILED:
        # parse using clf
        from joblib import load
        clf = load(PARSE_ENTRY_CLF_DIR)
        print("=====node list: {}=====".format([node.word for node in node_list]))
        for Ngram in n_grams_list:
            for item in Ngram:
                string = str()
                for node in item:
                    string += node.word + " "
                string_to_parse = string.strip()

                # print(string_to_parse)

                """
                TO-DO: TRY USING DIFFERENT MODELS TO PARSE THE STRING RETRIVED 
                """
                # 1: TRY WORD2VEC
                """
                one_hot = [0.0] * len(CHAR_ALLOWED_STR)
                for char in string.lower():
                    if char in CHAR_ALLOWED:
                        one_hot[CHAR_ALLOWED_STR.index(char)] += 1.0

                y_pred = clf.decision_function([one_hot])
                print(y_pred)
                """
                # print([node.word for node in item])
    return n_grams_list


def parse_word_nodes_n_gram_fixed(node_list, N=4):
    return generate_n_gram(node_list, N=N)


def part_of_speech_label_parsing_rule_based(node_list):
    """
    Plan:
        parse words, set POS tags in each node
        Using RULE-BASED APPROACH (see somewhere else for content-based approach)
            invoice number:
                expected two nouns, two words not similar, contains 'invoice' 'no' 'number', or etc, or
                expected one word, contains 'No.'

        => every time detected labels, set in Node class, the 'label' field
    :return:
    """
    try:
        # case 1: possible labels with one words only
        n_gram_one = parse_word_nodes_n_gram_fixed(node_list, 1)
        raw_token = [n[0].word for n in n_gram_one]
        raw_token_tag = pos_tag(raw_token)
        # list [(word, tag), (), ...]
        # set the POS tags in the Node class, will be used in next stage
        for index, tag in enumerate(raw_token_tag):
            n_gram_one[index][0].POS_tag = raw_token_tag[index][1]
            # also set the part of speech tags to the nodes
            node_list[index].POS_tag = raw_token_tag[index][1]
            # print("=== SET TAG ===")
            # print(n_gram_one[index][0].word, n_gram_one[index][0].POS_tag)
        """
        # grammar = r"""
        #   GROUP_N: {<.*>*}
        #    }<[\.VI].*>+{       # chink any verbs, prepositions or periods
        #    <.*>}{<DT>          # separate on determiners
        #  INV_L: {<.*><:>}
        # Notes: Parser is powerful, but need to understand grammar rules first
        # cp = nltk.RegexpParser(grammar)
        # tree = cp.parse(raw_token_tag)
        regex_parse_unigram(n_gram_one)
        # case 2: possible labels with two words
        n_gram_two = parse_word_nodes_n_gram_fixed(node_list, 2)
        # format: [[node, node], [node, node], ...]
        regex_parse_bigram(n_gram_two)
        n_gram_three = parse_word_nodes_n_gram_fixed(node_list, 3)
        regex_parse_trigram(n_gram_three)
        return True

    except LookupError as e:
        print(e)
        print("Download all necessary packages")
        nltk.download('averaged_perceptron_tagger')
        nltk.download('tagsets')
        # run again
        return False





