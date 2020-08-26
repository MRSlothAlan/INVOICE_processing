from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import CHAR_ALLOWED, CHAR_ALLOWED_STR
import numpy as np

allowed_char_list = list()


def base_text_features(text, features=['len', 'upper', 'lower', 'alpha', 'digit'], scale=20):
    def count_uppers(text):
        return sum([letter.isupper() for letter in text])

    def count_lowers(text):
        return sum([letter.islower() for letter in text])

    def count_alphas(text):
        return sum([letter.isalpha() for letter in text])

    def count_digits(text):
        return sum([letter.isdigit() for letter in text])

    use_cases = {
        'len': len,
        'upper': count_uppers,
        'lower': count_lowers,
        'alpha': count_alphas,
        'digit': count_digits,
    }
    repr = [use_cases[feature](text) for feature in features]

    if scale is not None:
        for i in range(len(repr)):
            repr[i] = min(repr[i] / scale, 1.0)

    return repr


def feature_from_text(word, values_scales=[100.0, 100000.0, 1000000000.0], scale=20, char_vocab=None):
    """
    Return a feature encoding for the word
    :param word:
    :return:
    """
    # check whether it is a value
    try:
        xtextasval = float(str(word).strip().replace(" ", "").replace("%", "").replace(",", ""))
        xtextisval = 1.0
        assert np.isfinite(xtextasval)
    except Exception as e:
        xtextasval = 0.0
        xtextisval = 0.0
    if xtextisval > 0.0:  # actually a value
        xtextasval = [min(xtextasval / scale, 1.0) for scale in values_scales]
    else:
        xtextasval = [0.0] * len(values_scales)

    allfeats = base_text_features(word, scale=scale)

    if len(word) <= 1:
        text_to_parse = " " + word + " "
    else:
        text_to_parse = word
    begfeats = base_text_features(text_to_parse[0:2], scale=scale)
    endfeats = base_text_features(text_to_parse[-2:0], scale=scale)
    return np.array(allfeats + begfeats + endfeats + xtextasval + [xtextisval])


def generate_encoded_features(word):
    """
    Return a two dimensional encoding
    :param word:
    :return:
    """
    two_dimensional_encoding = list()
    for character in word:
        encoding_for_char = [0] * len(CHAR_ALLOWED)
        encoding_for_char[allowed_char_list.index(character)] = 1
        two_dimensional_encoding.append(encoding_for_char)
    # check if dimension is not okay
    if len(two_dimensional_encoding) < 40:
        dummies_size = 40 - len(two_dimensional_encoding)
        dummy = [0] * len(CHAR_ALLOWED)
        for i in range(dummies_size):
            two_dimensional_encoding.append(dummy)
    return np.array(two_dimensional_encoding)


def generate_dataset(raw_list):
    """
    Generate features that can be used in machine learning models
        each invoice:
            each node:
                position: [top, left, width, height]
                neighbor_ids: [top, left, bottom, right]
                each char:
                    encoded_char [0, 0, ..., 1, 0]
                features[len, upper, lower, alpha, digits,
                        len, upper, lower, alpha, digits, # first two characters
                        len, upper, lower, alpha, digits, # last two characters
                        isvalue, scale_value, linenum # OCR defined]
    :return: dataset, format of EACH entry:
        [<position>, <neighbor_ids>, <encoded_2D_char_each_node>, features]
    """
    dataset = list()
    # convert the allowed character string into a list for encoding purpose
    for character in CHAR_ALLOWED_STR:
        allowed_char_list.append(character)

    for node in raw_list:
        temp = list()
        temp.append(np.array([node.top, node.left, node.width, node.height]))
        temp.append(np.array([node.top_node_id, node.left_node_id,
                     node.bottom_node_id, node.right_node_id]))
        temp.append(generate_encoded_features(node.word.lower()))
        temp.append(feature_from_text(node.word))
        # drop some rows if there are more than 40 rows
        if len(temp[2]) > 40:
            copy = temp[2]
            temp[2] = copy[:40]

        dataset.append(temp)

    return dataset


