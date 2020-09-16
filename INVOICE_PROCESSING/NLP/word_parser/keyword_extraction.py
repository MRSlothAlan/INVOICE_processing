from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from rake_nltk import Rake


def keyword_extraction(word_raw):
    """
    from raw nodes extract keywords and see the result
    :param word_raw:
    :return:
    """
    string_to_test = ""
    for node in word_raw:
        string_to_test += node.word + " "
    r = Rake()

    # Extraction given the text.
    r.extract_keywords_from_text(string_to_test)

    # Extraction given the list of strings where each string is a sentence.
    # r.extract_keywords_from_sentences()

    # To get keyword phrases ranked highest to lowest.
    # r.get_ranked_phrases()

    # To get keyword phrases ranked highest to lowest with scores.
    print(r.get_ranked_phrases_with_scores())