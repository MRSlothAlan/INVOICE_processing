"""
generate_n_gram.py
Goal: given a list of nodes, generate N-grams with different values of N
      then, put to a word model, determine the label

E.g. "invoice.no:", "invoice number:" belogs to "invoice_number" label
    then, you can split the entry
"""


def generate_n_gram(word_list, N=4):
    """
    Generate N-gram based on word-list
    :param word_list:
    :param N: generate N-gram, up to N
    :return:
    """
    ngrams = zip(*[word_list[i:] for i in range(N)])
    final_pairs = list()
    for item in ngrams:
        final_pairs.append(list(tuple(item)))
    # return [" ".join(ngram) for ngram in ngrams]
    return final_pairs

