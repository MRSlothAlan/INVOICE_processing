"""
a dict: {<block number> : [<raw words>]}
"""

from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.parser import parse_using_re


class SameBlock:
    def __init__(self):
        self.store_block = dict()

    def append_new_block_token(self, block_num, token=Node):
        try:
            self.store_block[block_num]
        except KeyError as e:
            self.store_block[block_num] = list()

        self.store_block[block_num].append(token)

    def print(self):
        return self.store_block

    def write_to_file(self, dir):
        with open(dir, "w+", encoding='utf-8') as f:
            for key in self.store_block:
                string = str()
                string = ' '.join([str(elem.word) for elem in self.store_block[key]])
                string = string.strip()
                string += "\n"
                f.write(string)
        f.close()

    def use_parser_re(self):
        results = list()
        for key in self.store_block:
            results += parse_using_re(self.store_block[key])
        return results
