"""
Store the whole invoice in a hierarchical order
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_line import SameLine
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.store_block import SameBlock

temp_threshold = 300


class InvoiceHierarchy:
    def __init__(self):
        self.segment_data = dict()

    def add_new_token(self, token=Node, page_num=int, block_num=int,
                      par_num=int, line_num=int):
        try:
            self.segment_data[page_num]
        except KeyError as k_err:
            self.segment_data[page_num] = dict()
        try:
            self.segment_data[page_num][block_num]
        except KeyError as k_err:
            self.segment_data[page_num][block_num] = dict()
        try:
            self.segment_data[page_num][block_num][par_num]
        except KeyError as k_err:
            self.segment_data[page_num][block_num][par_num] = dict()
        try:
            self.segment_data[page_num][block_num][par_num][line_num]
        except KeyError as k_err:
            self.segment_data[page_num][block_num][par_num][line_num] = list()
        try:
            self.segment_data[page_num][block_num][par_num][line_num].append(token)
        except Exception as e:
            print(e)
            print("Skipping... ")

        """
                        segment_data = {<page_num> : {<block_num>:{}, <>:{}... }}
                        for each {}: (block)
                            <block_num>: {<par_num> : {}, <par_num> : {}...}
                            for each {}: (par_num)
                                <par_num> : {<line_num> : [], <>:[]}
        """
    def print(self):
        print(self.segment_data)

    def get_lines_blocks(self, store_line=SameLine, store_block=SameBlock):
        """
        Return a list of proposed lines
        :return:
        """
        global_line_count = -1
        node_words_only = list()

        for key_p, page in self.segment_data.items():
            for key_b, block in page.items():
                for key_par, par in block.items():
                    for index_line, line_tuple in enumerate(par.items()):
                        global_line_count += 1
                        # print(global_line_count, index_line, line_tuple)
                        sub_line_index = 0
                        for index_word, word_node in enumerate(line_tuple[1]):
                            # append to block
                            store_block.append_new_block_token(block_num=int(key_b),
                                                               token=word_node)

                            # just append the entire line in here, no need to segment the lines
                            store_line.append_new_line_token(line_number=global_line_count,
                                                             sub_line_number=sub_line_index,
                                                             token=word_node)
                            word_node.line_no = index_line
                            node_words_only.append(word_node)
        return node_words_only, store_line, store_block

