"""
Goal: a structure which stores word and related features such as coordinates
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import CHAR_ALLOWED


class Node:
    def __init__(self, word=str,
                 left=int, top=int, width=int, height=int, word_num=int, sub_word_num=0):
        self.id = int()
        self.word = word
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.word_num = word_num
        # set this, because some OCR word will be splited
        # e.g. node: --> 'node', ':'
        self.sub_word_num = sub_word_num
        self.center_x = 0
        self.center_y = 0
        self.top_node_id = 0
        self.top_node_dis = 0.0
        self.right_node_id = 0
        self.right_node_dis = 0.0
        self.bottom_node_id = 0
        self.bottom_node_dis = 0.0
        self.left_node_id = 0
        self.left_node_dis = 0.0
        # I should have used linked structure before
        self.top_node_ptr = None
        self.left_node_ptr = None
        self.right_node_ptr = None
        self.bottom_node_ptr = None
        # add a label field for the label of the node
        self.label = ""
        # add a POS field for the tag
        self.POS_tag = ""

    def print_info(self):
        print(self.word)

    def word_parse(self):
        # only allow characters within the constant char string
        word_copy = str(self.word)
        self.word = word_copy.strip().translate(
            word_copy.maketrans('/\_<>`‘', '       ', 'p')
        )
        # parse, remove any word not in the set
        word_copy = self.word.lower()
        output = str()
        for index, char in enumerate(word_copy):
            if char in CHAR_ALLOWED:
                output += self.word[index]
        self.word = output


