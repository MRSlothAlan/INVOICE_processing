"""
Store entries on the same line
a dictionary: {<line_number> : {<sub_line>:[], <sub_line>:[], ...} }
"""
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.word_node import Node
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.parser import parse_using_re, decided_currency
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
import cv2
import math
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.NLP.word_parser.parse_words import *
import tqdm
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.ALGO.minimum_spanning_tree import GraphLineWeights


class SameLine:
    def __init__(self):
        self.storage = dict()

    def append_new_line_token(self, line_number, sub_line_number, token=Node):
        try:
            self.storage[line_number]
        except KeyError as e:
            self.storage[line_number] = dict()
        try:
            self.storage[line_number][sub_line_number]
        except KeyError as e:
            self.storage[line_number][sub_line_number] = list()
        self.storage[line_number][sub_line_number].append(token)

    def print(self):
        for key in self.storage:
            print(key)
            for key_line in self.storage[key]:
                print(key_line)
                print([node.word for node in self.storage[key][key_line]])

    def return_raw_node(self):
        raw_list = list()
        for key in self.storage:
            for key_line in self.storage[key]:
                raw_list += [node for node in self.storage[key][key_line]]
        return raw_list

    def draw_the_arrows(self, words_raw, node_id, node, image, resize_ratio, color, thickness=2):
        other_node = set(x for x in words_raw if x.id == node_id).pop()
        image = cv2.line(image, (int(node.center_x * resize_ratio),
                                        int(node.center_y * resize_ratio)),
                                (int(other_node.center_x * resize_ratio),
                                 int(other_node.center_y * resize_ratio)),
                                color, thickness)
        return image

    def draw_graph(self, words_raw, image, resize_ratio):
        """
        only draws top for now
        :return:
        """
        for key in self.storage:
            for key_line in self.storage[key]:
                for node in self.storage[key][key_line]:
                    try:
                        self.draw_the_arrows(words_raw, node.top_node_id, node, image, resize_ratio, (255, 0, 0))
                        self.draw_the_arrows(words_raw, node.bottom_node_id, node, image, resize_ratio, (0, 255, 0), thickness=4)
                        self.draw_the_arrows(words_raw, node.left_node_id, node, image, resize_ratio, (0, 0, 255))
                        self.draw_the_arrows(words_raw, node.right_node_id, node, image, resize_ratio, (100, 100, 100), thickness=4)
                    except Exception as e:
                        print(e)
        return image

    def return_start_last_tokens(self):
        processed_line = list()
        for key in self.storage:
            # print(key)
            for key_line in self.storage[key]:
                # print([node.word for node in self.storage[key][key_line]])
                # print(self.storage[key][key_line][0].word, self.storage[key][key_line][-1].word)
                processed_line.append([self.storage[key][key_line][0],
                                       self.storage[key][key_line][-1]])
        return processed_line

    def merge_token_version_two(self, width):
        """
        In this version, goal:
            Effectively separate colon and data
            e.g. raw line: "invoice number: 123        date:15092020     payment due date:15092020"

            expected segmented entry:
            "invoice number", ": 123", "date", ":15092020", "payment due date", ":15092020"

            AFTER segmentation, assign labels based on RE / POS

        Please keep the time complexity within O(N)

        Problem:
            segmented words may not be a phrase with single meaning
            for that, a word model is needed
        Algo:
            for a given line

        :param width:
        :return:
        """
        return NotImplementedError()



    def merge_nearby_token(self, width):
        threshold_merge = int(width * (1 / 7))
        words_raw_new = list()

        for key in self.storage:
            for key_line in self.storage[key]:
                # idea: if the left neighbor is within a distance of 200, merge them
                merged_nodes_list = list()
                current_index = 0
                max_length = len(self.storage[key][key_line])
                while current_index < max_length:
                    # check the left node only
                    # this looks clumsy but it works okay?
                    temp_merge = list()
                    temp_merge.append(self.storage[key][key_line][current_index])

                    while current_index + 1 < max_length and \
                        abs(self.storage[key][key_line][current_index].left - \
                            self.storage[key][key_line][current_index + 1].left) <= threshold_merge :
                        temp_merge.append(self.storage[key][key_line][current_index + 1])
                        current_index += 1

                    # remove unexpected space
                    temp_merge_remove_space = list()
                    for node in temp_merge:
                        if str(node.word).lower().strip() is not '':
                            temp_merge_remove_space.append(node)
                    # HOWEVER, IF WORD CONTAINS A : AT THE END OR THERE IS A :, DON'T MERGE THEM PLEASE
                    # consider n-gram, then parse each of them, if they match certain patterns, it works
                    list_of_final_temp = list()
                    """
                    N-GRAM operations
                    20082020: to-do
                        N-gram generation
                        
                        FAILED implementation:
                            data set generation
                            word model generation (look at invoice net for hints)
                        
                        PROPOSED implementation:
                            get bi-gram
                            parse word, get     
                    """
                    # list_of_final_temp = parse_word_nodes_n_gram_cumulative(temp_merge_remove_space, N_GRAM_NO)

                    def merge_nodes_to_one(node_list):
                        """
                        Given a list of node, merge it to one node
                        :param node_list:
                        :return:
                        """
                        # no need to set new id, it is handled in the next part of the workflow
                        new_left = Node(node_list[0]).left.__int__(node_list[0].left)
                        new_top = min([node.top for node in node_list])
                        new_word = ""
                        new_width = 0
                        new_height = 0
                        new_label = str()
                        # word_num: number of characters
                        if len(node_list) > 0:
                            new_label = node_list[0].label
                        for index, node in enumerate(node_list):
                            new_word += str(node.word) + " "
                            # width is a bit tricky
                            if index == 0:
                                new_width += Node(node).width.__int__(node.width)
                            else:
                                # because you need to consider the gap!
                                # sometimes there is no left (the margin)
                                # please, don't mess up init with int
                                # cannot add width like this,need to minus left
                                new_width += Node(node).width.__int__(node.width) + \
                                             Node(node).left.__int__(node.left) - \
                                             Node(node_list[index - 1]).left.__int__(node_list[index - 1].left) - \
                                             Node(node_list[index - 1]).width.__int__(node_list[index - 1].width)

                            # determine height is a bit tricky.
                            # top point is new_top
                            # I just check max top for now
                            new_height = max(Node(node).height.__int__(node.height), new_height)

                        new_word_num = len(new_word)
                        new_node_final = Node(new_word, new_left, new_top, new_width, new_height, new_word_num)
                        # print(new_node_final.word)
                        # set the label for the word
                        new_node_final.label = new_label
                        return new_node_final

                        # for node in node_list:
                    # avoid empty node
                    if len(temp_merge_remove_space) > 0:
                        """
                        current state:
                            get node (separated by space)
                        expected outcome:
                            further split the nodes into several more nodes based on
                                part of speech
                                RE parsing
                        """
                        # N-gram-operations
                        success = part_of_speech_label_parsing_rule_based(temp_merge_remove_space)
                        if not success:
                            # parse one more time
                            part_of_speech_label_parsing_rule_based(temp_merge_remove_space)
                        # after labelling, merge the nodes with same label

                        def partition_nodes(temp):
                            final_parsed = list()
                            final_TEMP = list()
                            # temp format: [<node>, <>, <>, ...]
                            # final format: [[<node>, <>], [<>], [], ...]
                            index = 1
                            total_length = len(temp)
                            prev_node = temp[0]

                            temp_partition = list()
                            temp_partition.append(prev_node)

                            TEMP = list()
                            TEMP.append(prev_node.word)

                            while index < total_length:

                                current_node = temp[index]
                                while current_node.label == prev_node.label and \
                                        index < total_length:
                                    temp_partition.append(current_node)
                                    TEMP.append(current_node.word)
                                    prev_node = current_node
                                    index += 1
                                    if index < total_length:
                                        current_node = temp[index]
                                # print([node.word for node in temp_partition])
                                # append the data first, then perform checking
                                # print("append: ", TEMP)
                                final_parsed.append(temp_partition)
                                final_TEMP.append(TEMP)
                                # print("current node: {}, prev node: {}".format(current_node.word, prev_node.word))

                                temp_partition = list()
                                TEMP = list()

                                if current_node.label != prev_node.label and index + 1 < total_length:
                                    temp_partition.append(current_node)
                                    TEMP.append(current_node.word)
                                    prev_node = current_node
                                    index += 1
                                    current_node = temp[index]

                                # handle one base case
                                if index + 1 == total_length and current_node.label != prev_node.label:
                                    final_parsed.append([current_node])
                                    final_TEMP.append([current_node.word])
                                    index += 1

                            """
                            16092020: Final results process?
                            Sometimes, the node is like this:
                            ['..', 'HWB', 'Number', ':...']
                            Add stuff BEFORE Number. only pick nearest words
                            final: ['..', 'HWB Number', ':...']
                            or to simply put, extract all possible entries with :
                            """
                            # final parsed? group entries with colon
                            print("Original: ", [node.word for node in temp], " width: ", width)
                            thresh_nei = abs(width / 85)
                            print(thresh_nei)
                            groupped = list()
                            groupped_for_display = list()

                            for index, node in enumerate(temp):
                                if str(node.word).__contains__(":"):
                                    temp_ptr = index - 1
                                    temp_group = list()
                                    KEEP_CHECKIN = True
                                    print("===Merging===")
                                    while temp_ptr >= 0 and KEEP_CHECKIN:
                                        temp_group.insert(0, temp[temp_ptr])
                                        # if is neighbor, merge them
                                        if temp_ptr > 0:
                                            print(temp[temp_ptr].left - \
                                                    (temp[temp_ptr - 1].left + temp[temp_ptr - 1].width))
                                            if temp[temp_ptr].left - \
                                                    (temp[temp_ptr - 1].left + temp[temp_ptr - 1].width) <= thresh_nei:
                                                print(temp[temp_ptr].word)
                                                temp_ptr -= 1
                                            else:
                                                temp_ptr -= 1
                                                KEEP_CHECKIN = False
                                        else:
                                            KEEP_CHECKIN = False
                                        if temp_ptr < 0:
                                            break
                                    groupped.append(temp_group)
                                    groupped_for_display.append([node.word for node in temp_group])
                            print(groupped_for_display)
                            print("\nFINAL RESULTS: ", final_TEMP)
                            return final_parsed

                        # print([(n.word, n.label) for n in temp_merge_remove_space])
                        final_temp = list()
                        if len(temp_merge_remove_space) > 1:
                            final_temp = partition_nodes(temp_merge_remove_space)
                        else:
                            final_temp.append(temp_merge_remove_space)
                        # after clustered the labels, split each entry according to colon
                        for node_partition in final_temp:
                            new_node = merge_nodes_to_one(node_partition)
                            merged_nodes_list.append(new_node)
                            # add new node
                            words_raw_new.append(new_node)

                        # new_node = merge_nodes_to_one(temp_merge_remove_space)
                        # merged_nodes_list.append(new_node)
                        # words_raw_new.append(new_node)
                    current_index += 1
                # show test results
                # print([(node.word, node.left, node.id) for node in self.storage[key][key_line]])
                # print([([node.word for node in part]) for part in line_with_merged_nodes])
                self.storage[key][key_line] = merged_nodes_list

        return words_raw_new

    def euclidean_distance_right(self, node1=Node(), node2=Node()):
        return math.sqrt((pow((node1.center_y - node2.center_y), 2) + pow((node1.center_x - node2.left), 2)))

    def euclidean_distance_left(self, node1=Node(), node2=Node()):
        return math.sqrt((pow((node1.center_y - node2.center_y), 2) + pow((node1.left - node2.left - node2.width), 2)))

    def euclidean_distance(self, node1=Node(), node2=Node()):
        # return a distance between two nodes
        return math.sqrt((pow((node1.center_y - node2.center_y), 2) + pow((node1.center_x - node2.center_x), 2)))

    def right_node_process(self, row, key_line, node_with_id, min_distance, key_same_line):
        threshold_height_inline = 5
        threshold_height = 5
        threshold_importance = 5

        # print(node_with_id.word)

        for node in self.storage[row][key_line]:
            if node.id is not node_with_id.id:
                # check the right node instead
                # add some buffer region

                if (node_with_id.left + node_with_id.width) <= node.left + 10:
                    # set a threshold, the centroid height differences should be small
                    # or the top point should be closer to the bottom point
                    if abs(node_with_id.center_y - node.center_y) < threshold_height_inline or \
                       abs(node_with_id.top - node.top + node.height) < threshold_height or \
                       abs(node.top - node_with_id.top + node_with_id.height) < threshold_height:
                        temp_dist = self.euclidean_distance_right(node_with_id, node)
                        # update: key_same_line is no longer a good metric since word may not be in the same line
                        """
                        if row is not key_same_line:
                            temp_dist = temp_dist * threshold_importance
                        else:
                            temp_dist = temp_dist / threshold_importance
                        """
                        if temp_dist < min_distance:
                            min_distance = temp_dist
                            node_with_id.right_node_id = node.id
                            node_with_id.right_node_ptr = node
                            # 8/9/2020: save the temporary distance
                            # will calculate the relative distance later
                            node_with_id.right_node_dis = temp_dist

        return min_distance, node_with_id

    def get_right_node(self, node_with_id, key_same_line, key_line, last_key):
        """
        similar to left node,
        height differences must be limited, but consider the right side now
        :param node_with_id:
        :param key_same_line:
        :param key_line:
        :param last_key:
        :return:
        """
        min_distance = 999999

        if key_same_line == 0 or key_same_line == last_key:
            top_row = -1
            bottom_row = -1
            if key_same_line == 0:
                top_row = key_same_line
                bottom_row = 1
            elif key_same_line == last_key:
                top_row = last_key - 1
                bottom_row = last_key
            # scan two rows for node_with_id
            if top_row > -1:
                min_distance, node_with_id = self.right_node_process(top_row, key_line, node_with_id, min_distance, key_same_line)
            if bottom_row > -1:
                min_distance, node_with_id = self.right_node_process(bottom_row, key_line, node_with_id, min_distance, key_same_line)
        else:
            rows_to_watch = [key_same_line - 1, key_same_line, key_same_line + 1]
            for row_num in rows_to_watch:
                min_distance, node_with_id = self.right_node_process(row_num, key_line, node_with_id, min_distance, key_same_line)

        if node_with_id.right_node_id is 0 and key_same_line >= 0:
            # in this case, node is not in line 0 but it also pointed to node 0
            # just point to itself
            # UPDATE (25/08/2020): OCR is not that capable, words does not read in a fixed order!!
            for row_num in range(last_key):
                min_distance, node_with_id = self.right_node_process(row_num, key_line, node_with_id, min_distance, key_same_line)
        if node_with_id.right_node_id is 0 and key_same_line >= 0:
            node_with_id.right_node_id = node_with_id.id
            node_with_id.right_node_ptr = None
            node_with_id.right_node_dis = 0

        if min_distance == 999999:
            min_distance = 0
        return min_distance

    def left_node_process(self, row, key_line, node_with_id, min_distance, key_same_line):
        threshold_height_inline = 5
        threshold_height = 5
        threshold_importance = 5
        saved_i = -1
        save_dist = 0

        for index, node in enumerate(self.storage[row][key_line]):
            if node.id is not node_with_id.id:
                if (node.left + node.width) < node_with_id.left:
                    # set a threshold, the centroid height differences should be small
                    # or the top point should be closer to the bottom point
                    if abs(node_with_id.center_y - node.center_y) < threshold_height_inline or \
                       abs(node_with_id.top - node.top + node.height) < threshold_height or \
                       abs(node.top - node_with_id.top + node_with_id.height) < threshold_height:
                        # need to calculate distance based on left side, not centriod
                        temp_dist = self.euclidean_distance_left(node_with_id, node)
                        # add a threshold term to the distance such that node at the same line have higher importance
                        # interestingly, higher the value, lower is the importance (further away)
                        """
                        if row != key_same_line:
                            temp_dist = temp_dist * threshold_importance
                        else:
                            temp_dist = temp_dist / threshold_importance
                        """

                        if temp_dist < min_distance:
                            save_dist = temp_dist
                            min_distance = temp_dist
                            node_with_id.left_node_id = node.id
                            node_with_id.left_node_ptr = node
                            node_with_id.left_node_dis = temp_dist
                            saved_i = index
        if saved_i is not -1:
            pass
            # print(node_with_id.word, "left: ", self.storage[row][key_line][saved_i].word, "dist: ", save_dist)
        return min_distance, node_with_id

    def get_left_node(self, node_with_id, key_same_line, key_line, last_key):
        """
        :param node_with_id:
        :param key_same_line: the key number, which is at the same line as the node
        :param key_line:
        :param last_key:
        :return:
        """
        min_distance = 999999

        if key_same_line == 0 or key_same_line == last_key:
            top_row = -1
            bottom_row = -1
            if key_same_line == 0:
                top_row = key_same_line
                bottom_row = 1
            elif key_same_line == last_key:
                top_row = last_key - 1
                bottom_row = last_key
            # scan two rows for node_with_id
            if top_row > -1:
                min_distance, node_with_id = self.left_node_process(top_row, key_line, node_with_id, min_distance, key_same_line)
            if bottom_row > -1:
                min_distance, node_with_id = self.left_node_process(bottom_row, key_line, node_with_id, min_distance, key_same_line)
        else:
            # how about just check all the rows?
            # print("\n\n")
            for row_num in range(last_key):
                min_distance, node_with_id = self.left_node_process(row_num, key_line, node_with_id, min_distance, key_same_line)

            """
            rows_to_watch = [key_same_line - 1, key_same_line, key_same_line + 1]
            for row_num in rows_to_watch:
                min_distance, node_with_id = self.left_node_process(row_num, key_line, node_with_id, min_distance, key_same_line)
            """
        # if node_with_id.left_node_id == 0 and key_same_line >= 0:
            # in this case, node is not in line 0 but it also pointed to node 0
            # just point to itself
        if node_with_id.left_node_id == 0 and key_same_line >= 0:
            node_with_id.left_node_id = node_with_id.id
            node_with_id.left_node_ptr = None
            node_with_id.left_node_dis = 0
        if min_distance == 999999:
            min_distance = 0
        return min_distance

    def get_bottom_node(self, node_with_id=Node(), key_bottom=int(), key_line=int, last_key=int):
        # scan the bottom row
        min_distance = 999999
        init_key_bottom = key_bottom
        # the node is already at the bottom
        if (key_bottom - 1) == last_key:
            node_with_id.bottom_node_id = node_with_id.id
            min_distance = 0
        else:
            saved_dist = 0
            min_center_dist = 999999

            # modification: also need to watch 4 row down below
            included_rows = 1
            rows_to_watch = list()
            # search at most 1 rows down stairs
            # SCAN ALL THE ROWS INSTEAD
            while key_bottom.__int__() <= last_key.__int__() and included_rows <= 4:
                rows_to_watch.append(key_bottom)
                key_bottom += 1
                included_rows += 1

            # for key_bottom_defined in rows_to_watch:
            for key_bottom_defined in range(last_key):
                # each node in the whole line
                saved_i = -1
                for index, bottom_node_with_id in enumerate(self.storage[key_bottom_defined][key_line]):
                    if bottom_node_with_id.top > (node_with_id.top + node_with_id.height):
                        temp_dist = self.euclidean_distance(node_with_id, bottom_node_with_id)
                        temp_dist_height = abs(node_with_id.top - bottom_node_with_id.top)

                        if temp_dist < min_distance and temp_dist < 300:
                            saved_i = index
                            saved_dist = temp_dist
                            min_distance = temp_dist
                            node_with_id.bottom_node_id = bottom_node_with_id.id
                            node_with_id.bottom_node_ptr = bottom_node_with_id
                            node_with_id.bottom_node_dis = temp_dist

                if saved_i is not -1:
                    pass
                    # print(node_with_id.word, "bottom: ", self.storage[key_bottom_defined][key_line][saved_i].word, "dist: ", saved_dist)
        if node_with_id.bottom_node_id is 0 and init_key_bottom <= last_key:
            # scan all rows. OCR is not always available to cluster text
            node_with_id.bottom_node_id = node_with_id.id
            node_with_id.bottom_node_ptr = None
            node_with_id.bottom_node_dis = 0
        if min_distance == 999999:
            min_distance = 0
        return min_distance

    def get_top_node(self, node_with_id=Node(), key_top=int, key_line=int, last_key=int):
        init_key_top = key_top
        min_distance = 999999

        if key_top == -1:
            node_with_id.top_node_id = node_with_id.id
            min_distance = 0
        else:
            # check top row only
            # update: set rows to watch to 2 rows on top
            included_rows = 1
            rows_to_watch = list()
            while key_top.__int__() > -1 and included_rows <= 2:
                rows_to_watch.append(key_top)
                key_top -= 1
                included_rows += 1

            # for key_top_defined in rows_to_watch:
            for key_top_defined in range(last_key):
                for top_node_with_id in self.storage[key_top_defined][key_line]:
                    if top_node_with_id.top < node_with_id.top:
                        temp_dist = self.euclidean_distance(node_with_id, top_node_with_id)
                        if temp_dist < min_distance and temp_dist < 300:

                            min_distance = temp_dist
                            node_with_id.top_node_id = top_node_with_id.id
                            # 02092020 update: linked structure
                            node_with_id.top_node_ptr = top_node_with_id
                            node_with_id.top_node_dis = temp_dist

        if node_with_id.top_node_id is 0 and init_key_top > 0:
            node_with_id.top_node_id = node_with_id.id
            node_with_id.top_node_ptr = None
            node_with_id.top_node_dis = 0
        if min_distance == 999999:
            min_distance = 0
        return min_distance

    def generate_graph(self):
        """
        Generate graph based on line data
        center_x and center_y are modified here
        :return:
        """
        glw = GraphLineWeights()

        node_id = 0
        last_key = list(self.storage.keys())[-1]

        for key in tqdm.tqdm(self.storage):
            for key_line in self.storage[key]:
                for node in self.storage[key][key_line]:
                    # set unique node id and calculate centroid
                    node.id = node_id
                    node.center_x = node.left + int(node.width / 2)
                    node.center_y = node.top + int(node.height / 2)
                    node_id += 1
        for key in self.storage:
            for key_line in self.storage[key]:
                for node_with_id in self.storage[key][key_line]:
                    # print(node_with_id.word)
                    # print(node_with_id.left, node_with_id.top, node_with_id.width, node_with_id.height)
                    # consider 4 sides: top, right, bottom, left
                    # glw: 0 -> 1 -> 2 -> 3
                    # 1. top, verified
                    min_dist = self.get_top_node(node_with_id, key - 1, key_line, last_key)
                    glw.add_node_id_connection(node_with_id.id, 0, node_with_id.top_node_id, min_dist)
                    # 2. bottom
                    min_dist = self.get_bottom_node(node_with_id, key + 1, key_line, last_key)
                    glw.add_node_id_connection(node_with_id.id, 2, node_with_id.bottom_node_id, min_dist)
                    # 3. left
                    min_dist = self.get_left_node(node_with_id, key, key_line, last_key)
                    glw.add_node_id_connection(node_with_id.id, 3, node_with_id.left_node_id, min_dist)
                    # 4. right
                    min_dist = self.get_right_node(node_with_id, key, key_line, last_key)
                    glw.add_node_id_connection(node_with_id.id, 1, node_with_id.right_node_id, min_dist)

        return glw

    def write_to_file(self, dir):
        with open(dir, "w+", encoding='utf-8') as f:
            for key in self.storage:
                for key_sub in self.storage[key]:
                    string = str()
                    string = ' '.join([str(elem.word) for elem in self.storage[key][key_sub]])
                    string = string.strip()
                    string += "\n"
                    f.write(string)
        f.close()

    def use_parser_re(self, currency_dict):
        results = list()
        currency = list()
        for key in self.storage:
            for key_sub in self.storage[key]:
                # for each text line
                results_temp, currency_temp = parse_using_re(self.storage[key][key_sub], currency_dict=currency_dict)
                results += results_temp
                currency += currency_temp
        if len(currency) > 0:
            currency_confirm = decided_currency(currency)
        else:
            currency_confirm = None
        return results, currency_confirm


CopyOfSameLine = type('CopyOfSameLine', SameLine.__bases__, dict(SameLine.__dict__))



