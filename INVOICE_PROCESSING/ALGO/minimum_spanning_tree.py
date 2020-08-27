"""
Make use of graph to perform minimum spanning tree to analysis layout
"""
from random import randint
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import *
import cv2


class GraphLineWeights:
    """
    members:
    node_id_connection: {id: [[top, weight], [right, weight], [bottom, weight], [left, weight]]

    ** modified to:
        {id: {top_id: w, right_id: w, bottom_id: w, left_id: w}}
    weight = absolute distances of nodes
    """
    def __init__(self):
        self.node_id_connections = dict()

    def add_node_id_connection(self, id_self, indicator=-1, id_node=int(), id_node_weight=int()):
        if id_self in self.node_id_connections.keys():
            if indicator is not -1:
                try:
                    self.node_id_connections[id_self][id_node] = id_node_weight
                except KeyError as e:
                    self.node_id_connections[id_self][id_node] = float()
                    self.node_id_connections[id_self][id_node] = id_node_weight
        else:
            # create a new entry
            self.node_id_connections[id_self] = dict()
            if id_node not in self.node_id_connections[id_self].keys():
                self.node_id_connections[id_self][id_node] = float()
                self.node_id_connections[id_self][id_node] = id_node_weight

    def print(self):
        for key in self.node_id_connections:
            print("node id: ", key)
            print("      Connected nodes: ")
            print("      ", self.node_id_connections[key])


class GraphMST:
    """
    A minimum spanning tree
    structure:
        MST: {id: {id_: w, id_: w, id_: w, ... (keep appending)}, id: {}, ...}
        added_nodes: set of added nodes
    """
    def __init__(self, init_id):
        self.MST = dict()
        self.added_nodes = set()

        def add_node_init_constructor(node_id):
            """
            An initialization of node
            :param node_id:
            :return:
            """
            if self.added_nodes.__len__() is not 0:
                self.added_nodes.clear()
            if self.MST.__len__() is not 0:
                self.MST.clear()
            self.added_nodes.add(node_id)
            self.MST[node_id] = dict()
        add_node_init_constructor(init_id)

    def is_in_tree(self, node_id):
        return node_id in self.added_nodes

    def add_node_init(self, node_id):
        """
        An initialization of node
        :param node_id:
        :return:
        """
        self.added_nodes.add(node_id)
        self.MST[node_id] = dict()

    def add_node(self, node_id, rel_child_id, weight):
        """
        add a relative child to the node
        :param node_id:
        :param rel_child_id:
        :param weight:
        :return:
        """
        if node_id not in self.MST.keys():
            self.MST[node_id] = dict()
        if rel_child_id not in self.MST[node_id].keys():
            self.MST[node_id][rel_child_id] = 0.0
        self.MST[node_id][rel_child_id] = weight

    def add_edge(self, edge_minimum_weight):
        """
        :param edge_minimum_weight: one edge with minimum weight
        :return:
        """
        if len(edge_minimum_weight) > 0:
            node_id_start = edge_minimum_weight[0]
            node_id_end = edge_minimum_weight[1]
            weight = edge_minimum_weight[2]

            self.added_nodes.add(node_id_start)
            self.added_nodes.add(node_id_end)
            # append to a dictionary of nodes
            self.add_node(node_id_start, node_id_end, weight)
            self.add_node(node_id_end, node_id_start, weight)

    def print_mst(self):
        for key in self.MST:
            print("Node: ", key)
            print(self.MST[key])

    def draw_mst_on_graph(self, word_raw, image, resize_ratio=resize_ratio):
        drew_node = set()
        for key in self.MST:
            for key_child in self.MST[key]:
                if key not in drew_node or key_child not in drew_node:
                    drew_node.add(key)
                    nodes = [node for node in word_raw if node.id is key or node.id is key_child]
                    if len(nodes) > 1:
                        image = cv2.line(image, (int(nodes[0].center_x * resize_ratio),
                                                 int(nodes[0].center_y * resize_ratio)),
                                                (int(nodes[1].center_x * resize_ratio),
                                                 int(nodes[1].center_y * resize_ratio)),
                                                (100, 0, 255), 2)
        return image

    def k_spanning_tree(self):
        return NotImplementedError()

"""
Utilities to calculate the MST
"""


def find_candidate_edges(glw, mst):
    candidate_edges = list()
    for key in glw.node_id_connections:
        if mst.is_in_tree(key):
            for key_connected in glw.node_id_connections[key]:
                if not mst.is_in_tree(key_connected):
                    # this edge has at most one node in graph, but not for the second node
                    node_id = key_connected
                    node_id_in_graph = key
                    edge_weight = glw.node_id_connections[key][key_connected]
                    candidate_edges.append([node_id_in_graph, node_id, edge_weight])
        if not mst.is_in_tree(key):
            for key_connected in glw.node_id_connections[key]:
                if mst.is_in_tree(key_connected):
                    # this edge has at most one node in graph, but not for the second node
                    node_id = key_connected
                    node_id_in_graph = key
                    edge_weight = glw.node_id_connections[key][key_connected]
                    candidate_edges.append([node_id_in_graph, node_id, edge_weight])

    return candidate_edges


"""
Your failed implementation:

A couple of things to do self revision:
    You are not always correct
    You need to have great logical skills to out-perform others in every field
    It is 

def swap(edge_list, index, index_sec):
    temp = edge_list[index]
    edge_list[index] = edge_list[index_sec]
    edge_list[index_sec] = temp
    return edge_list
    
    
    while pointer < length:
        temp = edge_list[pointer][1]
        while pointer <= (length - 2) and temp < edge_list[pointer + 1][1]:
            pointer += 1
            print("pointer now", pointer)
            if pointer > 1:
                temp = edge_list[pointer - 1][1]
        print("{} vs {}".format(temp, edge_list[pointer][1]))
        if temp > edge_list[pointer][1]:
            print("pointer now: ", pointer)
            pointer_temp = pointer + 1
            temp = edge_list[pointer_temp][1]
            while pointer_temp > 0 and temp < edge_list[pointer_temp - 1][1]:
                pointer_temp -= 1
            edge_list = swap(edge_list, pointer_temp, pointer)
        pointer += 1
    return edge_list
"""


def insertion_sort(edge_list):
    for i in range(1, len(edge_list)):
        key = edge_list[i][2]
        j = i - 1
        while j >= 0 and key < edge_list[j][2]:
            temp = edge_list[j + 1]
            edge_list[j + 1] = edge_list[j]
            edge_list[j] = temp
            j -= 1
        edge_list[j + 1][2] = key
    return edge_list


def generate_mst_graph(glw=GraphLineWeights(), tot_nodes=0):
    """
    :param glw:
    :param tot_nodes:
    :return:
    """
    """
    CAN HAVE MULTIPLE MSTs if some part of the invoice disconnected
    """
    # structure of glw: top -> right -> bottom -> left
    # randomly pick a starting node, from 0 to last_key
    starting_node_id = randint(0, tot_nodes)
    mst = GraphMST(init_id=starting_node_id)
    missed = 0
    missed_node_threshold = 30
    while mst.added_nodes.__len__() <= tot_nodes:
        list_of_edges = find_candidate_edges(glw, mst)
        try:
            edge_min_weight = insertion_sort(list_of_edges)[0]
            # allow zero edges
            mst.add_edge(edge_min_weight)
            missed = 0
        except IndexError as e:
            if abs(tot_nodes - len(mst.added_nodes)) > missed_node_threshold:
                # there are too many node missed.
                # Reason: nodes disconnected
                # Solution: randomly select a new starting point, which is not in added node
                while starting_node_id in mst.added_nodes:
                    starting_node_id = randint(0, tot_nodes)
                mst.add_node_init(starting_node_id)
                missed = 0
            missed += 1
            if missed >= 10:
                # most likely it cannot connect to one or two outlier.
                # don't worry, say goodbye anyway
                return mst, starting_node_id

    return mst, starting_node_id
