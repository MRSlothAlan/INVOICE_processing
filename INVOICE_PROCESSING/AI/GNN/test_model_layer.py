import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from scipy.linalg import fractional_matrix_power

# a constant
FEATURE_LENGTH = 59


class GraphOperator(Layer):
    """
    only return the size of node for now,
    but I can do something else
    """
    def __init__(self):
        super(GraphOperator, self).__init__(dynamic=True)

    def call(self, adj):
        print("size returned", len(adj[0]))
        return len(adj[0])

    def compute_output_shape(self, input_shape):
        return 1


class GCN_layer(Layer):
    def __init__(self, NODE_SIZE):
        super(GCN_layer, self).__init__(dynamic=True)
        # the size of N
        self.node_size = NODE_SIZE

    def call(self, inputs):
        """
        format of input:
            0: N * N adjacency matrix
            1: N * E feature vectors of node
        :param inputs:
        :return:
        """
        assert len(inputs[0]) == len(inputs[1]), "shape of adjacency matrix" \
                                                 "and feature vectors does not match!"

        adj_matrix = np.array(inputs[0][0])
        feature_vectors = np.array(inputs[1][0])

        deg_matrix = self.cal_degree_matrix(adj_matrix)
        d_half_norm = fractional_matrix_power(deg_matrix, -0.5)
        DAX = deg_matrix.dot(adj_matrix).dot(feature_vectors)
        # a merged matrix with shape N * E
        # DADX = d_half_norm.dot(adj_matrix).dot(d_half_norm).dot(feature_vectors)
        # a hard code relu function
        # DADX_relu = np.maximum(DADX, 0)
        DAX_relu = np.maximum(DAX, 0)
        return [adj_matrix, np.array([DAX_relu])]

    def cal_degree_matrix(self, adj_mat):
        degree = np.zeros(len(adj_mat))
        # sum along column
        col_sum = adj_mat.sum(axis=0)
        # with self attention
        for j in range(0, len(adj_mat)):
            try:
                degree[j] = col_sum[0, j] + 1
            except IndexError:
                degree[j] = col_sum[j] + 1
        diagMat = np.diag(degree)
        return diagMat

    def compute_output_shape(self, input_shape=None):
        if input_shape is None:
            input_shape = [(None, None),
                           (None, FEATURE_LENGTH)]
        # the shape is basically the same
        return [input_shape[0], input_shape[1]]
