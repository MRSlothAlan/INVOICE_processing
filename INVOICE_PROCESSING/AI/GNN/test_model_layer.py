import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from scipy.linalg import fractional_matrix_power
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.model_constant import *
from scipy.linalg import eigh
from keras_gcn import GraphConv


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
        print(inputs[0].shape, inputs[1].shape)
        assert len(inputs[0]) == len(inputs[1]), "length of adjacency matrix" \
                                                 "and feature vectors does not match!"
        # batch size = None, so loop over all inputs
        # given number of adj mat = number of feature vector
        adj_matrix = np.array(inputs[0][0])
        feature_vectors = np.array(inputs[1][0])
        deg_matrix = self.cal_degree_matrix(adj_matrix)

        d_half_norm = fractional_matrix_power(deg_matrix, -0.5)
        eye_matrix = np.identity(adj_matrix.shape[0])
        laplacian_matrix = eye_matrix - d_half_norm.dot(adj_matrix).dot(d_half_norm)
        eigen_lap_matrix = np.linalg.eig(laplacian_matrix)
        # let's try to only make the L matrix as the filter
        max_eigen_val = max(eigen_lap_matrix[0])
        eigen_lap_matrix_e = ((2 / float(max_eigen_val)) * eigen_lap_matrix[1]) - eye_matrix
        # the temporary filter
        result = eigen_lap_matrix_e.dot(feature_vectors)
        # the expected shape is N*E
        result_relu = np.maximum(result, 0)
        """
        # DAX = deg_matrix.dot(adj_matrix).dot(feature_vectors)
        # a merged matrix with shape N * E
        # DAX_relu = np.maximum(DAX, 0)
        # DAX_relu_expanded = np.expand_dims(DAX_relu, axis=0)
        """
        result_relu_expanded = np.expand_dims(result_relu, axis=0)
        # expand the dimension of adj_matrix by one again
        adj_matrix = np.expand_dims(adj_matrix, axis=0)
        return [adj_matrix, result_relu_expanded]

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
