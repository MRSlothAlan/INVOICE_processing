"""
proposed_model.py
"""
from tensorflow import keras
from tensorflow.keras import layers
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.test_model_layer import GCN_layer, GraphOperator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.model_constant import *
from keras_gcn import GraphConv

# there is no uniform matrix size,
# therefore a Dense layer should be applied at the first place


def GCN_model(CLASS_LEN):
    """
    The function to generate the GCN model
    :param CLASS_LEN:
    :return:
    """
    # typically, the shape is N * 109
    node_feature_input = keras.Input(shape=(None, FEATURE_LENGTH), batch_size=None)
    # shape = N * N
    node_adj_matrix_input = keras.Input(shape=(None, None), batch_size=None)

    node_size = GraphOperator()(node_adj_matrix_input)

    # the GCN layer, custome
    """
    gcn_1 = GCN_layer(NODE_SIZE=None)(inputs=[node_adj_matrix_input, node_feature_input])
    gcn_2 = GCN_layer(NODE_SIZE=None)(inputs=gcn_1)
    gcn_3 = GCN_layer(NODE_SIZE=None)(inputs=gcn_2)
    gcn_4 = GCN_layer(NODE_SIZE=None)(inputs=gcn_3)
    """
    # implementation made by others
    conv_layer = GraphConv(units=CLASS_LEN, step_num=2,)([node_feature_input, node_adj_matrix_input])

    # shape = N * ~25 (length of classes)
    # gcn_4[1]
    node_labels = layers.Dense(CLASS_LEN, batch_size=None, activation="relu", name="Classifier")(conv_layer)

    model = keras.Model([node_feature_input, node_adj_matrix_input],
                        node_labels)
    return model


def retrieve_model():

    model = GCN_model(CLASS_LEN=45)
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    plot_model(model, to_file="model.png", show_shapes=True)
    return model