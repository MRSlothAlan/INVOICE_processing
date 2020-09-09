"""
try to fit the model
"""
import pickle
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.test_model import retrieve_model
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.model_constant import *
import numpy as np
from pathlib import Path
from os import listdir
from tqdm import tqdm


def train_test_split(l):
    train = l[0: int(len(l) * TRAIN_RATIO)]
    test = l[int(len(l) * TRAIN_RATIO): int(len(l))]
    return train, test


# load a input file
processed_dir = Path.cwd().parents[1] / "AI/GNN/processed_GCN"
processed_files = listdir(str(processed_dir))

# prepare data set to train
gcn_adj_matrix_ALL = []
gcn_node_feature_ALL = []
node_label_list_ALL = []

# parse all data
for index, input_name in enumerate(processed_files):
    input_file = open(str(processed_dir / input_name), 'rb')

    data = pickle.load(input_file)
    """
    data_to_save = [gcn_node_label, gcn_adj_matrix, gcn_node_feature, CLASS_LIST_LEN, NODE_SIZE, FEATURE_LEN]
    """
    # convert dict to lists
    node_label_list = list()
    for key in sorted(data[0].keys()):
        node_label_list.append(data[0][key])

    gcn_adj_matrix = data[1]

    gcn_node_feature = list()
    for key in sorted(data[2].keys()):
        gcn_node_feature.append(data[2][key])

    gcn_adj_matrix = np.array([gcn_adj_matrix])
    gcn_node_feature = np.array([gcn_node_feature])
    node_label_list = np.array([node_label_list])

    node_label_list_ALL.append(node_label_list)
    gcn_node_feature_ALL.append(gcn_node_feature)
    gcn_adj_matrix_ALL.append(gcn_adj_matrix)

    # print(gcn_adj_matrix.shape)
    # print(gcn_node_feature.shape)
    # print(node_label_list.shape)

print("Length of data sets",
      len(node_label_list_ALL),
      len(gcn_node_feature_ALL),
      len(gcn_adj_matrix_ALL))

# do standard train test split
gcn_adj_matrix_ALL_train, gcn_adj_matrix_ALL_test = train_test_split(gcn_adj_matrix_ALL)
gcn_node_feature_ALL_train, gcn_node_feature_ALL_test = train_test_split(gcn_node_feature_ALL)
node_label_list_ALL_train, node_label_list_ALL_test = train_test_split(node_label_list_ALL)


"""
potential bug:
word feature generation fail WITHOUT error, WITHOUT warning
    if you need to make an investigation, it will take a lot of time
"""

model = retrieve_model()
# there are data with different length
# therefore, some data is list in np array

epoches = 20
for epoch in range(0, epoches):
    print("\n\n\nEpoch{}\n\n\n".format(epoch))
    for index, gcn_adj_matrix in tqdm(enumerate(gcn_adj_matrix_ALL_train)):
        # fit into the model one by one
        model.fit([gcn_node_feature_ALL_train[index],
                   gcn_adj_matrix],
                   node_label_list_ALL_train[index], batch_size=None, verbose=1)

"""
for index, gcn_adj_mat in enumerate(gcn_adj_matrix_ALL_train):
    model.fit([gcn_node_feature_ALL[index],
               gcn_adj_matrix_ALL[index]],
               node_label_list_ALL[index], batch_size=None, verbose=1, epochs=20)
"""
# test the model
avg_loss = 0
avg_accuracy = 0

for index, gcn_adj_mat_test in enumerate(gcn_adj_matrix_ALL_test):
    loss, accuracy = model.evaluate([gcn_node_feature_ALL_test[index],
                                     gcn_adj_mat_test],
                                     node_label_list_ALL_test[index], verbose=2)
    avg_loss += loss
    avg_accuracy += accuracy

avg_loss /= len(gcn_adj_matrix_ALL_test)
avg_accuracy /= len(gcn_adj_matrix_ALL_test)

print("average loss: {}, average accuracy: {}".format(avg_loss, avg_accuracy))






