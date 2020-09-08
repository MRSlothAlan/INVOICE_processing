"""
try to fit the model
"""
import pickle
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.GNN.test_model import retrieve_model
import numpy as np

input_file = open("0.txt", 'rb')
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
    if len(data[2][key]) == 59:
        gcn_node_feature.append(data[2][key])
    else:
        gcn_node_feature.append(data[2][key][:59])

gcn_adj_matrix = np.array([gcn_adj_matrix])
gcn_node_feature = np.array([gcn_node_feature])
node_label_list = np.array([node_label_list])

model = retrieve_model()
# there are data with different length
# therefore, some data is list in np array

"""
work to do:
support multiple value inputs
dataset generation need to be more careful
"""
model.fit([gcn_node_feature,
          gcn_adj_matrix],
          node_label_list, batch_size=None, verbose=1, epochs=5)
