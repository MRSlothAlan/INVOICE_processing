"""
model.py

base class of model
Plan: Model to use: SVM (testing)

all input words: set to lower-cases, keep punctuations
much easier

labels to detect: (focus today)
    0: invoice number     (e.g. invoice number: or invoice no:)
    1: invoice date       (e.g. invoice date)
    2: invoice total      (e.g. total:(hkd) or total:))
    3: supplier           (e.g. company name)
    4: others
        ignore this word

currency: later
"""
from pathlib import Path
from os import listdir
from os.path import join, isfile
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.constants import CHAR_ALLOWED_STR, CHAR_ALLOWED, DEBUG_DETAILED, DEBUG, PARSE_ENTRY_NAME
from joblib import dump, load


def svm_test_model():
    measurements = list()
    y_input = list()
    TRAIN_SIZE = 0.7

    dataset_directory = Path.cwd().parents[1] / "dataset"
    datasets = [f for f in listdir(str(dataset_directory))
                if (isfile(join(str(dataset_directory), f)))]
    DATA_COLUMN = "text"
    LABEL_COLUMN = "label"
    column_names = [DATA_COLUMN, LABEL_COLUMN]

    unique_labels = ['invoice_number', 'supplier', 'invoice_due_date', 'total', "others"]
    data_set_df = pd.DataFrame(columns=column_names)

    for index, text_file_name in enumerate(datasets):
        # print("Processing file: {}".format(text_file_name))
        with open(str(dataset_directory / text_file_name), 'r+', encoding='utf-8') as text_f:
            for line in text_f:
                text = line.split("`")[0].strip()
                label = line.split("`")[1].split("\n")[0].strip()
                # print("Text: ", text, "Label: ", label)
                # encode label to integer format
                # label_index = unique_labels.index(label)
                label_index = label
                y_input.append(label_index)
                data_set_df.loc[index] = [text] + [label_index]
                # append to the list
                entry = dict()
                entry[DATA_COLUMN] = str()
                entry[LABEL_COLUMN] = int()
                entry[DATA_COLUMN] = text
                entry[LABEL_COLUMN] = label

                measurements.append(entry)
            text_f.close()

        # step 1: convert dataset to BERT required format
        row_num = len(data_set_df.index)
        train_length = int(row_num * TRAIN_SIZE)

        train_df = data_set_df[:train_length]
        test_df = data_set_df[train_length:]

        vec = DictVectorizer()

        # split list
        length_mea = len(measurements)
        train_length = int(length_mea * TRAIN_SIZE)

        # mea_train = measurements[:train_length]
        # mea_test = measurements[train_length:]
        # print(len(mea_train), len(mea_test))

        y_train = y_input
        y_test = y_input[train_length:]

        # transform all the words into a standard one_hot encoding, instead of using fit_transform
        # X = vec.fit_transform(measurements).toarray()

        X = list()
        for data in measurements:
            one_hot = [0.0] * len(CHAR_ALLOWED_STR)
            for char in data['text'].lower():
                if char in CHAR_ALLOWED:
                    one_hot[CHAR_ALLOWED_STR.index(char)] += 1.0
            X.append(one_hot)

        # random_string = ["invoice no: In2132131 Date: 12312123 Amount: $212", "amount due: $21322", "date: 12-03-2020"]
        X_rand = list()

        X_train = X
        X_test = X[train_length:]

        X_test_string = measurements[train_length:]

        # clf = RandomForestClassifier(n_estimators=20, verbose=1)
        clf = SVC(probability=True)
        clf.fit(X_train, y_train)

        y_pred_prob = clf.decision_function(X_test)

        y_pred = clf.predict(X_test)
        print("Size of train: {}, size of test: {}".format(len(X_train), len(X_test)))
        print("Accuracy: {}%".format(accuracy_score(y_test, y_pred) * 100))
        if DEBUG:
            print("Some examples: ")
            for i in range(len(y_test)):
                if y_test[i] is not 0:
                    print("Original string: ", X_test_string[i])
                    print("Correct label: ", y_test[i])
                    print("Predicted label: ", y_pred[i])
                    print("Probabilities: ", y_pred_prob[i])
        # save
        dump(clf, PARSE_ENTRY_NAME)


svm_test_model()


