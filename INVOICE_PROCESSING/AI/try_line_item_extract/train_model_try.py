from GRAPH_AND_TEXT_FEATURES.INVOICE_PROCESSING.AI.try_line_item_extract.proposed_model import retrieve_model
import pickle
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

TRAIN_SPLIT_RATIO = 0.7


def train_test_split(l):
    train = l[0: int(len(l) * TRAIN_SPLIT_RATIO)]
    test = l[int(len(l) * TRAIN_SPLIT_RATIO): int(len(l))]
    return train, test


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def train_model():
    # load dataset
    cropped_image_path = Path.cwd() / "processed/crop"

    input_file = open("dataset.txt", 'rb')
    dataset = pickle.load(input_file)

    location_list = list()
    area_list = list()
    word_embedding_list = list()
    label_list = list()
    loaded_image_list = list()

    for id in dataset:

        location_list.append(dataset[id][0])
        area_list.append([dataset[id][1]])
        word_embedding_list.append(dataset[id][2])

        # one-hot encode the label
        label = dataset[id][3]
        new_one_hot = [0.0] * 4
        new_one_hot[label] = 1.0
        label_list.append(list(new_one_hot))

        image = cv2.imread(str(cropped_image_path / str(str(id) + ".jpg")), 1)
        resize = cv2.resize(image, (256, 256))
        loaded_image_list.append(list(resize.tolist()))

    model = retrieve_model()
    train_image, test_image = train_test_split(loaded_image_list)
    train_label, test_label = train_test_split(label_list)
    train_word_em, test_word_em = train_test_split(word_embedding_list)
    train_area, test_area = train_test_split(area_list)
    train_location, test_location = train_test_split(location_list)

    # split data set into batches
    train_image_batches = list(chunks(train_image, 5))
    train_label_batches = list(chunks(train_label, 5))

    # try fit the model
    epoches = 10
    for epoch in tqdm(range(epoches)):
        for index, batch in enumerate(train_image_batches):
            model.fit(batch, train_label_batches[index], batch_size=None, verbose=1)
            model.reset_states()
    loss, acc = model.evaluate([test_image], [test_label], batch_size=None, verbose=1)
    print("Test lost: ", loss)
    print("Test accuracy: ", acc)


if __name__ == "__main__":
    train_model()