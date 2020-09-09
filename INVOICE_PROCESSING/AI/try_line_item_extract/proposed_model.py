"""
Proposed model for image classification only
custom image detection will be implemented later
"""
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Layer

# define layer constants in here
LOCATION_SIZE = 4
WORD_ENCODING_SIZE = 54
# usually, image have a rectangular shape
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_COLOR_CHANNEL = 3

AREA_SIZE = 1
# there are 4 labels: address, line items, header, others
LABEL_SIZE = 4


class FeatureMerging(Layer):
    """
    A layer for merging features
    """
    def __init__(self):
        super(FeatureMerging, self).__init__(dynamic=True)

    def build(self, input_shape):
        return NotImplementedError()

    def call(self, inputs, **kwargs):
        return NotImplementedError()

    # for the calculation of the output shape of te layer
    def compute_output_shape(self, input_shape):
        return NotImplementedError()


def testing_model():
    """
    :return:
    """
    image_input = keras.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_COLOR_CHANNEL), batch_size=None)
    location_input = keras.Input(shape=LOCATION_SIZE, batch_size=None)
    word_encoding_input = keras.Input(shape=WORD_ENCODING_SIZE, batch_size=None)
    area_input = keras.Input(shape=AREA_SIZE, batch_size=None)

    # self-defined layer for feature merging
    conv_1 = layers.Conv2D(64, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_COLOR_CHANNEL))\
        (image_input)
    conv_2 = layers.Conv2D(64, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_COLOR_CHANNEL))\
        (conv_1)
    pool_1 = layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=None)\
        (conv_2)
    conv_3 = layers.Conv2D(128, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(128, 128, 128))\
        (pool_1)
    conv_4 = layers.Conv2D(128, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(128, 128, 128))\
        (conv_3)
    conv_5 = layers.Conv2D(128, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(128, 128, 128))\
        (conv_4)
    pool_2 = layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=None)\
        (conv_5)
    conv_6 = layers.Conv2D(256, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(64, 64, 256))\
        (pool_2)
    conv_7 = layers.Conv2D(256, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(64, 64, 256))\
        (conv_6)
    conv_8 = layers.Conv2D(256, kernel_size=(1, 1),
                           activation='relu',
                           input_shape=(64, 64, 256))\
        (conv_7)
    pool_3 = layers.MaxPooling2D(pool_size=(2, 2),
                                 strides=None)\
        (conv_8)
    # flattern
    # 960000
    flatten_1 = layers.Flatten()(pool_3)
    # dense
    dense_1 = layers.Dense(256, batch_size=None, activation='sigmoid', name="Dense_1")\
        (flatten_1)
    dense_2 = layers.Dense(128, batch_size=None, activation='sigmoid', name="Dense_2")\
        (dense_1)

    node_labels = layers.Dense(LABEL_SIZE, batch_size=None, activation='sigmoid', name='Classifier')\
        (dense_2)

    # then, use this dense output, merge with other features.


    """
        model = keras.Model(inputs=[image_input, location_input, word_encoding_input],
                        outputs=[node_labels])
    """
    model = keras.Model(inputs=[image_input],
                        outputs=[node_labels])
    return model


def retrieve_model():
    model = testing_model()
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    plot_model(model, to_file="model.png", show_shapes=True)
    return model
