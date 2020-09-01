"""

"""
from tensorflow.keras.layers import Conv2D, Input, Dense
from tensorflow import keras
from tensorflow.keras.utils import plot_model


def alexNet():
    input_shape = (1, 224, 224, 1)
    img_in = Input(shape=(224, 224, 1), batch_size=None)
    c1 = Conv2D(filters=2, kernel_size=3, activation='relu', padding="same", input_shape=input_shape[1:])(img_in)
    out = Dense(1)(c1)
    model = keras.Model(inputs=[img_in], outputs=[out])
    return model


m = alexNet()
m.compile(optimizer="adam", loss="mse")
plot_model(m, to_file="model.png", show_shapes=True)

# https://stackoverflow.com/questions/55882176/how-to-fix-the-batch-size-in-keras
# https://keras.io/api/layers/convolution_layers/convolution2d/


