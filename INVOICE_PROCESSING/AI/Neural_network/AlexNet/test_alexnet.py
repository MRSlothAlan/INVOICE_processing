import tensorflow as tf
from tensorflow.keras.utils import plot_model


# cnn layer with Relu activation
def conv2D(x, W, b, stride_size):
    xW = tf.nn.conv2d(x, W, strides=[1, stride_size, stride_size, 1], padding='SAME')
    z = tf.nn.bias_add(xW, b)
    a = tf.nn.relu(z)
    return a


# max pooling layer
def maxPooling2D(x, kernel_size, stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')


# Dense Layer
def dense(x, W, b):
    z = tf.add(tf.matmul(x, W), b)
    a = tf.nn.relu(z)
    return a

# (4) Define AlexNet
# Setting some parameters

w_init = tf.keras.initializers.GlorotNormal()
batch_size = 18
epochs = 1
progress = 40
n_classes = 17


# Function, x is the input features
def alexNet(img_input):
    # 1st Convolutional Layer
    w_c1 = tf.compat.v1.get_variable('w_c1', [11, 11, 3, 96], initializer=w_init)
    b_c1 = tf.Variable(tf.zeros([96]))
    c1 = conv2D(img_input, w_c1, b_c1, stride_size=4)
    # Pooling
    p1 = maxPooling2D(c1, kernel_size=2, stride_size=2)
    # Batch Normalisation
    bn1 = tf.compat.v1.layers.batch_normalization(p1)

    # 2nd Convolutional layer
    w_c2 = tf.compat.v1.get_variable('w_c2', [5, 5, 96, 256], initializer=w_init)
    b_c2 = tf.Variable(tf.zeros([256]))
    c2 = conv2D(bn1, w_c2, b_c2, stride_size=1)
    # Pooling
    p2 = maxPooling2D(c2, kernel_size=2, stride_size=2)
    # Batch Normalisation
    bn2 = tf.compat.v1.layers.batch_normalization(p2)

    # 3rd Convolutional Layer
    w_c3 = tf.compat.v1.get_variable('w_c3', [3, 3, 256, 384], initializer=w_init)
    b_c3 = tf.Variable(tf.zeros([384]))
    c3 = conv2D(bn2, w_c3, b_c3, stride_size=1)
    # Batch Normalisation
    bn3 = tf.compat.v1.layers.batch_normalization(c3)

    # 4th Convolutional Layer
    w_c4 = tf.compat.v1.get_variable('w_c4', [3, 3, 384, 384], initializer=w_init)
    b_c4 = tf.Variable(tf.zeros([384]))
    c4 = conv2D(bn3, w_c4, b_c4, stride_size=1)
    # Batch Normalisation
    bn4 = tf.compat.v1.layers.batch_normalization(c4)

    # 5th Convolutional Layer
    w_c5 = tf.compat.v1.get_variable('w_c5', [3, 3, 384, 256], initializer=w_init)
    b_c5 = tf.Variable(tf.zeros([256]))
    c5 = conv2D(bn4, w_c5, b_c5, stride_size=1)
    # Pooling
    p3 = maxPooling2D(c5, kernel_size=2, stride_size=2)
    # Batch Normalisation
    bn5 = tf.compat.v1.layers.batch_normalization(p3)

    # Flatten the conv layer - features has been reduced by pooling 3 times: 224/2*2*2
    flattened = tf.reshape(bn5, [-1, 28 * 28 * 256])

    # 1st Dense layer
    w_d1 = tf.compat.v1.get_variable('w_d1', [28 * 28 * 256, 4096], initializer=w_init)
    b_d1 = tf.Variable(tf.zeros([4096]))
    d1 = dense(flattened, w_d1, b_d1)
    # Dropout
    dropout_d1 = tf.nn.dropout(d1, 0.6)

    # 2nd Dense layer
    w_d2 = tf.compat.v1.get_variable('w_d2', [4096, 4096], initializer=w_init)
    b_d2 = tf.Variable(tf.zeros([4096]))
    d2 = dense(dropout_d1, w_d2, b_d2)
    # Dropout
    dropout_d2 = tf.nn.dropout(d2, 0.6)

    # 3rd Dense layer
    w_d3 = tf.compat.v1.get_variable('w_d3', [4096, 1000], initializer=w_init)
    b_d3 = tf.Variable(tf.zeros([1000]))
    d3 = dense(dropout_d2, w_d3, b_d3)
    # Dropout
    dropout_d3 = tf.nn.dropout(d3, 0.6)

    # Output layer
    w_out = tf.compat.v1.get_variable('w_out', [1000, n_classes], initializer=w_init)
    b_out = tf.Variable(tf.zeros([n_classes]))
    out = tf.add(tf.matmul(dropout_d3, w_out), b_out)

    return out


tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
predictions = alexNet(x)

# you may need to replicate this model using keras
plot_model(predictions, to_file="model.png", show_shapes=True)
