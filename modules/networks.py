import sys
sys.path.append('../') #root path
from SRCNN.modules.ops import *
from SRCNN.utils.other_utils import *
import tensorflow as tf

def SRCNN_915(input, scope_name, num_channels, reuse = tf.AUTO_REUSE, is_training=False, norm_type=None):
    BS, H, W, CH = input.get_shape().as_list()

    #initializer = tf.initializers.VarianceScaling()
    initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope_name, reuse = reuse):
        with tf.variable_scope("patch_extraction"):
            x = conv2D(input, name="conv1", ksize=9, strides=1, fsize=64, is_training=is_training, padding="SAME", initializer=initializer)

        with tf.variable_scope("non_linear_mapping"):
            x = tf.nn.relu(x)
            x = conv2D(x, name="conv1", ksize=1, strides=1, fsize=32, is_training=is_training, padding="SAME", initializer=initializer)
            x = tf.nn.relu(x)

        with tf.variable_scope("reconstruction"):
            x = conv2D(x, name="conv1", ksize=5, strides=1, fsize=num_channels, is_training=is_training, padding="SAME", initializer=initializer)

    return x

