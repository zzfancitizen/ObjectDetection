import tensorflow as tf
import numpy as np


class VGG_16(object):
    def __init__(self, input):
        self.input = tf.reshape(input, [-1, 224, 224, 3])

    def build_layer(self):
        # CONV layer
        # 224 * 224 * 3
        # conv1_1
        with tf.name_scope(name="conv1_1"):
            W_conv1_1 = self._weight_variable([3, 3, 3, 64])
            b_conv1_1 = self._bias_variable([64])
            h_conv1_1 = tf.nn.relu(self._conv2d(self.input, W_conv1_1) + b_conv1_1, name="output_conv11")
        # 224 * 224 * 64
        # conv1_2
        with tf.name_scope(name="conv1_2"):
            W_conv1_2 = self._weight_variable([3, 3, 64, 64])
            b_conv1_2 = self._bias_variable([64])
            h_conv1_2 = tf.nn.relu(self._conv2d(h_conv1_1, W_conv1_2) + b_conv1_2, name="output_conv12")
        # 224 * 224 * 64
        with tf.name_scope(name="max_pool1"):
            h_pool1 = self._max_pool_2x2(h_conv1_2)
        # 112 * 112 * 64
        # conv2_1
        with tf.name_scope(name="conv2_1"):
            W_conv2_1 = self._weight_variable([3, 3, 64, 128])
            b_conv2_1 = self._bias_variable([128])
            h_conv2_1 = tf.nn.relu(self._conv2d(h_pool1, W_conv2_1) + b_conv2_1, name="output_conv21")
        # 112 * 112 * 128
        # conv2_2
        with tf.name_scope(name="conv2_2"):
            W_conv2_2 = self._weight_variable([3, 3, 128, 128])
            b_conv2_2 = self._bias_variable([128])
            h_conv2_2 = tf.nn.relu(self._conv2d(h_conv2_1, W_conv2_2) + b_conv2_2, name="output_conv22")
        # 112 * 112 * 128
        with tf.name_scope(name="max_pool2"):
            h_pool2 = self._max_pool_2x2(h_conv2_2)

        with tf.name_scope(name="conv3_1"):
            W_conv3_1 = self._weight_variable([3, 3, 128, 256])
            b_conv3_1 = self._bias_variable([256])
            h_conv3_1 = tf.nn.relu(self._conv2d(h_pool2, W_conv3_1) + b_conv3_1, name="output_conv31")

        with tf.name_scope(name="conv3_2"):
            W_conv3_2 = self._weight_variable([3, 3, 256, 256])
            b_conv3_2 = self._bias_variable([256])
            h_conv3_2 = tf.nn.relu(self._conv2d(h_conv3_1, W_conv3_2) + b_conv3_2, name="output_conv32")

        with tf.name_scope(name="conv3_3"):
            W_conv3_3 = self._weight_variable([3, 3, 256, 256])
            b_conv3_3 = self._bias_variable([256])
            h_conv3_3 = tf.nn.relu(self._conv2d(h_conv3_2, W_conv3_3) + b_conv3_3, name="output_conv33")

        with tf.name_scope(name="max_pool3"):
            h_pool3 = self._max_pool_2x2(h_conv3_3)

        with tf.name_scope(name="conv4_1"):
            W_conv4_1 = self._weight_variable([3, 3, 256, 512])
            b_conv4_1 = self._bias_variable([512])
            h_conv4_1 = tf.nn.relu(self._conv2d(h_pool3, W_conv4_1) + b_conv4_1, name="output_conv41")

        with tf.name_scope(name="conv4_2"):
            W_conv4_2 = self._weight_variable([3, 3, 512, 512])
            b_conv4_2 = self._bias_variable([512])
            h_conv4_2 = tf.nn.relu(self._conv2d(h_conv4_1, W_conv4_2) + b_conv4_2, name="output_conv42")

        with tf.name_scope(name="conv4_3"):
            W_conv4_3 = self._weight_variable([3, 3, 512, 512])
            b_conv4_3 = self._bias_variable([512])
            h_conv4_3 = tf.nn.relu(self._conv2d(h_conv4_2, W_conv4_3) + b_conv4_3, name="output_conv43")

        with tf.name_scope(name="max_pool4"):
            h_pool4 = self._max_pool_2x2(h_conv4_3)

        with tf.name_scope(name="conv5_1"):
            W_conv5_1 = self._weight_variable([3, 3, 512, 512])
            b_conv5_1 = self._bias_variable([512])
            h_conv5_1 = tf.nn.relu(self._conv2d(h_pool4, W_conv5_1) + b_conv5_1, name="output_conv51")

        with tf.name_scope(name="conv5_2"):
            W_conv5_2 = self._weight_variable([3, 3, 512, 512])
            b_conv5_2 = self._bias_variable([512])
            h_conv5_2 = tf.nn.relu(self._conv2d(h_conv5_1, W_conv5_2) + b_conv5_2, name="output_conv52")

        with tf.name_scope(name="conv5_3"):
            W_conv5_3 = self._weight_variable([3, 3, 512, 512])
            b_conv5_3 = self._bias_variable([512])
            h_conv5_3 = tf.nn.relu(self._conv2d(h_conv5_2, W_conv5_3) + b_conv5_3, name="output_conv53")

        with tf.name_scope(name="max_pool5"):
            h_pool4 = self._max_pool_2x2(h_conv5_3)

        # fc layer
        with tf.name_scope(name="fc1"):
            h_pool4_flat = tf.reshape(h_pool4, [-1, 7 * 7 * 512])
            W_fc1 = self._weight_variable([7 * 7 * 512, 4096])
            b_fc1 = self._bias_variable([4096])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1, name="fc1")

        with tf.name_scope(name="fc2"):
            W_fc2 = self._weight_variable([4096, 4096])
            b_fc2 = self._bias_variable([4096])
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2, name="fc2")

        with tf.name_scope(name="fc3"):
            W_fc3 = self._weight_variable([4096, 1000])
            b_fc3 = self._bias_variable([1000])
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3, name="fc3")

        with tf.name_scope(name="output"):
            W_fc3 = self._weight_variable([1000, 1000])
            b_fc3 = self._bias_variable([1000])
            h_output = tf.add(tf.matmul(h_fc3, W_fc3), b_fc3, name="output")

            h_predict = tf.nn.softmax(h_output, name="predict")

        return h_predict

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, dtype=tf.float32, name="weight")

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, dtype=tf.float32, name="bias")

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name='conv2d')

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name="max_pool")


if __name__ == '__main__':
    pass
