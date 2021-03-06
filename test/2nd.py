import tensorflow as tf

if __name__ == '__main__':
    deltas = tf.reshape([[0.1, 0, 0.3], [0.1, 0.2, 0.3]], (2, 3))
    sigma2 = 1
    deltas_abs = tf.abs(deltas)
    smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
    x = tf.reduce_sum(tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                      (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1))
    with tf.Session() as sess:
        print(sess.run(x))

        # a = tf.constant(3., dtype=tf.float32)
        # b = tf.constant(2., dtype=tf.float32)
        #
        # flag = tf.less(a, b)
        #
        # init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        #
        # with tf.Session() as sess:
        #     sess.run(init)
        #
        #     print(sess.run(flag))
        #     print(tf.__version__)
