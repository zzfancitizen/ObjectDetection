import tensorflow as tf

if __name__ == '__main__':
    a = tf.constant(1.)
    b = tf.constant(1.)
    b = 2 * a

    g = tf.gradients(a + b, [a, b])[0]

    print(g)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(g))
