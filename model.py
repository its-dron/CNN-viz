import tensorflow as tf


def spatial_dropout(name, input, keep_prob=0.5, is_training=False):
    '''
    Randomly Drops entire channels

    Thanks Brian
    '''
    noise_shape = [1, 1, 1, tf.shape(input)[3]]
    keep_prob = tf.cond(tf.equal(is_training, True),
                        lambda: tf.constant(keep_prob),
                        lambda: tf.constant(1.0))
    return tf.nn.dropout(input,
                         keep_prob=keep_prob,
                         noise_shape=noise_shape,
                         name=name)


def res_block(name, input, filters, kernel, keep_prob=0.5, is_training=False):
    with tf.variable_scope(name):
        with tf.variable_scope('conv'):
            conv1 = tf.layers.conv2d(input,
                                     filters,
                                     kernel,
                                     use_bias=False,
                                     padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn1 = tf.layers.batch_normalization(conv1,
                                               scale=False,
                                               training=is_training,
                                               trainable=is_training)
            relu1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d(relu1,
                                     filters,
                                     kernel,
                                     use_bias=False,
                                     padding='same',
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn2 = tf.layers.batch_normalization(conv2,
                                                scale=False,
                                                training=is_training,
                                                trainable=is_training)

            dropped = spatial_dropout('dropout', bn2,
                                      keep_prob=keep_prob,
                                      is_training=is_training)
        with tf.variable_scope('residual'):
            skip = tf.layers.conv2d(input,
                                    filters,
                                    [1,1],
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
        summed = skip + dropped
        output = tf.nn.relu(summed)

    return output


def build_model(input, is_training=False):
    conv1 = res_block('conv1', input, 100, [3,3], keep_prob=0.01, is_training=is_training)
    conv2 = res_block('conv2', conv1, 16, [3,3], keep_prob=0.1, is_training=is_training)
    conv3 = res_block('conv3', conv2, 3, [3,3], keep_prob=0.5, is_training=is_training)

    return conv3

def build_train_op(labels, predictions, learning_rate, global_step=None):
    loss = tf.losses.mean_squared_error(labels, predictions)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op, loss
