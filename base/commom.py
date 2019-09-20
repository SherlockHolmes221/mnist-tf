import tensorflow as tf


def conv_relu(x, input_channels,filter_height, filter_width, num_filters, stride_y, stride_x, name, trainable, padding='SAME', groups=1):
    if trainable is not None:
        train = True
    else:
        train = tf.Session().run(trainable)

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', trainable=train, shape=[filter_height,
                                                filter_width,
                                                input_channels/groups,
                                                num_filters])
        biases = tf.get_variable('biases', shape=[num_filters], trainable=train)

    if groups == 1:
        conv = convolve(x, weights)
    else:
        # Split input and weights and convolve them separately
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups,value=weights)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]
        # Concat the convolved output together again
        conv = tf.concat(axis=3, values=output_groups)
    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)
    return relu


def fc(x,  input_num, output_num, name, trainable, relu=True):
    if trainable is not None:
        train = True
    else:
        train = tf.Session().run(trainable)

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weight', shape=[input_num, output_num], trainable=train)
        biases = tf.get_variable('biases', [output_num], trainable=train)
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
    if relu:
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def max_pooling(x , filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool2d(x, ksize=[1, filter_height, filter_width, 1],
                            strides=[1, stride_y, stride_x, 1],
                            padding=padding, name=name)


def drop_out(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)
