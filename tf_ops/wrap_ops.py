"""
Wrapping Functions for Common Use
Written by Yifeng-Chen
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

var_scope = tf.variable_scope
arg_scope = tf.contrib.framework.arg_scope
add_to_collection = tf.add_to_collection
add_arg_scope = tf.contrib.framework.add_arg_scope


@add_arg_scope
def tensor_shape(tensor):
    return [i.value for i in tensor.get_shape()]


@add_arg_scope
def get_variable(name, shape, dtype=tf.float32, device='0', init=None, reg=None, collections=None):
    if device == 'CPU' or 'cpu':
        with tf.device('/cpu:0'):
            var = tf.get_variable(name=name, shape=shape, dtype=dtype,
                                  initializer=init, regularizer=reg, collections=collections)
    elif device in ['0', '1', '2', '3']:
        with tf.device('/gpu:' + device):
            var = tf.get_variable(name=name, shape=shape, dtype=dtype,
                                  initializer=init, regularizer=reg, collections=collections)
    else:
        raise Exception('Invalid Device Specified')
    return var


@add_arg_scope
def conv2d(inputs, outc, ksize, strides=[1, 1], ratios=[1, 1], name=None, padding='SAME',
           activate=tf.nn.relu, batch_norm=True, init=None, reg=None, outputs_collections=None):
    """
    Wrapper for Conv layers
    :param inputs: [N, H, W, C]
    :param outc: output channels
    :param ksize: [hk, wk]
    :param strides: [hs, ws]
    :param ratios: [hr, wr]
    :param name: var_scope & operation name
    :param padding: padding mode
    :param activate: activate function
    :param batch_norm: whether performs batch norm
    :param init: initializer for filters
    :param reg: regularization for filters
    :param outputs_collections: add result to some collection
    :return: convolution after activation
    """
    indim = tensor_shape(inputs)[-1]

    with tf.variable_scope(name, 'conv'):
        filters = get_variable(name='weights', shape=ksize + [indim, outc],
                               init=init, reg=reg)
        if not batch_norm:
            biases = get_variable(name='biases', shape=[outc], init=tf.zeros_initializer)

    conv = tf.nn.conv2d(input=inputs,
                        filter=filters,
                        strides=[1] + strides + [1],
                        padding=padding,
                        use_cudnn_on_gpu=True,
                        data_format="NHWC",
                        dilations=[1] + ratios + [1],
                        name=name)

    tf.add_to_collection(outputs_collections, conv)

    if batch_norm:
        conv = batch_norm2d(conv)
    else:
        conv = conv + biases

    if activate is not None:
        conv = activate(conv)

    return conv


@add_arg_scope
def fully_connected(inputs, outc, name='None',
                    activate=tf.nn.relu, init=None, reg=None, outputs_collections=None):
    """
    Wrapper for FC layers
    :param inputs: [N, H, W, C]
    :param outc: output channels
    :param name: var_scope & operation name
    :param activate: activate function
    :param init: initializer for filters
    :param reg: regularization for filters
    :param outputs_collections: add result to some collection
    :return:
    """
    indim = tensor_shape(inputs)[-1]
    with tf.variable_scope(name, 'fully_connected'):
        weights = get_variable(name='weights', shape=[indim, outc],
                               init=init, reg=reg)
        biases = get_variable(name='biases', shape=[outc],
                              init=tf.zeros_initializer)

    dense = tf.tensordot(inputs, weights, axes=[[-1], [0]]) + biases
    tf.add_to_collection(outputs_collections, dense)

    if activate is not None:
        dense = activate(dense)

    return dense


@add_arg_scope
def max_pool2d(inputs, ksize=[2, 2], strides=[2, 2], padding='SAME', name=None, outputs_collections=None):
    """
    Wrapper for tf.nn.max_pool
    :param inputs: [N, H, W, C]
    :param ksize: [hk, wk]
    :param strides: [hs, ws]
    :param padding: padding mode
    :param name: var_scope & operation name
    :param outputs_collections: add result to some collection
    :return:
    """
    pool = tf.nn.max_pool(value=inputs,
                          ksize=[1] + ksize + [1],
                          strides=[1] + strides + [1],
                          padding=padding,
                          data_format='NHWC',
                          name=name)
    tf.add_to_collection(outputs_collections, pool)
    return pool


@add_arg_scope
def drop_out(inputs, kp_prob, is_training, name=None):
    if type(kp_prob) != float:
        print('Invalid Parameter Specified {}'.format(kp_prob))
    if is_training:
        return tf.nn.dropout(inputs, keep_prob=kp_prob, name=name)
    else:
        return inputs


@add_arg_scope
def batch_norm2d(inputs, is_training=True, eps=1e-05, decay=0.9, affine=True, name=None):
    """
    Do channel-wise batch normalization
    :param inputs: [N, H, W, C]
    :param is_training: bool var indicating mode
    :param eps: for stabilize
    :param decay: momentum factor
    :param affine: whether scale & offset
    :param name: var_scope & operation name
    :return: batch_norm output
    """
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tensor_shape(inputs)[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            # update moving_moments
            axes = list(np.arange(len(inputs.get_shape()) - 1))
            mean, variance = tf.nn.moments(inputs, axes, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                # https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
                return tf.identity(mean), tf.identity(variance)

        mean, variance = tf.cond(tf.constant(is_training), mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, eps)
        else:
            outputs = tf.nn.batch_normalization(inputs, mean, variance, None, None, eps)
        return outputs


@add_arg_scope
def l2_regularizer(scale, scope=None):
    """
    a simple wrapper for l2 norm
    :param scale: punishment weight
    :param scope: operation name
    :return: a regularization function
    """
    return tf.contrib.layers.l2_regularizer(
        scale,
        scope=scope
    )


@add_arg_scope
def trans_conv2d(inputs, outc, ksize, output_shape, strides=[1, 1], padding='SAME',
                 init=None, reg=None, name=None):
    with tf.variable_scope(name, 'trans_conv'):
        indim = tensor_shape(inputs)[-1]
        filters = get_variable(name='weights', shape=ksize + [outc, indim],
                               init=init, reg=reg)

    trans_conv = tf.nn.conv2d_transpose(
        inputs,
        filters,
        output_shape,
        strides,
        padding=padding,
        name=name
    )

    return trans_conv


@add_arg_scope
def crop(small, big):
    """
    crop big centrally according to small 's shape
    :param small: [Ns, hs, ws, cs]
    :param big: [NB, HB, WB, CB]
    :return: big cropped to [NB, hs, ws, CB]
    """
    small_shape = tensor_shape(small)
    big_shape = tensor_shape(big)

    assert small_shape[0] == big_shape[0]
    start_h = (big_shape[1] - small_shape[1]) // 2
    start_w = (big_shape[2] - small_shape[2]) // 2
    start = [0, start_h, start_w, 0]
    size = [big_shape[0], small_shape[1], small_shape[2], big_shape[3]]

    crop = tf.slice(big, start, size)
    return crop
