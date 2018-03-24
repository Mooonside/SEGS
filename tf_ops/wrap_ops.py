"""
Wrapping Functions for Common Use
Written by Yifeng-Chen
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

LOSS_COLLECTIONS = tf.GraphKeys.LOSSES
TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

var_scope = tf.variable_scope
arg_scope = tf.contrib.framework.arg_scope
add_to_collection = tf.add_to_collection
add_arg_scope = tf.contrib.framework.add_arg_scope

weight_collections = 'weights_collections'
bias_collections = 'bias_collections'
batch_norm_collections = 'batch_norm_collections'

WEIGHT_COLLECTIONS = [weight_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BIAS_COLLECTIONS = [bias_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BN_COLLECTIONS = [batch_norm_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]


@add_arg_scope
def tensor_shape(tensor):
    return [i.value for i in tensor.get_shape()]


@add_arg_scope
def get_variable(name, shape, dtype=tf.float32, device='/CPU:0', init=None, reg=None, collections=None):
    with tf.device(device):
        var = tf.get_variable(name=name, shape=shape, dtype=dtype,
                              initializer=init, regularizer=reg, collections=collections)
    return var


@add_arg_scope
def conv2d(inputs, outc, ksize, strides=[1, 1], ratios=[1, 1], name=None, padding='SAME',
           activate=tf.nn.relu, batch_norm=True,
           weight_init=None, weight_reg=None, bias_init=tf.zeros_initializer, bias_reg=None,
           outputs_collections=None):
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
    :param weight_init: weight initializer
    :param weight_reg: weight regularizer
    :param bias_init: bias initializer
    :param bias_reg: bias regularizer
    :param outputs_collections: add result to some collection
    :return: convolution after activation
    """
    indim = tensor_shape(inputs)[-1]

    with tf.variable_scope(name, 'conv'):
        filters = get_variable(name='weights', shape=ksize + [indim, outc],
                               init=weight_init, reg=weight_reg, collections=WEIGHT_COLLECTIONS)
        if not batch_norm:
            biases = get_variable(name='biases', shape=[outc], init=bias_init, reg=bias_reg,
                                  collections=BIAS_COLLECTIONS)

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
def fully_connected(inputs, outc, name='None', activate=tf.nn.relu,
                    weight_init=None, weight_reg=None, bias_init=tf.zeros_initializer, bias_reg=None,
                    outputs_collections=None):
    """
    Wrapper for FC layers
    :param inputs: [N, H, W, C]
    :param outc: output channels
    :param name: var_scope & operation name
    :param activate: activate function
    :param weight_init: weight initializer
    :param weight_reg: weight regularizer
    :param bias_init: bias initializer
    :param bias_reg: bias regularizer
    :param outputs_collections: add result to some collection
    :return:
    """
    indim = tensor_shape(inputs)[-1]
    with tf.variable_scope(name, 'fully_connected'):
        weights = get_variable(name='weights', shape=[indim, outc],
                               init=weight_init, reg=weight_reg, collections=WEIGHT_COLLECTIONS)
        biases = get_variable(name='biases', shape=[outc],
                              init=bias_init, reg=bias_reg, collections=BIAS_COLLECTIONS)

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
    :param inputs: print(shape1, shape2)
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
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer,
                                   collections=BN_COLLECTIONS)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer,
                                    collections=BN_COLLECTIONS)
            outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, eps)
        else:
            outputs = tf.nn.batch_normalization(inputs, mean, variance, None, None, eps)
        return outputs


@add_arg_scope
def regularizer(mode, scale, scope=None):
    if mode is None or scale is None:
        return None
    if mode.lower() == 'l2':
        return tf.contrib.layers.l2_regularizer(scale=scale, scope=scope)
    if mode.lower() == 'l1':
        return tf.contrib.layers.l1_regularizer(scale=scale, scope=scope)


@add_arg_scope
def trans_conv2d(inputs, outc, ksize, output_shape, strides=[1, 1], padding='SAME',
                 init=None, reg=None, name=None):
    """
    Deconvolution result
    :param inputs: print(shape1, shape2)
    :param outc: output channels
    :param ksize: [kh, kw]
    :param output_shape: a tensor shape [N,H,W,C] , N can be None
    :param strides: [sh, sw]
    :param padding:
    :param init: init for weight
    :param reg: reg for weight
    :param name: operation name
    :return: deconv result
    """
    with tf.variable_scope(name, 'trans_conv'):
        indim = tensor_shape(inputs)[-1]
        filters = get_variable(name='weights', shape=ksize + [outc, indim],
                               init=init, reg=reg, collections=WEIGHT_COLLECTIONS)

    trans_conv = tf.nn.conv2d_transpose(
        inputs,
        filters,
        output_shape,
        strides=[1] + strides + [1],
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


@add_arg_scope
def arg_max(tensors, axis, out_type=tf.int32, keep_dim=True, name=None):
    if keep_dim:
        return tf.expand_dims(tf.argmax(tensors, axis=axis, output_type=out_type), axis=-1, name=name)
    else:
        return tf.argmax(tensors, axis=tensors, name=name, output_type=out_type)


@add_arg_scope
def softmax_with_logits(predictions, labels, ignore_labels=[255]):
    """
    a loss vector [N*H*W, ]
    :param predictions: [N, H, W, c], raw outputs of model
    :param labels: [N ,H, W, 1] int32
    :param ignore_labels: ignore pixels with ground truth in ignore_labels
    :return: a sample_mean loss
    """
    dim = tensor_shape(predictions)[-1]
    logits = tf.reshape(predictions, shape=[-1, dim])
    labels = tf.reshape(labels, [-1])
    # which is all ones
    mask = tf.cast(tf.not_equal(labels, -1), tf.float32)
    for ignore in ignore_labels:
        mask *= tf.cast(tf.not_equal(labels, ignore), tf.float32)

    labels = tf.one_hot(labels, depth=dim)
    labels = tf.stop_gradient(labels)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=labels, name='sample_wise_loss')
    loss *= mask

    loss = tf.divide(tf.reduce_sum(loss), tf.reduce_sum(mask), name='mean_loss')
    tf.add_to_collection(LOSS_COLLECTIONS, loss)
    return loss


def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


@add_arg_scope
def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          data_format='NHWC',
          trainable=True,
          scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXNet, helping the port of models to
    TensorFlow. Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
      inputs: 4D input Tensor;
      pad: 2-Tuple with padding values for H and W dimensions;
      mode: Padding mode. C.f. `tf.pad`
      data_format:  NHWC or NCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        # Padding shape.
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net


@add_arg_scope
def channel_to_last(inputs,
                    data_format='NHWC',
                    scope=None):
    """Move the channel axis to the last dimension. Allows to
    provide a single output format whatever the input data format.

    Args:
      inputs: Input Tensor;
      data_format: NHWC or NCHW.
    Return:
      Input in NHWC format.
    """
    with tf.name_scope(scope, 'channel_to_last', [inputs]):
        if data_format == 'NHWC':
            net = inputs
        elif data_format == 'NCHW':
            net = tf.transpose(inputs, perm=(0, 2, 3, 1))
        return net
