"""
Wrapping Functions for Common Use
Written by Yifeng-Chen
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average

TRAINABLE_VARIABLES = tf.GraphKeys.TRAINABLE_VARIABLES
GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES

var_scope = tf.variable_scope
arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope

weight_collections = 'weights_collections'
bias_collections = 'bias_collections'
batch_norm_collections = 'batch_norm_collections'

WEIGHT_COLLECTIONS = [weight_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BIAS_COLLECTIONS = [bias_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
BN_COLLECTIONS = [batch_norm_collections, TRAINABLE_VARIABLES, GLOBAL_VARIABLES]
LOSS_COLLECTIONS = tf.GraphKeys.LOSSES


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
def same_padding(inputs, ksize, ratios):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      ksize: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      ratios: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_array = [[0, 0]]
    for idx, k in enumerate(ksize):
        k_effective = k + (k - 1) * (ratios[idx] - 1)
        pad_total = k_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        pad_array.append([pad_beg, pad_end])
    pad_array.append([0, 0])

    padded_inputs = tf.pad(inputs, pad_array)
    return padded_inputs


@add_arg_scope
def conv2d(inputs, outc, ksize, strides=[1, 1], ratios=[1, 1], name=None, padding='SAME', activate=tf.nn.relu,
           batch_norm=True, use_bias=None, weight_init=None, weight_reg=None, bias_init=tf.zeros_initializer,
           bias_reg=None, outputs_collections=None):
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
    :param use_bias: whether use bias addition
    :param weight_init: weight initializer
    :param weight_reg: weight regularizer
    :param bias_init: bias initializer
    :param bias_reg: bias regularizer
    :param outputs_collections: add result to some collection
    :return: convolution after activation
    """
    # can't use both
    if use_bias is None:
        use_bias = not batch_norm
    assert not (batch_norm and use_bias)
    indim = tensor_shape(inputs)[-1]

    with tf.variable_scope(name, 'conv'):
        filters = get_variable(name='weights', shape=ksize + [indim, outc],
                               init=weight_init, reg=weight_reg, collections=WEIGHT_COLLECTIONS)

        if padding == 'SAME':
            inputs = same_padding(inputs, ksize, ratios)

        conv = tf.nn.conv2d(input=inputs,
                            filter=filters,
                            strides=[1] + strides + [1],
                            padding='VALID',
                            use_cudnn_on_gpu=True,
                            data_format="NHWC",
                            dilations=[1] + ratios + [1],
                            name=name)

        # tf.add_to_collection(outputs_collections, conv)
        if batch_norm:
            conv = batch_norm2d(conv)
        elif use_bias:
            biases = get_variable(name='biases', shape=[outc], init=bias_init, reg=bias_reg,
                                  collections=BIAS_COLLECTIONS)
            conv = conv + biases

        if activate is not None:
            conv = activate(conv)

    tf.add_to_collection(outputs_collections, conv)
    return conv


@add_arg_scope
def sep_conv2d(inputs, outc, ksize, strides=[1, 1], ratios=[1, 1], depth_multiplier=1, padding='SAME',
               activate=tf.nn.relu, batch_norm=True, use_bias=False,
               weight_init=None, depthwise_weight_reg=None, pointwise_weight_reg=None,
               bias_init=tf.zeros_initializer, bias_reg=None,
               activate_middle=None, outputs_collections=None, name='separate_conv'):
    """
    separable convolution warped
    :param inputs: [H, H, W, C]
    :param outc: output channels
    :param ksize: [hk, wk]
    :param depth_multiplier: num of kernels per channel in depth_wise convolution
    :param strides: [hs, ws]
    :param ratios: [hr, wr]
    :param padding: padding: padding mode
    :param activate: activate function
    :param activate_middle: whether apply activation between depth_wise and pointwise conv
    :param batch_norm: whether performs batch norm
    :param use_bias: whether apply bias addition
    :param weight_init: weight initializer
    :param pointwise_weight_reg: weight regularizer for pointwise weight
    :param depthwise_weight_reg: weight regularizer for depthwise weight
    :param bias_init: bias initializer
    :param bias_reg: bias regularizer
    :param outputs_collections: add result to some collection
    :param name: var_scope & operation name
    :return:
    """
    with tf.variable_scope(name, 'separate_conv'):
        with tf.variable_scope('depthwise_conv'):
            indim = tensor_shape(inputs)[-1]
            depthwise_filter = get_variable(name='depthwise_weights', shape=ksize + [indim, depth_multiplier],
                                            init=weight_init, reg=depthwise_weight_reg, collections=WEIGHT_COLLECTIONS)

            if padding == 'SAME':
                inputs = same_padding(inputs, ksize, ratios)

            conv = tf.nn.depthwise_conv2d(
                input=inputs,
                filter=depthwise_filter,
                strides=[1] + strides + [1],
                padding='VALID',
                rate=ratios,
                name='depthwise_conv',
                data_format="NHWC"
            )

            if batch_norm:
                conv = batch_norm2d(conv)
            elif use_bias:
                biases = get_variable(name='biases', shape=[outc], init=bias_init, reg=bias_reg,
                                      collections=BIAS_COLLECTIONS)
                conv = conv + biases

            if activate_middle is not None:
                conv = activate_middle(conv)
            add_to_collection(outputs_collections, conv)

        with tf.variable_scope('pointwise_conv'):
            pointwise_filter = get_variable(name='pointwise_weights', shape=[1, 1] + [indim * depth_multiplier, outc],
                                            init=weight_init, reg=pointwise_weight_reg, collections=WEIGHT_COLLECTIONS)

            conv = tf.nn.conv2d(input=conv,
                                filter=pointwise_filter,
                                strides=[1] + [1, 1] + [1],
                                padding='VALID',
                                use_cudnn_on_gpu=True,
                                data_format="NHWC",
                                dilations=[1] + [1, 1] + [1],
                                name='pointwise_conv')

            if batch_norm:
                conv = batch_norm2d(conv)
            elif use_bias:
                biases = get_variable(name='biases', shape=[outc], init=bias_init, reg=bias_reg,
                                      collections=BIAS_COLLECTIONS)
                conv = conv + biases

            if activate is not None:
                conv = activate(conv)
            add_to_collection(outputs_collections, conv)

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

    if activate is not None:
        dense = activate(dense)
    tf.add_to_collection(outputs_collections, dense)

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
def avg_pool2d(inputs, ksize=[2, 2], strides=[2, 2], padding='SAME', name=None, outputs_collections=None):
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
    pool = tf.nn.avg_pool(value=inputs,
                          ksize=[1] + ksize + [1],
                          strides=[1] + strides + [1],
                          padding=padding,
                          data_format='NHWC',
                          name=name)
    tf.add_to_collection(outputs_collections, pool)
    return pool


@add_arg_scope
def drop_out(inputs, kp_prob, is_training=True, name=None):
    if type(kp_prob) != float:
        print('Invalid Parameter Specified {}'.format(kp_prob))
    if is_training:
        return tf.nn.dropout(inputs, keep_prob=kp_prob, name=name)
    else:
        return inputs


@add_arg_scope
def batch_norm2d(inputs, is_training=True, eps=1e-05, decay=0.9, affine=True, force_update=False, name=None):
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
        moving_mean = tf.get_variable('moving_mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        # mean_var_with_update is deprecated !
        # tf.nn.moments is computing the sample variance,
        # whereas tf.nn.fused_batch_norm is computing the unbiased variance estimator.
        # The difference between the two is a factor n/n-1
        # def mean_var_with_update():
        #     # update moving_moments
        #     axes = list(np.arange(len(inputs.get_shape()) - 1))
        #     mean, variance = tf.nn.moments(inputs, axes, name='moments')
        #     with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay, zero_debias=False),
        #                                   assign_moving_average(moving_variance, variance, decay, zero_debias=False)]):
        #         # https://stackoverflow.com/questions/34877523/in-tensorflow-what-is-tf-identity-used-for
        #         return tf.identity(mean), tf.identity(variance)

        if affine:
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer,
                                   collections=BN_COLLECTIONS)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer,
                                    collections=BN_COLLECTIONS)
        else:
            gamma = tf.constant(value=np.ones(params_shape, dtype=np.float32))
            beta = tf.constant(value=np.zeros(params_shape, dtype=np.float32))

        def training_mode():
            outputs, batch_mean, batch_var = tf.nn.fused_batch_norm(inputs, gamma, beta, epsilon=eps)
            return outputs, batch_mean, batch_var

        def inference_mode():
            outputs, batch_mean, batch_var = tf.nn.fused_batch_norm(inputs, gamma, beta, moving_mean, moving_variance,
                                                                    epsilon=eps, is_training=False)
            return outputs, batch_mean, batch_var

        outputs, batch_mean, batch_var = tf.cond(tf.constant(is_training), training_mode, inference_mode)
        update_ops = [assign_moving_average(moving_mean, batch_mean, decay, zero_debias=False),
                      assign_moving_average(moving_variance, batch_var, decay, zero_debias=False)]
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_ops)

        return outputs


@add_arg_scope
def l2_norm_1D(inputs, norm_dim=-1, eps=1e-12, scale=True, scale_initializer=tf.ones_initializer, scope=None):
    with tf.variable_scope(scope, 'L2_Norm1D', [inputs]) as sc:
        # output = x / sqrt(max(sum(x**2), epsilon))
        outputs = tf.nn.l2_normalize(inputs, norm_dim, eps)
        if scale:
            gamma = get_variable(name='gamma', shape=tensor_shape(inputs)[-1:], dtype=tf.float32,
                                 init=scale_initializer)
            outputs = outputs * gamma
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
def softmax_with_logits(predictions, labels,
                        ignore_labels=[255],
                        loss_collections=LOSS_COLLECTIONS,
                        weights=None):
    """
    a loss vector [N*H*W, ]
    :param predictions: [N, H, W, c], raw outputs of model
    :param labels: [N ,H, W, 1] int32
    :param ignore_labels: ignore pixels with ground truth in ignore_labels
    :param loss_collections: add to which loss collections
    :param weights: set weight to each loss
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

    if weights is not None:
        loss *= weights
        mask *= tf.cast(tf.not_equal(weights, 0), tf.float32)

    loss = tf.where(
        tf.reduce_sum(mask) < 1e-7,
        tf.constant(0.0, dtype=tf.float32),
        tf.divide(tf.reduce_sum(loss), tf.reduce_sum(mask), name='mean_loss')
    )
    if loss_collections is not None:
        tf.add_to_collection(loss_collections, loss)
    return loss


def ms_softmax_with_logits(scales_to_logits,
                           labels,
                           ignore_label,
                           upsample_logits=True,
                           scope=None):
    """Adds softmax cross entropy loss for logits of each scale.

    Args:
      scales_to_logits: A map from logits names for different scales to logits.
        The logits have shape [batch, logits_height, logits_width, num_classes].
      labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
      num_classes: Integer, number of target classes.
      ignore_label: Integer, label to ignore.
      loss_weight: Float, loss weight.
      upsample_logits: Boolean, upsample logits or not.
      scope: String, the scope for the loss.

    Raises:
      ValueError: Label or logits is None.
    """
    total_loss = 0
    for scale, logits in scales_to_logits.items():
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits, tf.shape(labels)[1:3], align_corners=True)
            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels, tf.shape(logits)[1:3], align_corners=True)

        with tf.name_scope(loss_scope):
            loss = softmax_with_logits(logits, scaled_labels,
                                       ignore_labels=ignore_label,
                                       loss_collections=LOSS_COLLECTIONS,
                                       weights=None)
            total_loss += loss

    return total_loss


def smooth_l1(x):
    square_selector = tf.cast(tf.less(tf.abs(x), 1), tf.float32)
    x = square_selector * 0.5 * tf.square(x) + (1 - square_selector) * (tf.abs(x) - 0.5)
    return x


def add_to_collection(collections, variable, name=None):
    if name is None:
        tf.add_to_collection(collections, variable)
    else:
        tf.add_to_collection(collections, tf.identity(variable, name=name))
