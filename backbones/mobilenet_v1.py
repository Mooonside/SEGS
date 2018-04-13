'''
Implement of mobilenet_v1
By jiabao
'''

# Tensorflow mandates these.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

from tf_ops.wrap_ops import *
from tf_utils import partial_restore

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

arg_scope = tf.contrib.framework.arg_scope

# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]

def mobilenet_v1_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False,
                           batch_norm_decay=0.9997,
                           batch_norm_epsilon=0.001):
    """Defines the default MobilenetV1 arg scope.
  Args:
    is_training: Whether or not we're training the model.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
  Returns:
    An `arg_scope` to use for the mobilenet v1 model.
    """
    batch_norm_params = {
          'is_training': is_training,
          'affine': True,
          'decay': batch_norm_decay,
          'eps': batch_norm_epsilon,
          'name': 'BatchNorm',
    }

# Set weight_decay for weights in Conv and DepthSepConv layers.separable_convolution2d
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer

    else:
        depthwise_regularizer = None
    with arg_scope([conv2d],
                        weight_init=weights_init,
                        activate=tf.nn.relu6, batch_norm = True):
        with arg_scope([separable_conv2d],
                            weight_init=weights_init,
                            activate=tf.nn.relu6, batch_norm = True):#normalizer_fn = batch_norm2d):
            with arg_scope([batch_norm2d], **batch_norm_params):
                with arg_scope([conv2d], weight_reg=regularizer):
                    with arg_scope([separable_conv2d],
                                  weight_reg=depthwise_regularizer) as sc:
                        return sc

def _fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.
  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.
  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
    rate: An integer, rate for atrous convolution.
  Returns:
    output: A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                    [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs


def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      use_explicit_padding=False,
                      scope=None):
    """Mobilenet v1.
  Constructs a Mobilenet v1 network from inputs to the given final endpoint.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
      'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
      'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
      'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
      'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    output_stride: An integer that specifies the requested ratio of input to
      output spatial resolution. If not None, then we invoke atrous convolution
      if necessary to prevent the network from reducing the spatial resolution
      of the activation maps. Allowed values are 8 (accurate fully convolutional
      mode), 16 (fast fully convolutional mode), 32 (classification mode).
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    scope: Optional variable_scope.
  Returns:
    tensor_out: output tensor corresponding to the final_endpoint.
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0, or the target output_stride is not
                allowed.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with arg_scope([conv2d, separable_conv2d], padding=padding):
          # The current_stride variable keeps track of the output stride of the
          # activations, i.e., the running product of convolution strides up to the
          # current network layer. This allows us to invoke atrous convolution
          # whenever applying the next convolution would result in the activations
          # having output stride larger than the target output_stride.
          current_stride = 1

        # The atrous convolution rate parameter.
          rate = 1

          net = inputs
          for i, conv_def in enumerate(conv_defs):
              # print("i = "+str(i))
              end_point_base = 'Conv2d_%d' % i

              # print("1.conv_def.depth="+str(conv_def.depth))
              if output_stride is not None and current_stride == output_stride:
                  # If we have reached the target output_stride, then we need to employ
                  # atrous convolution with stride=1 and multiply the atrous rate by the
                  # current unit's stride for use in subsequent layers.
                  layer_stride = 1
                  layer_rate = rate
                  rate *= conv_def.stride
              else:
                  layer_stride = conv_def.stride
                  layer_rate = 1
                  current_stride *= conv_def.stride

              # print("2.conv_def.depth="+str(conv_def.depth))
              if isinstance(conv_def, Conv):
                  end_point = end_point_base
                  if use_explicit_padding:
                      net = _fixed_padding(net, conv_def.kernel)
                  net = conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                    strides=[conv_def.stride,conv_def.stride],
                                    name=end_point)
                  # print("conv:")
                  # print(net)
                  end_points[end_point] = net
                  if end_point == final_endpoint:
                      return net, end_points

              elif isinstance(conv_def, DepthSepConv):
                  end_point = end_point_base + '_depthwise'
                  # print(net)
                  # print(end_point)
                  # By passing filters=None
                  # separable_conv2d produces only a depthwise convolution layer
                  if use_explicit_padding:
                      net = _fixed_padding(net, conv_def.kernel, layer_rate)
                  net = separable_conv2d(net, None, conv_def.kernel,
                                              depth_multiplier=1,
                                              stride=layer_stride,
                                              rate=layer_rate,
                                              batch_norm = True,
                                              name=end_point)


                  # print("after separable_conv2d:")
                  # print("depthwise:")
                  # print(net)
                  # print(end_point)
                  end_points[end_point] = net
                  if end_point == final_endpoint:
                      return net, end_points

                  end_point = end_point_base + '_pointwise'

                  # print("conv_def.depth="+str(conv_def.depth))
                  # print(net)
                  # print(end_point)
                  net = conv2d(net, depth(conv_def.depth), [1, 1],
                                    strides=[1,1],
                                    batch_norm = True,
                                    name=end_point)

                  # print("pointwise:")
                  # print(net)
                  end_points[end_point] = net
                  if end_point == final_endpoint:
                      return net, end_points
              else:
                  raise ValueError('Unknown convolution type %s for layer %d'
                                 % (conv_def.ltype, i))
              # print(end_point)
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v1(inputs,
                 num_classes=1001,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 min_depth=8,
                 depth_multiplier=1.0,
                 conv_defs=None,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='MobilenetV1',
                 global_pool=False):
    """Mobilenet v1 model for classification.
  Args:
    inputs: a tensor of shape [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    conv_defs: A list of ConvDef namedtuples specifying the net architecture.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.
  Returns:
    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    end_points: a dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: Input rank is invalid.
    """
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' %
                         len(input_shape))

    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=tf.AUTO_REUSE) as scope:
        with arg_scope([drop_out],
                            is_training=is_training):
            net, end_points = mobilenet_v1_base(inputs, scope=scope,
                                              min_depth=min_depth,
                                              depth_multiplier=depth_multiplier,
                                              conv_defs=conv_defs)
            with tf.variable_scope('Logits'):
                if global_pool:
                  # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                  # Pooling with a fixed kernel size.
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = avg_pool2d(net, kernel_size, padding='VALID',
                                          name='AvgPool_1a')
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 1024
                net = drop_out(net, kp_prob=dropout_keep_prob, name='Dropout_1b')
                logits = conv2d(net, num_classes, [1, 1], activate=None,
                                     batch_norm = False, name='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return logits, end_points

mobilenet_v1.default_image_size = 224


def wrapped_partial(func, *args, **kwargs):
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func

mobilenet_v1_075 = wrapped_partial(mobilenet_v1, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet_v1, depth_multiplier=0.50)
mobilenet_v1_025 = wrapped_partial(mobilenet_v1, depth_multiplier=0.25)


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out

# if __name__ == '__main__':
#     # sess = tf.Session()
#     inputs = tf.placeholder(name='inputs', shape=[5, 224, 224, 3], dtype=tf.float32)
    # with slim.arg_scope(mobilenet_v1_arg_scope()):
    #     net, end_points = mobilenet_v1_base(inputs)
    # print(net)
    # trainable_vars_m = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # print(len(tf.global_variables()))
    # print(len(trainable_vars_m))
    # print(len(slim.get_model_variables()))

    # with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
    #                     normalizer_fn=slim.batch_norm):
    #     mobilenet_v1_base(inputs)
    #     total_params, _ = slim.model_analyzer.analyze_vars(slim.get_model_variables())
    #     # print(slim.get_model_variables())
    #     print(total_params)
    #     print(len(tf.global_variables()))
    #     print(len(tf.trainable_variables()))
    #     print(len(tf.model_variables()))

    # for i in (tf.global_variables()):
    #     print(i.name)
if __name__ == '__main__':
    sess = tf.Session()
    # inputs = tf.placeholder(name='inputs', shape=[16, 224, 224, 3], dtype=tf.float32)
    # inputs = tf.random_uniform((1, 224, 224, 3),dtype=tf.float32)
    inputs = tf.placeholder(shape=[1, 224, 224, 3], dtype=tf.float32, name='inputs')
    # in_array = sess.run(tf.Print(inputs,[inputs]))
    # np.save("input.npy",in_array)
    # np.savetxt("input.txt",in_array.reshape(224*224,3))
    with arg_scope(mobilenet_v1_arg_scope()):
        nets, end_points = mobilenet_v1(inputs)
    partial_restore_op = partial_restore(sess, tf.global_variables(),'/mnt/disk/jiabao/mobilenet_v1_1/mobilenet_v1_1.0_224.ckpt')
    sess.run(partial_restore_op)


    out_array = sess.run(nets, feed_dict={inputs: np.load('input.npy')})
    print(np.mean(out_array), np.var(out_array))
    np.savetxt("output_1.txt",out_array)

    print('='*8)
    # sess.run(partial_restore_op)
    # print('Recovering From Pretrained Model ')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path='/mnt/disk/jiabao/mobilenet_v1_1/mobilenet_v1_1.0_224.ckpt')
    # sess.run(nets)
    # print(nets)
    out_array = sess.run(nets, feed_dict={inputs: np.load('input.npy')})
    print(np.mean(out_array), np.var(out_array))
    np.savetxt("output_2.txt",out_array)

    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess=sess, save_path='/mnt/disk/jiabao/mobilenet_v1_1/mobilenet_v1_1.0_224.ckpt')
    # out_tensor = sess.run(nets)
    # print(out_tensor)
    # out_array = sess.run(tf.Print(nets,[nets],summarize=1001))
    # np.savetxt("output_3.txt",out_tensor)

    # trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # trainable_vars_m = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
    # print(trainable_vars_m)
    # print(trainable_vars[0], sess.run(trainable_vars[0]))
    # print(trainable_vars[0])
    # print(len(tf.global_variables()))
    # for (i,j) in zip(tf.trainable_variables(),range(len(tf.trainable_variables()))):
    #     print(i)
    #     print(trainable_vars[j])
    #     print(np.mean(sess.run(i)),np.mean(sess.run(trainable_vars[j])))
