# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Xception model.

"Xception: Deep Learning with Depthwise Separable Convolutions"
Fran{\c{c}}ois Chollet
https://arxiv.org/abs/1610.02357

We implement the modified version by Jifeng Dai et al. for their COCO 2017
detection challenge submission, where the model is made deeper and has aligned
features for dense prediction tasks. See their slides for details:

"Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge
2017 Entry"
Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei and Jifeng Dai
ICCV 2017 COCO Challenge workshop
http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf

We made a few more changes on top of MSRA's modifications:
1. Fully convolutional: All the max-pooling layers are replaced with separable
  conv2d with stride = 2. This allows us to use atrous convolution to extract
  feature maps at any resolution.

2. We support adding ReLU and BatchNorm after depthwise convolution, motivated
  by the design of MobileNetv1.

"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision
Applications"
Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
Tobias Weyand, Marco Andreetto, Hartwig Adam
https://arxiv.org/abs/1704.04861
"""
import collections

import tensorflow as tf

from tf_ops.wrap_ops import sep_conv2d, conv2d, batch_norm2d, drop_out, regularizer

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope
add_to_collection = tf.add_to_collection

_DEFAULT_MULTI_GRID = [1, 1, 1]


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing an Xception block.

  Its parts are:
    scope: The scope of the block.
    unit_fn: The Xception unit function which takes as input a tensor and
      returns another tensor with the output of the Xception unit.
    args: A list of length equal to the number of units in the block. The list
      contains one dictionary for each unit in the block to serve as argument to
      unit_fn.
  """


@add_arg_scope
def xception_module(inputs,
                    depth_list,
                    skip_connection_type,
                    strides,
                    unit_rate_list=None,
                    rate=1,
                    activation_fn_in_separable_conv=False,
                    outputs_collections=None,
                    scope=None):
  """An Xception module.

  The output of one Xception module is equal to the sum of `residual` and
  `shortcut`, where `residual` is the feature computed by three separable
  convolution. The `shortcut` is the feature computed by 1x1 convolution with
  or without striding. In some cases, the `shortcut` path could be a simple
  identity function or none (i.e, no shortcut).

  Note that we replace the max pooling operations in the Xception module with
  another separable convolution with striding, since atrous rate is not properly
  supported in current TensorFlow max pooling implementation.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth_list: A list of three integers specifying the depth values of one
      Xception module.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    strides: The block unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    unit_rate_list: A list of three integers, determining the unit rate for
      each separable convolution in the xception module.
    rate: An integer, rate for atrous convolution.
    activation_fn_in_separable_conv: use func between depthwise and pointwise convolution
    outputs_collections: Collection to add the Xception unit output.
    scope: Optional variable_scope.

  Returns:
    The Xception module's output.

  Raises:
    ValueError: If depth_list and unit_rate_list do not contain three elements,
      or if stride != 1 for the third separable convolution operation in the
      residual path, or unsupported skip connection type.

  """
  if len(depth_list) != 3:
    raise ValueError('Expect three elements in depth_list.')
  if unit_rate_list:
    if len(unit_rate_list) != 3:
      raise ValueError('Expect three elements in unit_rate_list.')

  with tf.variable_scope(scope, 'xception_module', [inputs]):
    residual = inputs

    for i in range(3):
      if activation_fn_in_separable_conv is None:
        residual = tf.nn.relu(residual)
        activate_fn = None
      else:
        activate_fn = tf.nn.relu
      residual = sep_conv2d(inputs=residual,
                            outc=depth_list[i],
                            ksize=[3, 3],
                            depth_multiplier=1,
                            ratios=[rate * unit_rate_list[i], rate * unit_rate_list[i]],
                            activate_middle=activation_fn_in_separable_conv,
                            activate=activate_fn,
                            strides=strides if i == 2 else [1, 1],
                            name='separable_conv' + str(i + 1))
    if skip_connection_type == 'conv':
      shortcut = conv2d(inputs=inputs,
                        outc=depth_list[-1],
                        ksize=[1, 1],
                        strides=strides,
                        activate=None,
                        name='shortcut')
      outputs = residual + shortcut
    elif skip_connection_type == 'sum':
      outputs = residual + inputs
    elif skip_connection_type == 'none':
      outputs = residual
    else:
      raise ValueError('Unsupported skip connection type.')

    add_to_collection(outputs_collections, outputs)
    return outputs


@add_arg_scope
def stack_blocks_dense(net,
                       blocks,
                       output_stride=None,
                       outputs_collections=None):
  """Stacks Xception blocks and controls output feature density.

  First, this function creates scopes for the Xception in the form of
  'block_name/unit_1', 'block_name/unit_2', etc.

  Second, this function allows the user to explicitly control the output
  stride, which is the ratio of the input to output spatial resolution. This
  is useful for dense prediction tasks such as semantic segmentation or
  object detection.

  Control of the output feature density is implemented by atrous convolution.

  Args:
    net: A tensor of size [batch, height, width, channels].
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution, which needs to be equal to
      the product of unit strides from the start up to some level of Xception.
      For example, if the Xception employs units with strides 1, 2, 1, 3, 4, 1,
      then valid values for the output_stride are 1, 2, 6, 24 or None (which
      is equivalent to output_stride=24).
    outputs_collections: Collection to add the Xception block outputs.

  Returns:
    net: Output tensor with stride equal to the specified output_stride.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  # The current_stride variable keeps track of the effective stride of the
  # activations. This allows us to invoke atrous convolution whenever applying
  # the next residual unit would result in the activations having stride larger
  # than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]):
      for i, unit in enumerate(block.args):
        if output_stride is not None and current_stride > output_stride:
          raise ValueError('The target output_stride cannot be reached.')
        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          # If we have reached the target output_stride, then we need to employ
          # atrous convolution with stride=1 and multiply the atrous rate by the
          # current unit's stride for use in subsequent layers.
          if output_stride is not None and current_stride == output_stride:
            net = block.unit_fn(net, rate=rate, **dict(unit, strides=[1, 1]))
            # print('Unit {} : Preforming {} Dilated Conv'.format(tf.get_variable_scope().name, rate))
            # make rate increases as stride,
            # so that conv kernel still matches exactly the same feature position
            rate *= unit.get('strides', [1, 1])[0]
          else:
            net = block.unit_fn(net, rate=1, **unit)
            current_stride *= unit.get('strides', [1, 1])[0]

      # Collect activations at the block's end before performing subsampling.
      add_to_collection(outputs_collections, net)
  if output_stride is not None and current_stride != output_stride:
    raise ValueError('The target output_stride cannot be reached.')

  return net


def xception(inputs,
             blocks,
             num_classes=None,
             is_training=True,
             global_pool=True,
             keep_prob=0.5,
             output_stride=None,
             reuse=tf.AUTO_REUSE,
             scope=None):
  """Generator for Xception models.

  This function generates a family of Xception models. See the xception_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce Xception of various depths.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels]. Must be
      floating point. If a pretrained checkpoint is used, pixel values should be
      the same as during training (see go/slim-classification-models for
      specifics).
    blocks: A list of length equal to the number of Xception blocks. Each
      element is an Xception Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks.
      If 0 or None, we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    keep_prob: Keep probability used in the pre-logits dropout layer.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is 0 or None,
      then net is the output of the last Xception block, potentially after
      global average pooling. If num_classes is a non-zero integer, net contains
      the pre-softmax activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.

  Raises:
    ValueError: If the target output_stride is not valid.
  """
  with tf.variable_scope(
    scope, 'xception', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + 'end_points'
    with arg_scope([conv2d, sep_conv2d,
                    xception_module,
                    stack_blocks_dense],
                   outputs_collections=end_points_collection):
      with arg_scope([batch_norm2d], is_training=is_training):
        net = inputs
        if output_stride is not None:
          if output_stride % 2 != 0:
            raise ValueError('The output_stride needs to be a multiple of 2.')
          # divide it by 2 for the entry_flow/conv1_1 convolution with stride = 2
          output_stride /= 2
        # Root block function operated on inputs.
        net = conv2d(net, 32, ksize=[3, 3], strides=[2, 2], name='entry_flow/conv1_1')
        net = conv2d(net, 64, ksize=[3, 3], strides=[1, 1], name='entry_flow/conv1_2')

        # Extract features for entry_flow, middle_flow, and exit_flow.
        net = stack_blocks_dense(net, blocks, output_stride)

        # Convert end_points_collection into a dictionary of end_points.
        end_points = tf.get_collection(end_points_collection)
        end_points = dict([(ep.name, ep) for ep in end_points])

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], name='global_pool', keepdims=True)
          end_points['global_pool'] = net
        if num_classes:
          net = drop_out(net, keep_prob=keep_prob, is_training=is_training,
                         name='prelogits_dropout')
          net = conv2d(net, num_classes, ksize=[1, 1], activation=None, batch_norm=False, name='logits')
          end_points[sc.name + '/logits'] = net
          end_points['predictions'] = tf.nn.softmax(net, axis=-1, name='predictions')
        return net, end_points


def xception_block(scope,
                   depth_list,
                   skip_connection_type,
                   activation_fn_in_separable_conv,
                   num_units,
                   strides,
                   unit_rate_list=None):
  """Helper function for creating a Xception block.

  Args:
    scope: The scope of the block.
    depth_list: The depth of the bottleneck layer for each unit.
    skip_connection_type: Skip connection type for the residual path. Only
      supports 'conv', 'sum', or 'none'.
    activation_fn_in_separable_conv: Includes activation function in the
      separable convolution or not.
    num_units: The number of units in the block.
    strides: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
    unit_rate_list: A list of three integers, determining the unit rate in the
      corresponding xception block.

  Returns:
    An Xception block.
  """
  if unit_rate_list is None:
    unit_rate_list = _DEFAULT_MULTI_GRID
  return Block(scope, xception_module, [{
    'depth_list': depth_list,
    'skip_connection_type': skip_connection_type,
    'activation_fn_in_separable_conv': activation_fn_in_separable_conv,
    'strides': strides,
    'unit_rate_list': unit_rate_list,
  }] * num_units)


def xception_65(inputs,
                num_classes=None,
                is_training=True,
                global_pool=True,
                keep_prob=0.5,
                output_stride=None,
                multi_grid=None,
                reuse=None,
                scope='xception_65'):
  """Xception-65 model."""
  blocks = [
    xception_block('entry_flow/block1',
                   depth_list=[128, 128, 128],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('entry_flow/block2',
                   depth_list=[256, 256, 256],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('entry_flow/block3',
                   depth_list=[728, 728, 728],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('middle_flow/block1',
                   depth_list=[728, 728, 728],
                   skip_connection_type='sum',
                   activation_fn_in_separable_conv=None,
                   num_units=16,
                   strides=[1, 1]),
    xception_block('exit_flow/block1',
                   depth_list=[728, 1024, 1024],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('exit_flow/block2',
                   depth_list=[1536, 1536, 2048],
                   skip_connection_type='none',
                   activation_fn_in_separable_conv=tf.nn.relu,
                   num_units=1,
                   strides=[1, 1],
                   unit_rate_list=multi_grid),
    xception_block('exit_flow/block3',
                   depth_list=[1536, 1536, 2048],
                   skip_connection_type='none',
                   activation_fn_in_separable_conv=tf.nn.relu,
                   num_units=1,
                   strides=[1, 1],
                   unit_rate_list=multi_grid),

  ]
  return xception(inputs,
                  blocks=blocks,
                  num_classes=num_classes,
                  is_training=is_training,
                  global_pool=global_pool,
                  keep_prob=keep_prob,
                  output_stride=output_stride,
                  reuse=reuse,
                  scope=scope)


def xception_entry_flow(inputs,
                        output_stride=None,
                        end_points_collection=None):
  if end_points_collection is None:
    end_points_collection = tf.get_variable_scope().name + '/end_points'
  net = inputs
  if output_stride is not None:
    if output_stride % 2 != 0:
      raise ValueError('The output_stride needs to be a multiple of 2.')
    # divide it by 2 for the entry_flow/conv1_1 convolution with stride = 2
    output_stride /= 2
  # Root block function operated on inputs.
  net = conv2d(net, 32, ksize=[3, 3], strides=[2, 2], name='entry_flow/conv1_1')
  net = conv2d(net, 64, ksize=[3, 3], strides=[1, 1], name='entry_flow/conv1_2')

  # Extract features for entry_flow, middle_flow, and exit_flow.
  entry_blocks = [
    xception_block('entry_flow/block1',
                   depth_list=[128, 128, 128],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('entry_flow/block2',
                   depth_list=[256, 256, 256],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('entry_flow/block3',
                   depth_list=[728, 728, 728],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2])
  ]

  net = stack_blocks_dense(net, entry_blocks, output_stride)

  # Convert end_points_collection into a dictionary of end_points.
  end_points = tf.get_collection(end_points_collection)
  end_points = dict([(ep.name, ep) for ep in end_points])
  return net, end_points


def xception_mid_flow(inputs,
                      output_stride=None,
                      end_points_collection=None):
  if end_points_collection is None:
    end_points_collection = tf.get_variable_scope().name + '/end_points'
  net = inputs
  # Extract features for entry_flow, middle_flow, and exit_flow.
  mid_blocks = [
    xception_block('middle_flow/block1',
                   depth_list=[728, 728, 728],
                   skip_connection_type='sum',
                   activation_fn_in_separable_conv=None,
                   num_units=16,
                   strides=[1, 1])
  ]
  net = stack_blocks_dense(net, mid_blocks, output_stride)
  # Convert end_points_collection into a dictionary of end_points.
  end_points = tf.get_collection(end_points_collection)
  end_points = dict([(ep.name, ep) for ep in end_points])
  return net, end_points


def xception_exit_flow(inputs,
                       output_stride=None,
                       end_points_collection=None,
                       multi_grid=None):
  if end_points_collection is None:
    end_points_collection = tf.get_variable_scope().name + '/end_points'
  net = inputs
  # Extract features for entry_flow, middle_flow, and exit_flow.
  mid_blocks = [
    xception_block('exit_flow/block1',
                   depth_list=[728, 1024, 1024],
                   skip_connection_type='conv',
                   activation_fn_in_separable_conv=None,
                   num_units=1,
                   strides=[2, 2]),
    xception_block('exit_flow/block2',
                   depth_list=[1536, 1536, 2048],
                   skip_connection_type='none',
                   activation_fn_in_separable_conv=tf.nn.relu,
                   num_units=1,
                   strides=[1, 1],
                   unit_rate_list=multi_grid),
    xception_block('exit_flow/block3',
                   depth_list=[1536, 1536, 2048],
                   skip_connection_type='none',
                   activation_fn_in_separable_conv=tf.nn.relu,
                   num_units=1,
                   strides=[1, 1],
                   unit_rate_list=multi_grid)
  ]
  net = stack_blocks_dense(net, mid_blocks, output_stride)
  # Convert end_points_collection into a dictionary of end_points.
  end_points = tf.get_collection(end_points_collection)
  end_points = dict([(ep.name, ep) for ep in end_points])

  return net, end_points


def xception_65_detection(inputs,
                          num_classes=None,
                          is_training=True,
                          global_pool=True,
                          keep_prob=0.5,
                          output_stride=16,
                          multi_grid=None,
                          reuse=tf.AUTO_REUSE,
                          scope='xception_65'):
  with tf.variable_scope(
    scope, 'xception', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + 'end_points'
    with arg_scope([conv2d, sep_conv2d,
                    xception_module,
                    stack_blocks_dense],
                   outputs_collections=end_points_collection):
      with arg_scope([batch_norm2d], is_training=is_training):
        end_points_collection = sc.original_name_scope + 'end_points'
        entry_net, entry_end_points = xception_entry_flow(inputs,
                                                          output_stride=output_stride,
                                                          end_points_collection=end_points_collection)
        output_stride = output_stride // 16 if output_stride is not None else None

        mid_net, mid_end_points = xception_mid_flow(entry_net,
                                                    output_stride=output_stride,
                                                    end_points_collection=end_points_collection)

        exit_net, exit_end_points = xception_exit_flow(mid_net,
                                                       output_stride=output_stride,
                                                       end_points_collection=end_points_collection,
                                                       multi_grid=multi_grid)

        with tf.variable_scope('detection_branch'):
          det_exit_net, det_exit_end_points = xception_exit_flow(mid_net,
                                                                 output_stride=None,
                                                                 end_points_collection=end_points_collection,
                                                                 multi_grid=multi_grid)

        end_points = {**entry_end_points, **mid_end_points, **exit_end_points, **det_exit_end_points}
  return exit_net, end_points


def xception_arg_scope(weight_decay=0.00004,
                       batch_norm_decay=0.9997,
                       batch_norm_epsilon=0.001,
                       batch_norm_scale=True,
                       weights_initializer_stddev=0.09,
                       activation_fn=tf.nn.relu,
                       regularize_depthwise=False,
                       use_batch_norm=True):
  """Defines the default Xception arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
    weights_initializer_stddev: The standard deviation of the trunctated normal
      weight initializer.
    activation_fn: The activation function in Xception.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for the Xception models.
  """
  regular_func = regularizer(mode='l2', scale=weight_decay)
  depthwise_regularizer = regular_func if regularize_depthwise else None

  with arg_scope(
    [conv2d, sep_conv2d],
    weight_init=tf.truncated_normal_initializer(stddev=weights_initializer_stddev),
    activate=activation_fn,
    batch_norm=use_batch_norm):
    with arg_scope([batch_norm2d],
                   decay=batch_norm_decay, eps=batch_norm_epsilon, affine=batch_norm_scale):
      with arg_scope([conv2d],
                     weight_reg=regular_func):
        with arg_scope([sep_conv2d],
                       depthwise_weight_reg=depthwise_regularizer,
                       pointwise_weight_reg=regular_func) as arg_sc:
          return arg_sc


if __name__ == '__main__':
  with tf.device('/CPU:0'):
    with arg_scope(xception_arg_scope()):
      inputs = tf.placeholder(name='inputs', shape=[16, 513, 513, 3], dtype=tf.float32)
      net, end_points = xception_65_detection(inputs, global_pool=False, is_training=True, output_stride=16)
      # net, end_points = xception_65(inputs, global_pool=False, is_training=True, output_stride=16)

  for i in end_points.keys():
    print(end_points[i])
    # for i in tf.global_variables():
    #     print(i.name)

    # Validation
    # sess = tf.Session()
    # saver = tf.train.Saver()
    # saver.restore(sess, save_path="/home/yifeng/TF_Models/pretrain/SEGS/xception_voc_trainval/model.ckpt")
    # # saver.restore(sess, save_path="/home/yifeng/Models/pretrain/SEGS/xception_imagenet/model.ckpt")
    # #
    # # inputs_v = np.random.rand(16, 513, 513, 3)
    # # np.save('inputs_v.npy', inputs_v)
    # inputs_v = np.load('inputs_v.npy')
    # print(np.mean(inputs_v), np.var(inputs_v))
    # key = 'xception_65/exit_flow/block2/unit_1/xception_module/separable_conv3/pointwise_conv/Relu:0'
    # output = sess.run(end_points[key], feed_dict={inputs: inputs_v})
    # np.save('output_org.npy', output)
    # print(np.mean(output), np.var(output))
    # # print(output)
