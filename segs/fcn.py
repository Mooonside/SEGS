"""
An implementation of Fully Convolution Network
By Yifeng Chen
"""
import tensorflow as tf

from backbones.vgg_16 import vgg_16, vgg_arg_scope
from tf_ops.wrap_ops import trans_conv2d, conv2d, tensor_shape

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope


@add_arg_scope
def fcn_upsample(small, big, ksize=[4, 4], strides=[2, 2], padding='SAME',
                 name=None, outputs_collections=None):
    """
    the upsample block for fcn, the specific strategy is :
        1. [1,1] conv to reduce big's channels so that channels match
        2. trans_conv to recover small's resolution so that resolution match
    :param small: low resolution feature
    :param big: high resolution feature
    :param ksize: trans_conv kernel size
    :param strides: trans_conv kernel stride
    :param padding: trans_conv kernel padding mode
    :param name: name for this op
    :param outputs_collections: add this op's output to outputs_collections
    :return:
    """
    # trans_conv small to big size
    with tf.variable_scope(name, 'fcn_upsample'):
        outc = tensor_shape(small)[-1]
        big = conv2d(big, outc, ksize=[1, 1], activate=None, name='score_conv')
        big_shape = tensor_shape(big)
        big_dim = big_shape[-1]
        trans_conv = trans_conv2d(small, outc=big_dim, ksize=ksize, output_shape=big_shape,
                                  strides=strides, padding=padding)
        summary = trans_conv + big
    tf.add_to_collection(outputs_collections, summary)
    return summary


def fcn_8(inputs, num_classes=21):
    image_shape = tensor_shape(inputs)

    with arg_scope(vgg_arg_scope()):
        fcn32, end_points = vgg_16(inputs, num_classes=num_classes,
                                   spatial_squeeze=False, fc_conv_padding='SAME')
    with tf.name_scope('upscale') as ns:
        end_points_collection = ns + '_end_points'
        with arg_scope([conv2d], outputs_collections=end_points_collection):
            # conv7 deconv and add with pool4 [jump = 16]
            pool4 = end_points['vgg_16/pool4:0']
            fcn16 = fcn_upsample(fcn32, pool4, ksize=[4, 4], name='to_16')

            pool3 = end_points['vgg_16/pool3:0']
            fcn8 = fcn_upsample(fcn16, pool3, ksize=[4, 4], name='to_8')

            fcn1 = trans_conv2d(fcn8, outc=num_classes, ksize=[16, 16], strides=[8, 8],
                                output_shape=image_shape[:-1] + [num_classes], name='to_1')

            print(tf.get_collection(end_points_collection))
            end_points.update(
                dict([(ep.name, ep) for ep in tf.get_collection(end_points_collection)]))
        end_points[ns + '_to_32'] = fcn32
        end_points[ns + '_to_16'] = fcn16
        end_points[ns + '_to_8'] = fcn8
        end_points[ns + '_to_1'] = fcn1

    return fcn1, end_points


def fcn_16(inputs, num_classes=21):
    image_shape = tensor_shape(inputs)

    with arg_scope(vgg_arg_scope()):
        fcn32, end_points = vgg_16(inputs, num_classes=num_classes,
                                   spatial_squeeze=False, fc_conv_padding='SAME')
    with tf.name_scope('upscale') as ns:
        end_points_collection = ns + '_end_points'
        with arg_scope([conv2d], outputs_collections=end_points_collection):
            # conv7 deconv and add with pool4 [jump = 16]
            pool4 = end_points['vgg_16/pool4:0']
            fcn16 = fcn_upsample(fcn32, pool4, ksize=[4, 4], name='to_16')

            fcn1 = trans_conv2d(fcn16, outc=num_classes, ksize=[32, 32], strides=[16, 16],
                                output_shape=image_shape[:-1] + [num_classes], name='to_1')

            print(tf.get_collection(end_points_collection))
            end_points.update(
                dict([(ep.name, ep) for ep in tf.get_collection(end_points_collection)]))
        end_points[ns + '_to_32'] = fcn32
        end_points[ns + '_to_16'] = fcn16
        end_points[ns + '_to_1'] = fcn1

    return fcn1, end_points


def fcn_32(inputs, num_classes=21):
    image_shape = tensor_shape(inputs)

    with arg_scope(vgg_arg_scope()):
        fcn32, end_points = vgg_16(inputs, num_classes=num_classes,
                                   spatial_squeeze=False, fc_conv_padding='SAME')
    with tf.name_scope('upscale') as ns:
        end_points_collection = ns + '_end_points'
        with arg_scope([conv2d], outputs_collections=end_points_collection):
            # conv7 deconv and add with pool4 [jump = 16]
            fcn1 = trans_conv2d(fcn32, outc=num_classes, ksize=[64, 64], strides=[32, 32],
                                output_shape=image_shape[:-1] + [num_classes], name='to_1')

            print(tf.get_collection(end_points_collection))
            end_points.update(
                dict([(ep.name, ep) for ep in tf.get_collection(end_points_collection)]))
        end_points[ns + '_to_1'] = fcn1
    return fcn1, end_points


def fcn(inputs, num_classes=21, mode='8'):
    assert mode in ['8', '16', '32']
    if mode == '8':
        return fcn_8(inputs, num_classes)
    if mode == '16':
        return fcn_16(inputs, num_classes)
    if mode == '32':
        return fcn_32(inputs, num_classes)


def test():
    """
    merely for testing, IGNORE IT
    :return:
    """
    inputs = tf.placeholder(name='inputs', shape=[16, 224, 224, 3], dtype=tf.float32)
    fcn_1, end_points = fcn(inputs)
    for ep in end_points.keys():
        print(ep, end_points[ep].shape)


if __name__ == '__main__':
    test()
