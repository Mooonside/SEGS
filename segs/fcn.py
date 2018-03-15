"""
An implementation of Fully Convolution Network
By Yifeng Chen
"""
import tensorflow as tf

from backbones.vgg_16 import vgg_16, vgg_arg_scope
from tf_ops.wrap_ops import trans_conv2d, crop, conv2d, tensor_shape

arg_scope = tf.contrib.framework.arg_scope
add_arg_scope = tf.contrib.framework.add_arg_scope


@add_arg_scope
def fcn_upsample_1(small, big, ksize=[4, 4], strides=[2, 2], padding='SAME',
                   name=None, outputs_collections=None):
    # trans_conv small to big size
    with tf.variable_scope(name, 'fcn_upsample'):
        big_shape = tensor_shape(big)
        big_dim = big_shape[-1]
        trans_conv = trans_conv2d(small, outc=big_dim, ksize=ksize, output_shape=big_shape,
                                  strides=strides, padding=padding)
        summary = trans_conv + big
    tf.add_to_collection(outputs_collections, summary)
    return summary


@add_arg_scope
def fcn_upsample_2(small, big, ksize=[4, 4], strides=[2, 2], padding='SAME',
                   name=None, outputs_collections=None):
    # trans_conv small to big size
    with tf.variable_scope(name, 'fcn_upsample'):
        outc = tensor_shape(small)[-1]
        big = conv2d(big, outc, ksize=[1, 1], activate=None, name='score_conv')
        big_shape = [i.value for i in big.get_shape()]
        big_dim = big_shape[-1]
        big_crop = crop(small, big)
        summary = small + big_crop

        trans_conv = trans_conv2d(summary, outc=big_dim, ksize=ksize, output_shape=big_shape,
                                  strides=strides, padding=padding)
    tf.add_to_collection(outputs_collections, trans_conv)
    return trans_conv


def fcn_8(inputs, num_classes=21):
    image_shape = tensor_shape(inputs)

    with arg_scope(vgg_arg_scope()):
        fcn_32, end_points = vgg_16(inputs, num_classes=num_classes,
                                    spatial_squeeze=False, fc_conv_padding='SAME')
    with tf.name_scope('upscale') as ns:
        end_points_collection = ns + '_end_points'
        with arg_scope([conv2d], outputs_collections=end_points_collection):
            # conv7 deconv and add with pool4 [jump = 16]
            pool4 = end_points['vgg_16/pool4:0']
            fcn_16 = fcn_upsample_2(fcn_32, pool4, ksize=[4, 4], name='to_16')

            pool3 = end_points['vgg_16/pool3:0']
            fcn_8 = fcn_upsample_2(fcn_16, pool3, ksize=[4, 4], name='to_8')

            fcn_1 = trans_conv2d(fcn_8, outc=num_classes, ksize=[16, 16], strides=[8, 8],
                                 output_shape=image_shape[:-1] + [num_classes], name='to_1')

            print(tf.get_collection(end_points_collection))
            end_points.update(
                dict([(ep.name, ep) for ep in tf.get_collection(end_points_collection)]))
        end_points[ns + '_to_16'] = fcn_16
        end_points[ns + '_to_8'] = fcn_8
        end_points[ns + '_to_1'] = fcn_1

    return fcn_1, end_points


inputs = tf.placeholder(name='inputs', shape=[16, 224, 224, 3], dtype=tf.float32)
fcn_1, end_points = fcn_8(inputs)
for ep in end_points.keys():
    print(ep)
