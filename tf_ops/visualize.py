import tensorflow as tf

from datasets.pascal_voc_utils import pascal_voc_palette
from tf_ops.wrap_ops import tensor_shape


def paint(predictions, palette=pascal_voc_palette):
    num_classes = palette.shape[0]
    paint = tf.one_hot(predictions, depth=num_classes, axis=-1, dtype=predictions.dtype)
    paint = tf.squeeze(tf.tensordot(paint,
                                    tf.cast(pascal_voc_palette, predictions.dtype),
                                    axes=[[-1], [0]]), axis=3)
    return paint


def compare(predictions, labels):
    same = tf.cast(tf.equal(predictions, labels), tf.int32)
    paint = tf.one_hot(same, depth=2, axis=-1, dtype=predictions.dtype)
    paint = tf.squeeze(tf.tensordot(paint,
                                    tf.cast([[255, 0, 0], [0, 0, 0]], predictions.dtype),
                                    axes=[[-1], [0]]), axis=3)
    return paint



def locate_boundary(labels):
    """ locate boundaries in labels
    todo: test this function
    :param labels: [N, H, W, C]
    :return: a bool tensor, true indicating boundaries
    """
    H, W = tensor_shape(labels)[1:3]
    pad = tf.pad(labels, [[0, 0], [0, 1], [0, 0], [0, 0]], mode='REFLECT')[:, 1:, :, :]
    boundary = tf.equal(pad, labels)
    pad = tf.pad(labels, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='REFLECT')[:, :, 1:, :]
    boundary = tf.logical_or(boundary, tf.equal(pad, labels))

    expansions = tf.cast(
        tf.pad(labels, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'),
        tf.bool
    )
    for xmove in [-1, 0, 1]:
        for ymove in [-1, 0, 1]:
            boundary = tf.logical_or(boundary, expansions[:, 1 + xmove:1 + xmove + H, 1 + ymove:1 + ymove + W, :])
    return boundary
