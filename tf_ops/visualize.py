import tensorflow as tf

from datasets.pascal_voc_utils import pascal_voc_palette


def paint(predictions, palette=pascal_voc_palette):
    num_classes = palette.shape[0]
    paint = tf.one_hot(predictions, depth=num_classes, axis=-1, dtype=predictions.dtype)
    paint = tf.squeeze(tf.tensordot(paint,
                                    tf.cast(pascal_voc_palette, predictions.dtype),
                                    axes=[[-1], [0]]), axis=3)
    return paint
