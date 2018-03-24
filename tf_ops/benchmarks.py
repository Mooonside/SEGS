import tensorflow as tf

from tf_ops.wrap_ops import tensor_shape


def mAP(tensor1, tensor2):
    shape1 = tensor_shape(tensor1)
    shape2 = tensor_shape(tensor2)
    type1 = tensor1.dtype
    type2 = tensor2.dtype

    assert shape1 == shape2
    assert type1 == type2

    equal = tf.cast(tf.equal(tensor1, tensor2), tf.float32)
    acc = tf.reduce_mean(equal, name='mAP')
    return acc


def mIOU(predictions, labels, ignore_label=[0], num_classes=21, max_classes=255 + 1):
    """
    return mean_iou, which eliminates ignore label, and ious which keep ignored label
    :param predictions:
    :param labels:
    :param ignore_label:
    :param num_classes:
    :return:
    """
    ignore_label = [0 if i in ignore_label else 1 for i in range(num_classes)]

    confusion_mat = tf.confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                        labels=tf.reshape(labels, [-1]),
                                        num_classes=max_classes)
    # ignore those invalid labels
    confusion_mat = confusion_mat[:num_classes, :num_classes]
    diag = tf.diag_part(confusion_mat)
    row_sum = tf.reduce_sum(confusion_mat, axis=0)
    col_sum = tf.reduce_sum(confusion_mat, axis=1)

    denominator = row_sum + col_sum - diag

    selector = tf.cast(
        tf.equal(denominator, 0), tf.int32)
    denominator += selector
    ious = diag / denominator

    ignore_ious = ious * ignore_label
    selector = 1 - selector
    selector *= ignore_label

    miou = tf.reduce_sum(ignore_ious) / tf.cast(tf.reduce_sum(selector), tf.float64)
    return miou, ious
