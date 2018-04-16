import tensorflow as tf

<<<<<<< HEAD
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


def confusion_matrix(predictions, labels, streaming=True, num_classes=21, max_classes=255 + 1,
                     updates_collections=tf.GraphKeys.UPDATE_OPS):
    if streaming:
        confusion_mat, update_op = _streaming_confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                               labels=tf.reshape(labels, [-1]),
                                                               num_classes=max_classes)
        confusion_mat = tf.cast(confusion_mat, tf.int32)
        tf.add_to_collection(updates_collections, tf.identity(update_op))
    else:
        confusion_mat = tf.confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                            labels=tf.reshape(labels, [-1]),
                                            num_classes=max_classes)
    # ignore those invalid labels
    confusion_mat = confusion_mat[:num_classes, :num_classes]
    return confusion_mat


def validation_metrics(predictions, labels, ignore_label=None, num_classes=21, max_classes=255 + 1,
                       updates_collections=tf.GraphKeys.UPDATE_OPS):
    confusion_mat = confusion_matrix(predictions, labels,
                                     streaming=True,
                                     num_classes=num_classes,
                                     # for 255 background labels
                                     max_classes=max_classes,
                                     updates_collections=updates_collections)
    tf.add_to_collection('shenjin', confusion_mat)
    metrics = {}
    metrics['mAP'] = mAP(confusion_mat=confusion_mat)
    metrics['mIOU'] = mIOU(confusion_mat=confusion_mat, ignore_label=ignore_label, num_classes=num_classes)
    metrics['mPRC'] = mPRC(confusion_mat=confusion_mat, ignore_label=ignore_label, num_classes=num_classes)
    metrics['mREC'] = mREC(confusion_mat=confusion_mat, ignore_label=ignore_label, num_classes=num_classes)

    return metrics


def mAP(predictions=None, labels=None, confusion_mat=None, num_classes=21, max_classes=255 + 1,
        streaming=True, updates_collections=tf.GraphKeys.UPDATE_OPS):
    """
    feed in either predictions and labels or confusion matrix,
    then ignore labels>=num_classes
    return mAP, and class_wise ap
    :param updates_collections:
    :param streaming:
    :param confusion_mat:
    :param predictions:
    :param labels:
    :param ignore_label:
    :param num_classes:
    :return: mean_iou and ious
    """
    assert not (confusion_mat is None and (predictions is None or labels is None))
    if confusion_mat is None:
        if streaming:
            confusion_mat, update_op = _streaming_confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                                   labels=tf.reshape(labels, [-1]),
                                                                   num_classes=max_classes)
            confusion_mat = tf.cast(confusion_mat, tf.int32)
            tf.add_to_collection(updates_collections, update_op)
        else:
            confusion_mat = tf.confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                labels=tf.reshape(labels, [-1]),
                                                num_classes=max_classes)
        confusion_mat = confusion_mat[:num_classes, :num_classes]

    diag = tf.reduce_sum(tf.diag_part(confusion_mat))
    denominator = tf.reduce_sum(confusion_mat)
    acc = diag / denominator
    return acc


def mPRC(predictions=None, labels=None, confusion_mat=None, ignore_label=None, num_classes=21, max_classes=255 + 1,
         streaming=True, updates_collections=tf.GraphKeys.UPDATE_OPS):
    assert not (confusion_mat is None and (predictions is None or labels is None))
    ignore_label = [0 if i in ignore_label else 1 for i in range(num_classes)]

    if confusion_mat is None:
        if streaming:
            confusion_mat, update_op = _streaming_confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                                   labels=tf.reshape(labels, [-1]),
                                                                   num_classes=max_classes)
            confusion_mat = tf.cast(confusion_mat, tf.int32)
            tf.add_to_collection(updates_collections, update_op)
        else:
            confusion_mat = tf.confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                labels=tf.reshape(labels, [-1]),
                                                num_classes=max_classes)
        confusion_mat = confusion_mat[:num_classes, :num_classes]

    # TP
    diag = tf.diag_part(confusion_mat)

    # TP + FP
    col_sum = tf.reduce_sum(confusion_mat, axis=1)
    # check where denominator == 0
    denominator = col_sum
    selector = tf.cast(
        tf.equal(denominator, 0), tf.int32)
    denominator += selector
    precision = diag / denominator

    ignore_precision = precision * ignore_label
    selector = 1 - selector
    selector *= ignore_label

    mean = tf.reduce_sum(ignore_precision) / tf.cast(tf.reduce_sum(selector), tf.float64)
    return mean, precision


def mREC(predictions=None, labels=None, confusion_mat=None, ignore_label=None, num_classes=21, max_classes=255 + 1,
         streaming=True, updates_collections=tf.GraphKeys.UPDATE_OPS):
    assert not (confusion_mat is None and (predictions is None or labels is None))
    ignore_label = [0 if i in ignore_label else 1 for i in range(num_classes)]

    if confusion_mat is None:
        if streaming:
            confusion_mat, update_op = _streaming_confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                                   labels=tf.reshape(labels, [-1]),
                                                                   num_classes=max_classes)
            confusion_mat = tf.cast(confusion_mat, tf.int32)
            tf.add_to_collection(updates_collections, update_op)
        else:
            confusion_mat = tf.confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                labels=tf.reshape(labels, [-1]),
                                                num_classes=max_classes)
        confusion_mat = confusion_mat[:num_classes, :num_classes]

    # TP
    diag = tf.diag_part(confusion_mat)
    # TP + FN
    row_sum = tf.reduce_sum(confusion_mat, axis=0)
    # check where denominator == 0
    denominator = row_sum
    selector = tf.cast(
        tf.equal(denominator, 0), tf.int32)
    denominator += selector
    recall = diag / denominator

    ignore_recall = recall * ignore_label
    selector = 1 - selector
    selector *= ignore_label

    mean = tf.reduce_sum(ignore_recall) / tf.cast(tf.reduce_sum(selector), tf.float64)
    return mean, ignore_recall


def mIOU(predictions=None, labels=None, confusion_mat=None, ignore_label=None, num_classes=21, max_classes=255 + 1,
         streaming=True, updates_collections=tf.GraphKeys.UPDATE_OPS):
    """
    feed in either predictions and labels or confusion matrix,
    then ignore labels>=num_classes or in igonre_label arguments
    return mean_iou, and class_wise iou

    :param updates_collections:
    :param streaming:
    :param confusion_mat:
=======
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
>>>>>>> c7a1431cf14c39f7216eebd64388f13fb13bada2
    :param predictions:
    :param labels:
    :param ignore_label:
    :param num_classes:
<<<<<<< HEAD
    :return: mean_iou and ious
    """
    assert not (confusion_mat is None and (predictions is None or labels is None))
    ignore_label = [0 if i in ignore_label else 1 for i in range(num_classes)]

    if confusion_mat is None:
        if streaming:
            confusion_mat, update_op = _streaming_confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                                   labels=tf.reshape(labels, [-1]),
                                                                   num_classes=max_classes)
            confusion_mat = tf.cast(confusion_mat, tf.int32)
            tf.add_to_collection(updates_collections, update_op)
        else:
            confusion_mat = tf.confusion_matrix(predictions=tf.reshape(predictions, [-1]),
                                                labels=tf.reshape(labels, [-1]),
                                                num_classes=max_classes)
        confusion_mat = confusion_mat[:num_classes, :num_classes]

    # TP
    diag = tf.diag_part(confusion_mat)
    # TP + FN
    row_sum = tf.reduce_sum(confusion_mat, axis=0)
    # TP + FP
    col_sum = tf.reduce_sum(confusion_mat, axis=1)

    denominator = row_sum + col_sum - diag
=======
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

>>>>>>> c7a1431cf14c39f7216eebd64388f13fb13bada2
    selector = tf.cast(
        tf.equal(denominator, 0), tf.int32)
    denominator += selector
    ious = diag / denominator

    ignore_ious = ious * ignore_label
    selector = 1 - selector
    selector *= ignore_label

    miou = tf.reduce_sum(ignore_ious) / tf.cast(tf.reduce_sum(selector), tf.float64)
    return miou, ious
