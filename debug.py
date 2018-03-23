import tensorflow as tf

label = tf.Variable(
    [
        [1, 1, 1, 2],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
    ]
)

pred = tf.Variable(
    [
        [1, 1, 1, 1],
        [1, 0, 2, 1],
        [1, 0, 0, 1],
        [2, 1, 1, 1],
    ]
)

_, confusion_mat = tf.metrics.mean_iou(predictions=pred, labels=label, num_classes=3)
# with tf.control_dependencies(update_op):
#     hahaha = tf.identity(mean_iou)
#
#
diag = tf.diag_part(confusion_mat)
row_sum = tf.reduce_sum(confusion_mat, axis=0)
col_sum = tf.reduce_sum(confusion_mat, axis=1)

denominator = row_sum + col_sum - diag

selector = tf.cast(
    tf.equal(denominator, 0), tf.float64)
denominator += selector
ious = diag / denominator
miou = tf.reduce_sum(ious) / tf.reduce_sum(1 - selector)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
#
# sess.run([update_op])
# # miou = sess.run(mean_iou)
print(sess.run(miou) * 3)
