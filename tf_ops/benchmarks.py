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
