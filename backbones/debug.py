import tensorflow as tf

with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
    a = tf.get_variable('a', shape=[3,3], dtype=tf.float32)

with tf.variable_scope('scope', reuse=tf.AUTO_REUSE) as sc:
    b = tf.get_variable('a', shape=[3,3], dtype=tf.float32)

print(b.name)