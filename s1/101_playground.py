import tensorflow as tf
import numpy as np

import os;os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

v0 = tf.Variable([[1, 2, 3], [4,5,6]], dtype=tf.float32)
v1 = tf.reduce_mean(v0, 0)
print(v0)
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(v1))