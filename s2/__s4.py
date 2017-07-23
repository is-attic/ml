import tensorflow as tf
import numpy as np

a0 = tf.constant(
  np.arange(1, 25, dtype = np.float32), shape = [4, 2, 3])
a1 = tf.constant(
  np.arange(1, 7, dtype = np.float32), shape = [3, 2])
a2 = tf.constant(
  np.arange(1, 5, dtype = np.float32), shape = [2, 2])


a3 = tf.constant(np.arange(1, 3), np.float32)
t0 = tf.arg_max(a0, 2)

with tf.Session() as sess:
  p0 = sess.run(t0)
  print(p0)
