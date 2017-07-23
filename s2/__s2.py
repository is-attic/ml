import tensorflow as tf
import numpy as np

a0 = tf.constant(
  np.arange(1, 25, dtype = np.float32), shape = [4, 2, 3])
a1 = tf.constant(
  np.arange(1, 7, dtype = np.float32), shape = [3, 2])
a2 = tf.constant(
  np.arange(1, 5, dtype = np.float32), shape = [2, 2])


a3 = tf.constant(np.arange(1, 3), np.float32)

c = tf.reshape(tf.matmul(tf.reshape(a0, [-1, 3]), a1), [-1, 2, 2])
print(c.shape)
d = c + a2
print(d.shape)

e = tf.matmul(a3, a1)
print(e.shape)
with tf.Session() as sess:
  p0, p1 = sess.run([c,d])
  print(p0)
  print(p1)