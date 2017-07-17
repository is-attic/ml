import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

x1 = tf.placeholder(tf.float32, None)
y1 = tf.placeholder(tf.float32, None)
z1 = tf.add(x1, y1)

x2 = tf.placeholder(tf.float32, [2, 1])
y2 = tf.placeholder(tf.float32, [1, 2])
z2 = tf.matmul(x2, y2)

with tf.Session() as sess:
  z1_value = sess.run(z1,
    feed_dict = {x1: 1, y1: 1})
  print(z1_value)
  z1_value, z2_value = sess.run([z1, z2],
    feed_dict = {
      x1: 1, y1: 1,
      x2: [[2], [2]], y2:[[3, 3]]
    })
  print(z1_value)
  print(z2_value)

a = [[1,1,1], [2,2,2], [3,3,3]]
b = [1,2,3]
c = np.transpose(b)
d = np.matmul(a, c)
print(d)