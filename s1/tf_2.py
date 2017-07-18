import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DOMAIN_SIZE = 25


a = np.arange(DOMAIN_SIZE, dtype = np.float32)
a = a.reshape([DOMAIN_SIZE, 1])

b = np.zeros([DOMAIN_SIZE, DOMAIN_SIZE], dtype = np.float32)
b[:,:] = 0.001
for i in range(DOMAIN_SIZE):
  b[i, i] = 1


x = a
y = b


INPUT_SIZE = 1
OUTPUT_SIZE = DOMAIN_SIZE
HIDDEN_SIZE = 100

###
tf_x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
tf_y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

layer1 = tf.layers.dense(tf_x, HIDDEN_SIZE, tf.nn.sigmoid)
output = tf.layers.dense(layer1, OUTPUT_SIZE)


# accuracy = tf.metrics.accuracy(
#   labels=tf.squeeze(tf_y),
#   predictions=tf.argmax(output, axis=1))[1]

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(tf_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels=tf_y, logits=output))
#loss = tf.reduce_mean(tf.pow(tf_y * 10 - output, 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
#optimizer = tf.train.MomentumOptimizer(learning_rate = 0.1, momentum=0.5)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(
  tf.global_variables_initializer(),
  tf.local_variables_initializer())
sess.run(init_op)

for _ in range(10):
  for __ in range(500):
    sess.run(train_op, feed_dict = {
      tf_x: x, tf_y: y})

  a, l = sess.run([accuracy, loss], feed_dict={tf_x: x, tf_y: y})

  print("%d %f %f" % (_, a, l))

def map2(sess, value):
  _x = np.array([[value]])
  _y = sess.run(output, feed_dict = {tf_x: _x})
  return np.argmax(_y[0])

def map3(sess, value):
  _x = np.array([[value]])
  return sess.run(output, feed_dict = {tf_x: _x})

correct = 0
for i in range(DOMAIN_SIZE):
  _y = map2(sess, i)
  if i != _y:
    print("%d != %d" % (i, _y))
  else:
    correct += 1

print("%d/%d" % (correct, DOMAIN_SIZE))
print(map3(sess, 2))
print(map3(sess, 3))
