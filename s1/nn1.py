import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def char_set(base, r):
  b = ord(base)
  return ''.join([chr(b + i) for i in range(r)])


###
DIGIT_SET = char_set('0', 10)
LOWER_CHAR_SET = char_set('a', 26)
UPPER_CHAR_SET = char_set('A', 26)

###
#CHAR_IN = DIGIT_SET + LOWER_CHAR_SET + UPPER_CHAR_SET
#CHAR_OUT = DIGIT_SET + UPPER_CHAR_SET + LOWER_CHAR_SET
CHAR_IN = LOWER_CHAR_SET + UPPER_CHAR_SET + DIGIT_SET
#CHAR_OUT = DIGIT_SET + UPPER_CHAR_SET + LOWER_CHAR_SET
CHAR_OUT = CHAR_IN

x = np.empty([len(CHAR_IN), len(CHAR_IN)])
y = np.empty([len(CHAR_IN), len(CHAR_IN)])


###
def char_to_vector(s, c):
  r = np.zeros(len(s))
  r[s.index(c)] = 1
  return r


for i in range(len(CHAR_IN)):
  x[i] = char_to_vector(CHAR_IN, CHAR_IN[i])
  y[i] = char_to_vector(CHAR_IN, CHAR_OUT[i])

# print(y)
# print(x)

INPUT_SIZE = len(CHAR_IN)
OUTPUT_SIZE = len(CHAR_OUT)
HIDDEN_SIZE = 24

###
tf_x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
tf_y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

l1 = tf.layers.dense(tf_x, HIDDEN_SIZE, tf.nn.relu6)
output = tf.layers.dense(l1, OUTPUT_SIZE)

# print(tf_y)
# print(output)

# accuracy = tf.metrics.accuracy(
#   labels=tf.squeeze(tf_y),
#   predictions=tf.argmax(output, axis=1))[1]

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(tf_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# loss = tf.reduce_mean(
#   tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=output))

loss = tf.reduce_mean(tf.pow(tf_y * 10 - output, 2))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init_op = tf.group(
  tf.global_variables_initializer(),
  tf.local_variables_initializer())
sess.run(init_op)

for _ in range(200):
  for __ in range(1000):
    sess.run(train_op, feed_dict = {
      tf_x: x, tf_y:y})

  l, a = sess.run([loss, accuracy], feed_dict={tf_x: x, tf_y: y})

  print("%d %f %f" % (_, a, l))


def map_char(sess, cin):
  v = char_to_vector(CHAR_IN, cin)
  h = np.argmax(sess.run(output, feed_dict={tf_x:[v]}))
  return CHAR_IN[h]

def map_char_to_vector(sess, cin):
  v = char_to_vector(CHAR_IN, cin)
  v.shape = [len(v), 1]
  return sess.run(output, feed_dict = {tf_x: np.transpose(v)})

for i in range(len(CHAR_IN)):
  if (map_char(sess, CHAR_IN[i])) != CHAR_OUT[i]:
    print(i, CHAR_IN[i], map_char(sess, CHAR_IN[i]))
  else:
    print("%c -> %c OK" % (CHAR_IN[i], CHAR_OUT[i]))

print("OK")
#print (sess.run(output, feed_dict = {
#  tf_x: x[1:2,:]}))

#print (sess.run(output, feed_dict = {
#  tf_x: x[3:4,:]}))
print(map_char_to_vector(sess, 'a'))