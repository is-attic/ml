from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.contrib import rnn

from common import *

def init_config():
  conf = config_update('flow.yaml')
  return config_freeze(conf)

print('loading config ...')
C = init_config()

HIDDEN_SIZE = C.hidden_size
INPUT_SIZE = C.input_size
OUTPUT_SIZE = C.output_size
TIMESTEP = C.timestep
LEARNING_RATE = C.learning_rate
BATCH_SIZE = C.batch_size

display_step = 200
random.seed(0x1234)


def BiRNN(x, n_timestep, n_hidden, weight, bias):
  '''
  Reference:
  https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
  '''
  n_output = bias.shape[0].value

  x = tf.unstack(x, n_timestep, 1)
  lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
  lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

  outputs, _, _ = rnn.static_bidirectional_rnn(
    lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
  # weight: tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
  # bias: tf.Variable(tf.random_normal([n_classes]))

  oa = tf.reshape(tf.stack(outputs, 1), [-1, n_hidden * 2])
  ob = tf.matmul(oa, weight)
  oc = tf.reshape(ob, [-1, n_timestep, n_output])
  return oc + bias


def train_data_source(batch_size):
  return [ G_ccc_ddd() for x in range(batch_size)]

## __variables
W = {}
B = {}
N = {}

X = tf.placeholder(tf.float32, [None, TIMESTEP, INPUT_SIZE])
Y = tf.placeholder(tf.float32, [None, TIMESTEP, OUTPUT_SIZE])

W['out'] = tf.Variable(tf.random_normal([2 * HIDDEN_SIZE, OUTPUT_SIZE]))
B['out'] = tf.Variable(tf.random_normal([OUTPUT_SIZE]))

pred = BiRNN(X, TIMESTEP, HIDDEN_SIZE, W['out'], B['out'])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 2), tf.argmax(Y, 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

best_loss = 100000
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

for step in range(20000):
  _x, _y = train_data_numpy_array(
    batch_size = BATCH_SIZE,
    timestep= TIMESTEP,
    datas = train_data_source(BATCH_SIZE))

  feed_dict = {
    X: _x,
    Y: _y
  }

  sess.run(optimizer, feed_dict = feed_dict)

  if step % display_step == 0:
    loss, accu = sess.run([cost, accuracy], feed_dict = feed_dict)
    print("%d %e %.2f%%" % (step * BATCH_SIZE, loss, accu * 100))
    if loss < best_loss:
      saver_path = saver.save(sess, 'model/model_flow_0_full_ce.ckpt')
      best_loss = loss
      #print("save model to" + saver_path + "for " + str(loss))
      print("____")
