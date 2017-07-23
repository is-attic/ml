import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

from .common import *

C = config_update('flow.yaml')

'''
Reference:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
'''
def BiRNN(x, n_timestep, n_hidden, weight, bias):
  x = tf.unstack(x, n_timestep, 1)
  lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
  lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)

  outputs, _, _ = rnn.static_bidirectional_rnn(
    lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
  # weight: tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
  # bias: tf.Variable(tf.random_normal([n_classes]))

  return tf.matmul(outputs, weight) + bias

