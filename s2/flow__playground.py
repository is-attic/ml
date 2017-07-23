import tensorflow as tf

from .common import *
from .flow__birnn import BiRNN

classes = [OUTPUT_DROP, OUTPUT_KEEP]

n_timestep = 32
n_input = 2
n_classes = len(classes)

x = tf.placeholder(tf.float32, [None, n_timestep, n_input])
y = tf.placeholder(tf.float32, [None, n_timestep, n_classes])

