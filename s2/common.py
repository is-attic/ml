from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import string
import yaml
import json
import random

import numpy as np

##
def config_update(*params):
  conf = {}
  for p in params:
    if type(p) == dict:
      conf.update(p)
    elif type(p) == str:
      with open(p, 'r') as fi:
        conf.update(yaml.safe_load(fi))
  return conf

class ConfigHolder(object):
  def __init__(self, dict_):
    self.__dict__.update(dict_)

def config_freeze(config):
  return json.loads(json.dumps(config), object_hook=ConfigHolder)


# ---
LAB_END =0
LAB_LETTER = 2
LAB_DIGIT = 3
LAB_DASH = 4
LAB_SYM = 5
LAB_SPACE = 6
LAB_PAD = 7
LAB_UNK = 8

OUTPUT_KEEP = 0
OUTPUT_DROP = 1

## ----
def char_to_vector(ch):
  o = ord(ch)
  if ch in string.ascii_letters:
    return (LAB_LETTER, o)

  if ch in string.digits:
    return (LAB_DIGIT, o)

  if ch == '-':
    return (LAB_DASH, o)

  if ch in "@_.|()":
    return (LAB_SYM, o)

  return (LAB_UNK, o)


def chars(n):
  return ''.join([string.ascii_uppercase[random.randint(0, 25)] for x in range(n)])


def digits(n):
  return ''.join([string.digits[random.randint(0,9)] for x in range(n)])


def G_ccc_ddd():
  return chars(3) + "-" + digits(3), 'KKKDKKK'


def seq_to_vector(s):
  res = [char_to_vector(ch) for ch in s]
  res.append((LAB_END, 0))
  return res

def padding_seq(seq, length):
  while(len(seq) != length):
    seq.append([LAB_UNK, 0])
  return seq

def input_seq_to_array(time_step, instr):
  seq = seq_to_vector(instr)
  if (len(seq) > time_step):
    seq = seq[:time_step]
  elif (len(seq) < time_step):
    seq = padding_seq(seq, time_step)
  return np.array(seq)


def output_label_to_array(time_step, instr):
  res = np.zeros([time_step, 2], dtype = np.float32)

  if len(instr) > time_step:
    instr = instr[:time_step]
  else:
    instr += 'D' * (time_step - len(instr))

  for i in range(len(instr)):
    if instr[i] == 'K':
      res[i, 0] = 1
    else:
      res[i, 1] = 1
  return res


def train_data_numpy_array(batch_size, timestep, datas):
  x = np.zeros([batch_size, timestep, 2], dtype = np.float32)
  y = np.zeros([batch_size, timestep, 2], dtype = np.float32)

  bs = min(batch_size, len(datas))
  for i in range(bs):
    x[i,:,:] = input_seq_to_array(timestep, datas[i][0])
    y[i,:,:] = output_label_to_array(timestep, datas[i][1])

  return (x, y)


def numpy_array_to_label(a, origin = None):
  L = 'KD'
  labels = ''.join([L[x] for x in np.argmax(a, 1)])

  if origin != None:
    return labels[:len(origin)]

  return labels