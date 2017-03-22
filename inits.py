from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np


def weight_variable(shape, init='kaiming', name='weights'):
  with tf.name_scope(name):
    if init == 'lecun':
      return _lecun_variable(shape, name=name)
    elif init == 'xavier':
      return _xavier_variable(shape, name=name)
    elif init == 'kaiming':
      return _kaiming_variable(shape, name=name)
    elif init == 'bilinear':
      return _bilinear_variable(shape, name=name)
    else:
      raise ValueError('Unrecognized variable init %s' % init)


def bias_variable(shape, value=0.0, dtype=tf.float32, name='biases'):
  with tf.name_scope(name):
    initial = tf.constant(value, dtype=dtype, shape=shape)
    biases = tf.Variable(initial, name=name)
    tf.add_to_collection(name, biases)
    return biases


def _get_fans(shape):
  if len(shape) == 2:  # fully connected layer
    fan_in = shape[0]
    fan_out = shape[1]
  elif len(shape) == 4:  # convolutional layer
    receptive_field_size = np.prod(shape[:2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  else:
    raise ValueError('Unrecognized shape %s' % str(shape))
  return fan_in, fan_out


def _bilinear_variable(shape, name='bilinear_variable'):
  height, width, depth, _ = shape
  f = np.ceil(width / 2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros((height, width))
  for y in range(height):
    for x in range(width):
      value = (1 - np.abs(x / f - c)) * (1 - np.abs(y / f - c))
      bilinear[y, x] = value

  weights = np.zeros(shape)
  for i in range(depth):
    weights[:, :, i, i] = bilinear

  #initial = tf.constant(weights, dtype=tf.float32, name='bilinear_init')
  return tf.Variable(weights, dtype=tf.float32, name=name)

def _lecun_variable(shape, name='lecun_variable'):
  fan_in, fan_out = _get_fans(shape)
  scale = np.sqrt(3. / fan_in)
  initial = tf.random_uniform(shape,
                              minval=-scale,
                              maxval=scale,
                              name='lecun_init')
  return tf.Variable(initial, dtype=tf.float32, name=name)


def _xavier_variable(shape, name='xavier_variable'):
  fan_in, fan_out = _get_fans(shape)
  scale = np.sqrt(6. / (fan_in + fan_out))
  initial = tf.random_uniform(shape,
                              minval=-scale,
                              maxval=scale,
                              name='xavier_init')
  return tf.Variable(initial, dtype=tf.float32, name=name)


def _kaiming_variable(shape, name='kaiming_variable'):
  fan_in, fan_out = _get_fans(shape)
  scale = np.sqrt(6. / fan_in)
  initial = tf.random_uniform(shape,
                              minval=-scale,
                              maxval=scale,
                              name='kaiming_init')
  return tf.Variable(initial, dtype=tf.float32, name=name)

