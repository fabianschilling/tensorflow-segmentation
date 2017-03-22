"""DeepLabv2 implementation: https://arxiv.org/abs/1606.00915"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import layers

def inference(inputs, num_classes=34, keep_prob=0.5, is_training=False):

  conv1_1 = layers.conv2d(inputs, ksize=3, depth=64, name='conv1_1')
  conv1_2 = layers.conv2d(conv1_1, ksize=3, depth=64, name='conv1_2')

  pool1 = layers.max_pool(conv1_2, ksize=3, stride=2, name='pool1')

  conv2_1 = layers.conv2d(pool1, ksize=3, depth=128, name='conv2_1')
  conv2_2 = layers.conv2d(conv2_1, ksize=3, depth=128, name='conv2_2')

  pool2 = layers.max_pool(conv2_2, ksize=3, stride=2, name='pool2')

  conv3_1 = layers.conv2d(pool2, ksize=3, depth=256, name='conv3_1')
  conv3_2 = layers.conv2d(conv3_1, ksize=3, depth=256, name='conv3_2')
  conv3_3 = layers.conv2d(conv3_2, ksize=3, depth=256, name='conv3_3')

  pool3 = layers.max_pool(conv3_3, ksize=3, stride=2, name='pool3')

  conv4_1 = layers.conv2d(pool3, ksize=3, depth=512, name='conv4_1')
  conv4_2 = layers.conv2d(conv4_1, ksize=3, depth=512, name='conv4_2')
  conv4_3 = layers.conv2d(conv4_2, ksize=3, depth=512, name='conv4_3')

  pool4 = layers.max_pool(conv4_3, ksize=3, stride=1, name='pool4')

  conv5_1 = layers.conv2d(pool4, ksize=3, depth=512, rate=2, name='conv5_1')
  conv5_2 = layers.conv2d(conv5_1, ksize=3, depth=512, rate=2, name='conv5_2')
  conv5_3 = layers.conv2d(conv5_2, ksize=3, depth=512, rate=2, name='conv5_3')

  pool5 = layers.max_pool(conv5_3, ksize=3, stride=1, name='pool5')

  # hole 6
  fc6_1 = layers.conv2d(pool5, ksize=3, depth=1024, rate=6, name='fc6_1')
  if is_training:
    fc6_1 = layers.dropout(fc6_1, keep_prob=keep_prob, name='drop6_1')

  fc7_1 = layers.conv2d(fc6_1, ksize=1, depth=1024, name='fc7_1')
  if is_training:
    fc7_1 = layers.dropout(fc7_1, keep_prob=keep_prob, name='drop7_1')

  fc8_1 = layers.conv2d(fc7_1, ksize=1, depth=num_classes, activation=None, name='fc8_1')

  # hole 12
  fc6_2 = layers.conv2d(pool5, ksize=3, depth=1024, rate=12, name='fc6_2')
  if is_training:
    fc6_2 = layers.dropout(fc6_2, keep_prob=keep_prob, name='drop6_2')

  fc7_2 = layers.conv2d(fc6_2, ksize=1, depth=1024, name='fc7_2')
  if is_training:
    fc7_2 = layers.dropout(fc7_2, keep_prob=keep_prob, name='drop7_2')

  fc8_2 = layers.conv2d(fc7_2, ksize=1, depth=num_classes, activation=None, name='fc8_2')

  # hole 18
  fc6_3 = layers.conv2d(pool5, ksize=3, depth=1024, rate=18, name='fc6_3')
  if is_training:
    fc6_3 = layers.dropout(fc6_3, keep_prob=keep_prob, name='drop6_3')

  fc7_3 = layers.conv2d(fc6_3, ksize=1, depth=1024, name='fc7_3')
  if is_training:
    fc7_3 = layers.dropout(fc7_3, keep_prob=keep_prob, name='drop7_3')

  fc8_3 = layers.conv2d(fc7_3, ksize=1, depth=num_classes, activation=None, name='fc8_3')

  #hole 24
  fc6_4 = layers.conv2d(pool5, ksize=3, depth=1024, rate=24, name='fc6_4')
  if is_training:
    fc6_4 = layers.dropout(fc6_4, keep_prob=keep_prob, name='drop6_4')

  fc7_4 = layers.conv2d(fc6_4, ksize=1, depth=1024, name='fc7_4')
  if is_training:
    fc7_4 = layers.dropout(fc7_4, keep_prob=keep_prob, name='drop7_4')

  fc8_4 = layers.conv2d(fc7_4, ksize=1, depth=num_classes, activation=None, name='fc8_4')

  fuse = tf.add_n([fc8_1, fc8_2, fc8_3, fc8_4], name='add')

  logits = layers.deconv2d(fuse, 16, num_classes, stride=8, bias=False, activation=None, init='bilinear', name='logits')

  return logits
