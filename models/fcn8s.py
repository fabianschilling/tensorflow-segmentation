"""FCN8s implementation: https://arxiv.org/pdf/1605.06211v1.pdf"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import layers

def inference(inputs, num_classes=34, is_training=False):

  conv1_1 = layers.conv2d(inputs, 3, 64, name='conv1_1')
  conv1_2 = layers.conv2d(conv1_1, 3, 64, name='conv1_2')

  pool1 = layers.max_pool(conv1_2, name='pool1')

  conv2_1 = layers.conv2d(pool1, 3, 128, name='conv2_1')
  conv2_2 = layers.conv2d(conv2_1, 3, 128, name='conv2_2')

  pool2 = layers.max_pool(conv2_2, name='pool2')

  conv3_1 = layers.conv2d(pool2, 3, 256, name='conv3_1')
  conv3_2 = layers.conv2d(conv3_1, 3, 256, name='conv3_2')
  conv3_3 = layers.conv2d(conv3_2, 3, 256, name='conv3_3')

  pool3 = layers.max_pool(conv3_3, name='pool3')

  conv4_1 = layers.conv2d(pool3, 3, 512, name='conv4_1')
  conv4_2 = layers.conv2d(conv4_1, 3, 512, name='conv4_2')
  conv4_3 = layers.conv2d(conv4_2, 3, 512, name='conv4_3')

  pool4 = layers.max_pool(conv4_3, name='pool4')

  conv5_1 = layers.conv2d(pool4, 3, 512, name='conv5_1')
  conv5_2 = layers.conv2d(conv5_1, 3, 512, name='conv5_2')
  conv5_3 = layers.conv2d(conv5_2, 3, 512, name='conv5_3')

  pool5 = layers.max_pool(conv5_3, name='pool5')

  fc6 = layers.conv2d(pool5, 7, 4096, name='fc6')

  if is_training:
    fc6 = layers.dropout(fc6, keep_prob=0.5, name='drop6')

  fc7 = layers.conv2d(fc6, 1, 4096, name='fc7')

  if is_training:
    fc7 = layers.dropout(fc7, keep_prob=0.5, name='drop7')

  score_fr = layers.conv2d(fc7, 1, num_classes, name='score_fr')

  upscore2 = layers.deconv2d(score_fr, 4, num_classes, stride=2, bias=False, activation=None, init='bilinear', name='upscore2')

  score_pool4 = layers.conv2d(pool4, 1, num_classes, activation=None, name='score_pool4')
  fuse_pool4 = tf.add(upscore2, score_pool4, name='fuse_pool4')
  upscore4 = layers.deconv2d(fuse_pool4, 4, num_classes, stride=2, bias=False, activation=None, init='bilinear', name='upscore4')

  score_pool3 = layers.conv2d(pool3, 1, num_classes, activation=None, name='score_pool3')
  fuse_pool3 = tf.add(upscore4, score_pool3, name='fuse_pool3')
  upscore8 = layers.deconv2d(fuse_pool3, 16, num_classes, stride=8, bias=False, activation=None, init='bilinear', name='upscore8')

  return upscore8

