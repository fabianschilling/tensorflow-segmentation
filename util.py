from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
import shutil
import sys
import pickle
from collections import namedtuple


def get_checkpoint(checkpoint_dir):
  checkpoint_path = os.path.abspath(checkpoint_dir)
  if not os.path.exists(checkpoint_path):
    raise IOError('Checkpoint {} does not exist.'.format(checkpoint_path))

  if checkpoint_path.endswith('checkpoint'):
    checkpoint_path, _ = os.path.split(checkpoint_path)

  state = tf.train.get_checkpoint_state(checkpoint_path)
  if state is not None:
    checkpoint = state.model_checkpoint_path
  else:
    checkpoint = checkpoint_path

  path, _ = os.path.split(checkpoint)
  _, model_name = os.path.split(path)
  return checkpoint, path, model_name


def save_source(module, model_path):
  # get absolute path of module
  src_path = os.path.abspath(module.__file__)
  # get only filename (with and without extension)
  filename_with_ext = src_path.split('/')[-1]
  filename = filename_with_ext.split('.')[0]

  # create directory if necessary
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  # copy module content to destination
  dst_path = os.path.join(model_path, filename_with_ext)
  print('Copying {} -> {}'.format(src_path, dst_path))
  shutil.copyfile(src_path, dst_path)


def count(tensor, value):
  equal_to_value = tf.equal(tensor, value)
  as_ints = tf.cast(equal_to_value, tf.int32)
  return tf.reduce_sum(as_ints)


def label_count_summary(labels, mappings):
  with tf.name_scope('count_labels'):
    total_cnt = tf.size(labels)
    for l in mappings:
      cnt = count(labels, l.id)
      label_name = l.name.replace(' ', '_')
      full_name = '{}/{}/{}'.format('labels', l.id, label_name)
      perc = tf.div(tf.cast(cnt, tf.float32), tf.cast(total_cnt, tf.float32))
      tf.scalar_summary(full_name, perc)


def moment_summary(tensor):
  mean, variance = tf.nn.moments(tensor, [0, 1, 2, 3])
  tf.scalar_summary('images/mean', mean)
  tf.scalar_summary('images/variance', variance)


def colorize(images, labels, name='colorize'):
  with tf.name_scope(name):
    num_batches, height, width, channels = images.get_shape().as_list()
    shape = (height, width, channels)
    images_list = tf.unpack(images)
    color_list = []
    for image in images_list:
      red = tf.zeros(shape, dtype=tf.uint8)
      grn = tf.zeros(shape, dtype=tf.uint8)
      blu = tf.zeros(shape, dtype=tf.uint8)
      for label in labels:
        comp = tf.equal(image, tf.constant(label.id, dtype=tf.uint8))
        red = tf.select(comp, tf.constant(label.color[0], dtype=tf.uint8, shape=shape), red)
        grn = tf.select(comp, tf.constant(label.color[1], dtype=tf.uint8, shape=shape), grn)
        blu = tf.select(comp, tf.constant(label.color[2], dtype=tf.uint8, shape=shape), blu)
      color_list.append(tf.squeeze(tf.pack([red, grn, blu], axis=2)))
    color_batch = tf.pack(color_list)

  return color_batch


def coarsen(labels, mappings, name='coarsen'):
  with tf.name_scope(name):
    labels = tf.squeeze(labels)
    coarse = tf.zeros_like(labels, dtype=tf.uint8)
    shape = coarse.get_shape().as_list()
    for l in mappings:
      comp = tf.equal(labels, tf.constant(l.id, dtype=tf.uint8))
      coarse = tf.select(comp, tf.constant(l.drivability, dtype=tf.uint8, shape=shape), coarse)
    return coarse


def fine_to_coarse(labels, dictionary):
  height, width, _ = labels.get_shape().as_list()
  labels = tf.reshape(labels, [-1])
  for fine, coarse in dictionary.items():
    comparison = tf.equal(labels, tf.constant(fine, dtype=tf.uint8))
    new_labels = tf.constant(coarse, dtype=tf.uint8, shape=labels.get_shape().as_list())
    labels = tf.select(comparison, new_labels, labels)
  labels = tf.reshape(labels, (height, width, 1))
  return labels


def label_to_navigable(labels, mappings, name='to_navigable'):
  with tf.name_scope(name):
    nav_labels = tf.zeros_like(labels)
    for l in mappings:
      comp = tf.equal(labels, tf.constant(l.id, dtype=tf.int32))
      new_label = tf.constant(l.navigability, dtype=tf.int32, shape=labels.get_shape())
      nav_labels = tf.select(comp, new_label, nav_labels)
    return nav_labels


def assign_weights(weights_path, name='assign'):
  weights = pickle.load(open(weights_path, 'rb'))
  assign_ops = []
  with tf.name_scope('assign'):
    for var in tf.trainable_variables():
      layer_name, var_name, _ = var.op.name.split('/')
      if layer_name in weights and var_name in weights[layer_name]:
        assign_ops.append(var.assign(weights[layer_name][var_name]))
        print('Assigning to {} from {}/{}/{}'.format(var.op.name, weights_path, layer_name, var_name))
  return assign_ops

def get_ignore_mask(labels, mappings, name='ignore_mask'):
  with tf.name_scope(name):
    ignore_mask = tf.ones_like(labels)
    for l in mappings:
      if l.ignore:
        comp = tf.equal(labels, tf.constant(l.id, dtype=tf.int32))
        ignore_mask = tf.select(comp, tf.zeros_like(labels), ignore_mask)
  return ignore_mask


def main():
  checkpoint_dir = sys.argv[-1]
  print('Reading from {}'.format(checkpoint_dir))
  print(get_checkpoint(checkpoint_dir))


if __name__ == '__main__':
  main()
