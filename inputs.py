from __future__ import absolute_import, division, print_function

import tensorflow as tf
import importlib

NUM_THREADS = 8

def load_batches(image_filenames=None,
                 label_filenames=None,
                 shape=(1024, 2048),
                 augment=False,
                 batch_size=1,
                 shuffle=True,
                 resize_shape=None,
                 crop_shape=None,
                 num_epochs=None,
                 file_type='png'):

  contents_list = list()
  filename_list = list()

  if image_filenames is not None:
    filename_list.append(image_filenames)

  if label_filenames is not None:
    filename_list.append(label_filenames)

  capacity = 3 * batch_size

  if crop_shape is not None:
    resize_height, resize_width = resize_shape
    crop_height, crop_width = crop_shape
    offset_height = tf.random_uniform([], minval=0,
        maxval=(resize_height - crop_height), dtype=tf.int32, name='offset_height')
    offset_width = tf.random_uniform([], minval=0,
        maxval=(resize_width - crop_width), dtype=tf.int32, name='offset_width')
    crop_shape = (offset_height, offset_width, crop_height, crop_width)


  # Process all inputs
  with tf.name_scope('inputs'):
    queues = tf.train.slice_input_producer(filename_list,
        num_epochs=num_epochs,
        shuffle=shuffle,
        capacity=capacity)

    # Process images
    if image_filenames is not None:
      image = read_rgb(queues[0], file_type=file_type, crop_shape=crop_shape, resize_shape=resize_shape, augment=augment)
      contents_list.append(image)

    # Process labels
    if label_filenames is not None:
      label = read_label(queues[1], file_type=file_type, crop_shape=crop_shape, resize_shape=resize_shape)
      contents_list.append(label)

    batches = tf.train.shuffle_batch(contents_list,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=batch_size,
        num_threads=NUM_THREADS)

    return batches


def read_rgb(queue, file_type='png', dtype=tf.uint8, crop_shape=None, resize_shape=None, augment=False, name='rgb'):
  with tf.name_scope(name):
    image = _read_image(queue, channels=3, file_type=file_type, dtype=dtype)
    if resize_shape is not None:
      image = _resize_image(image, shape=resize_shape)
    if crop_shape is not None:
      image = _crop_image(image, shape=crop_shape)
    with tf.name_scope('standardize'):
      mean, variance = tf.nn.moments(image, axes=[0, 1, 2])
      image = ((image - mean) / tf.sqrt(variance))
    #with tf.name_scope('whiten'):
      #image = tf.image.per_image_whitening(image)
    if augment:
      with tf.name_scope('augmentation'):
        image = _augmentation(image)
  return image


def read_label(queue, file_type='png', dtype=tf.uint8, crop_shape=None, resize_shape=None, name='label'):
  with tf.name_scope('label'):
    label = _read_image(queue, channels=1, file_type=file_type, dtype=dtype)
    if resize_shape is not None:
      label = _resize_image(label, shape=resize_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if crop_shape is not None:
      label = _crop_image(label, shape=crop_shape)
  return label


def _normalize_image(image, name='normalize'):
  with tf.name_scope(name):
    minimum = tf.reduce_min(image)
    maximum = tf.reduce_max(image)
    return (image - minimum) / (maximum - minimum)


def _crop_image(image, shape, name='crop'):
  with tf.name_scope(name):
    offset_height, offset_width, height, width = shape
    return tf.image.crop_to_bounding_box(image, offset_height, offset_width, height, width)


def _resize_image(image, shape, method=tf.image.ResizeMethod.BICUBIC, name='resize'):
  with tf.name_scope(name):
    height, width = shape
    return tf.image.resize_images(image, (height, width), method=method)


def _read_image(queue, channels=1, file_type='png', dtype=tf.uint8, name='read'):
  with tf.name_scope(name):
    contents = tf.read_file(queue)
    if file_type == 'png':
      image = tf.image.decode_png(contents, channels=channels, dtype=dtype)
    elif file_type == 'jpg':
      image = tf.image.decode_jpeg(contents, channels=channels)
    else:
      raise ValueError('Unrecognized file type {}.'.format(file_type))
    return image


def _augmentation(image):
  image = tf.image.random_brightness(image, max_delta=0.25)
  image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
  image = tf.image.random_hue(image, max_delta=0.125)
  image = tf.image.random_saturation(image, lower=0.75, upper=1.25)
  return image
