from __future__ import absolute_import, division, print_function

import tensorflow as tf
import inits


def fully_connected(inputs,
                    shape=None,
                    activation=tf.nn.relu,
                    bias=True,
                    histogram_summary=False,
                    activation_summary=False,
                    name='fully_connected'):

  if not shape:
    raise ValueError('Must specify shape for fully connected layer.')
  with tf.name_scope(name):
    weights = inits.weight_variable(shape)
    if histogram_summary:
      tf.histogram_summary('weights/' + name, weights)
    outputs = tf.matmul(inputs, weights, name='matmul')
    if bias:
      biases = inits.bias_variable([shape[-1]])
      if histogram_summary:
        tf.histogram_summary('biases/' + name, biases)
      outputs = tf.nn.bias_add(outputs, biases, name='bias_add')
    if activation is not None:
      outputs = activation(outputs, name=activation.__name__)
      if activation_summary:
        tf.histogram_summary('activations/' + name, outputs)
        tf.summary.scalar('sparsity/' + name, tf.nn.zero_fraction(outputs))

    return outputs


def conv2d(inputs, ksize, depth,
           stride=1,
           init='kaiming',
           padding='same',
           activation=tf.nn.relu,
           batchnorm=False,
           rate=1,
           bias=True,
           histogram_summary=False,
           activation_summary=False,
           name='conv2d'):

  strides = [1, stride, stride, 1]
  input_shape = inputs.get_shape().as_list()
  filter_shape = [ksize, ksize, input_shape[-1], depth]
  if rate > 1 and stride > 1:
    raise ValueError('Only stride or rate can be larger than one.')
  with tf.name_scope(name):
    weights = inits.weight_variable(filter_shape, init=init)
    tf.add_to_collection('weights', weights)
    if histogram_summary:
      tf.histogram_summary('weights/' + name, weights)
    if rate > 1:
      outputs = tf.nn.atrous_conv2d(inputs, weights, rate,
                                 padding=padding.upper(),
                                 name='atrous_conv2d')
    else:
      outputs = tf.nn.conv2d(inputs, weights,
                             strides=strides,
                             padding=padding.upper(),
                             name='conv2d')
    if bias:
      biases = inits.bias_variable([depth])
      tf.add_to_collection('biases', biases)
      if histogram_summary:
        tf.histogram_summary('biases/' + name, weights)
      outputs = tf.nn.bias_add(outputs, biases, name='bias_add')
    if batchnorm:
      with tf.name_scope('batchnorm'):
        outputs = batch_norm(outputs)
    if activation is not None:
      outputs = activation(outputs, name=activation.__name__)
      if activation_summary:
        tf.histogram_summary('activations/' + name, outputs)
        tf.summary.scalar('sparsity/' + name, tf.nn.zero_fraction(outputs))

    return outputs


def deconv2d(inputs, ksize, depth,
             output_shape=None,
             stride=1,
             init='kaiming',
             padding='same',
             activation=tf.nn.relu,
             batchnorm=False,
             bias=True,
             histogram_summary=False,
             activation_summary=False,
             name='deconv2d'):

  strides = [1, stride, stride, 1]
  batch_size, height, width, channels = inputs.get_shape().as_list()
  filter_shape = [ksize, ksize, depth, channels]
  if not output_shape:
    output_shape = [batch_size, height * stride, width * stride, depth]
  with tf.name_scope(name):
    weights = inits.weight_variable(filter_shape, init=init)
    tf.add_to_collection('weights', weights)
    if histogram_summary:
      tf.histogram_summary('weights/' + name, weights)
    outputs = tf.nn.conv2d_transpose(inputs, weights,
                                     output_shape=output_shape,
                                     strides=strides,
                                     padding=padding.upper(),
                                     name='deconvolution')
    if bias:
      biases = inits.bias_variable([depth])
      tf.add_to_collection('biases', biases)
      if histogram_summary:
        tf.histogram_summary('biases/' + name, weights)
      outputs = tf.nn.bias_add(outputs, biases, name='bias_add')
    if batchnorm:
      outputs = batch_norm(outputs)
    if activation is not None:
      outputs = activation(outputs, name=activation.__name__)
      if activation_summary:
        tf.histogram_summary('activations/' + name, outputs)
        tf.summary.scalar('sparsity/' + name, tf.nn.zero_fraction(outputs))

    return outputs


def dropout(inputs, keep_prob=0.5, name='dropout'):
  return tf.nn.dropout(inputs, keep_prob=keep_prob, name=name)


def batch_norm(inputs, name='batch_norm'):
  shape = inputs.get_shape().as_list()
  with tf.name_scope(name):
    beta = tf.Variable(tf.zeros([shape[-1]]), name='beta')
    gamma = tf.Variable(tf.ones([shape[-1]]), name='gamma')
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    training = tf.get_collection('training')[0]
    mean, var = tf.cond(training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

    return outputs


def max_pool(inputs, ksize=2, stride=2, padding='same', name='max_pool'):
  with tf.name_scope(name):
    return tf.nn.max_pool(inputs,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding=padding.upper(),
                          name=name)


def predictions(logits, name='predictions'):
  with tf.name_scope(name):
    predictions = tf.argmax(logits, dimension=3, name='argmax')
    predictions = tf.cast(predictions, tf.uint8)
    predictions = tf.expand_dims(predictions, dim=-1)
  return predictions


def confidence(logits, name='confidence'):
  with tf.name_scope(name):
    logits = tf.nn.softmax(logits, dim=-1, name='softmax')
    logits = tf.reduce_max(logits, reduction_indices=[3])
    logits = tf.expand_dims(logits, dim=-1)
  return logits


def binary_to_probability(logits, name='probability'):
  with tf.name_scope(name):
    logits = tf.nn.softmax(logits, dim=-1, name='softmax')
    zeros, ones = tf.unpack(logits, axis=-1, name='unpack')
    return tf.expand_dims(ones, dim=-1)


def loss(logits, labels, mask=None, weight_decay=0.0, name='loss'):
  with tf.name_scope(name):
    with tf.name_scope('data_loss'):
      labels = tf.squeeze(labels, squeeze_dims=[3])

      total_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name='cross_entropy')

      if mask is not None:
        with tf.name_scope('mask_loss'):
          mask = tf.squeeze(mask, squeeze_dims=[3])
          total_loss = tf.mul(total_loss, tf.cast(mask, tf.float32))

      total_loss = tf.reduce_mean(total_loss)
      tf.summary.scalar('loss/data_loss', total_loss)

    # apply l2 loss to all weights of the network
    if weight_decay > 0.0:
      with tf.name_scope('regularization_loss'):
        weights = tf.get_collection('weights')
        regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in weights]) * weight_decay
        tf.summary.scalar('loss/regularization_loss', regularization_loss)
      with tf.name_scope('total_loss'):
        total_loss += regularization_loss
        tf.summary.scalar('loss/total_loss', total_loss)

    return total_loss


def ignore_mask(labels, mappings):
  with tf.name_scope('ignore_mask'):
    labels = tf.squeeze(labels, squeeze_dims=[3])
    zeros = tf.zeros_like(labels)
    mask = tf.ones_like(labels)
    # zero out all labels with trainId == 255
    for l in mappings:
      if l.train_id == 255:
        condition = tf.equal(tf.constant(l.id, dtype=tf.uint8), labels)
        mask = tf.select(condition, zeros, mask)
    return mask

def optimize(loss, learning_rate, global_step=None, name='optimizer'):
  with tf.name_scope(name):
    opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    # opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads_and_vars = opt.compute_gradients(loss)

    # summary of gradients and variables during training
    # for grad, var in grads_and_vars:
      # layer, var_name, _ = var.op.name.split('/') # var.op.name has no redundant :n flags
      # tf.histogram_summary('gradients/' + grad.op.name, grad)
      # tf.histogram_summary('variables/' + var.op.name, var)

    return opt.apply_gradients(grads_and_vars, global_step=global_step)
