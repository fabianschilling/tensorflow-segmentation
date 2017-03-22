from __future__ import absolute_import, division, print_function

import tensorflow as tf
import layers
import inputs
import os
import time
import sys
import argparse
import util
import importlib


class Trainer:

  def __init__(self, args):
    self.checkpoint = args.checkpoint
    self.model_name = args.model_name

    if self.checkpoint is None:
      self.model_path = os.path.join(args.model_path, self.model_name)
      self.epoch = 1
    else:
      self.checkpoint, self.model_path, self.model_name = util.get_checkpoint(args.checkpoint)
      self.epoch = int(self.checkpoint.split('-')[-1]) + 1

    # COMMENT IN THE FOLLOWING LINES FOR FINETUNING
    self.epoch = 1
    self.model_path += '_finetune'
    self.log('finetuning model {}'.format(self.model_path))

    self.learning_rate = args.learning_rate
    self.weight_decay = args.weight_decay
    self.dataset = importlib.import_module('datasets.' + args.dataset_name)
    self.model = importlib.import_module('models.fcn8s')# + self.model_name)
    self.batch_size = args.batch_size
    self.num_classes = args.num_classes

    self.log('model path: {}'.format(self.model_path))
    self.log('model checkpoint: {}'.format(self.checkpoint))

  def log(self, message):
    START = '\033[92m'
    STOP = '\033[0m'
    print(START + message + STOP)

  def run(self, run_type):

    is_training = True if run_type == 'train' else False

    self.log('{} epoch: {}'.format(run_type, self.epoch))

    image_filenames, label_filenames = self.dataset.load_filenames(run_type)

    global_step = tf.Variable(1, name='global_step', trainable=False)

    images, labels = inputs.load_batches(image_filenames,
                                         label_filenames,
                                         shape=self.dataset.SHAPE,
                                         batch_size=self.batch_size,
                                         resize_shape=self.dataset.SHAPE,
                                         crop_shape=(256, 512),
                                         augment=True)

    with tf.name_scope('labels'):
      color_labels = util.colorize(labels, self.dataset.augmented_labels)
      labels = tf.cast(labels, tf.int32)
      ignore_mask = util.get_ignore_mask(labels, self.dataset.augmented_labels)
      tf.summary.image('label', color_labels, 1)
      tf.summary.image('weights', tf.cast(ignore_mask * 255, tf.uint8), 1)

    tf.summary.image('image', images, 1)

    logits = self.model.inference(images, num_classes=self.num_classes, is_training=is_training)

    with tf.name_scope('outputs'):
      predictions = layers.predictions(logits)
      color_predictions = util.colorize(predictions, self.dataset.augmented_labels)
      tf.summary.image('prediction', color_predictions, 1)

    # Add some metrics
    with tf.name_scope('metrics'):
      accuracy_op, accuracy_update_op = tf.contrib.metrics.streaming_accuracy(
          predictions, labels, weights=ignore_mask)
      mean_iou_op, mean_iou_update_op = tf.contrib.metrics.streaming_mean_iou(
          predictions, labels, num_classes=self.num_classes, weights=ignore_mask)

    if is_training:
      loss_op = layers.loss(logits, labels, mask=ignore_mask, weight_decay=self.weight_decay)
      train_op = layers.optimize(loss_op, learning_rate=self.learning_rate, global_step=global_step)

    # Merge all summaries into summary op
    summary_op = tf.summary.merge_all()

    # Create restorer for restoring
    saver = tf.train.Saver()

    # Initialize session and local variables (for input pipeline and metrics)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())

    if self.checkpoint is None:
      sess.run(tf.global_variables_initializer())
      self.log('{} {} from scratch.'.format(run_type, self.model_name))
    else:
      start_time = time.time()
      saver.restore(sess, self.checkpoint)
      duration = time.time() - start_time
      self.log('{} from previous checkpoint {:s} ({:.2f}s)'.format(run_type, self.checkpoint, duration))

    # Create summary writer
    summary_path = os.path.join(self.model_path, run_type)
    step_writer = tf.summary.FileWriter(summary_path, sess.graph)

    # Start filling the input queues
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_examples = self.dataset.NUM_TRAIN_EXAMPLES if is_training else self.dataset.NUM_VALID_EXAMPLES

    for local_step in range(num_examples // self.batch_size):

      # Take time!
      start_time = time.time()

      if is_training:
        _, loss, accuracy, mean_iou, summary = sess.run(
            [train_op, loss_op, accuracy_update_op, mean_iou_update_op, summary_op])
        duration = time.time() - start_time
        self.log('Epoch: {} train step: {} loss: {:.4f} accuracy: {:.2f}% duration: {:.2f}s' \
            .format(self.epoch, local_step + 1, loss, accuracy * 100, duration))
      else:
        accuracy, mean_iou, summary = sess.run(
            [accuracy_update_op, mean_iou_update_op, summary_op])
        duration = time.time() - start_time
        self.log('Epoch: {} eval step: {} accuracy: {:.2f}% duration: {:.2f}s'\
            .format(self.epoch, local_step + 1, accuracy * 100, duration))

      # Save summary and print stats
      step_writer.add_summary(summary, global_step=global_step.eval(session=sess))

    # Write additional epoch summaries
    epoch_writer = tf.summary.FileWriter(summary_path)
    epoch_summaries = []
    if is_training:
      epoch_summaries.append(tf.summary.scalar('params/weight_decay', self.weight_decay))
      epoch_summaries.append(tf.summary.scalar('params/learning_rate', self.learning_rate))
    epoch_summaries.append(tf.summary.scalar('params/batch_size', self.batch_size))
    epoch_summaries.append(tf.summary.scalar('metrics/accuracy', accuracy_op))
    epoch_summaries.append(tf.summary.scalar('metrics/mean_iou', mean_iou_op))
    epoch_summary_op = tf.summary.merge(epoch_summaries)
    summary = sess.run(epoch_summary_op)
    epoch_writer.add_summary(summary, global_step=self.epoch)

    # Save after each epoch when training
    if is_training:
      checkpoint_path = os.path.join(self.model_path, self.model_name + '.checkpoint')
      start_time = time.time()
      self.checkpoint = saver.save(sess, checkpoint_path, global_step=self.epoch)
      duration = time.time() - start_time
      self.log('Model saved as {:s} ({:.2f}s)'.format(self.checkpoint, duration))

    # Stop queue runners and reset the graph
    coord.request_stop()
    coord.join(threads)
    sess.close()
    tf.reset_default_graph()


def main():

  parser = argparse.ArgumentParser(description='Run segmentation pipeline.',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-c', '--checkpoint',
                      help='path to a model checkpoint')
  parser.add_argument('-n', '--model_name',
                      help='model name',
                      default='fcn8s',
                      type=str)
  parser.add_argument('-s', '--save_name',
                      help='name to save model as',
                      type=str)
  parser.add_argument('-e', '--num_epochs',
                      help='number of epochs to run',
                      default=None,
                      type=int)
  parser.add_argument('-m', '--model_path',
                      help='path to save model checkpoints.',
                      default='/ssd/tf_models/',
                      type=str)
  parser.add_argument('-d', '--dataset_name',
                      help='name of dataset',
                      default='cityscapes',
                      type=str)
  parser.add_argument('-b', '--batch_size',
                      help='batch size',
                      default=10,
                      type=int)
  parser.add_argument('-l', '--learning_rate',
                      help='initial learning rate',
                      default=0.0001,
                      type=float)
  parser.add_argument('-w', '--weight_decay',
                      help='weight decay (l2 regularization)',
                      default=0.00005,
                      type=float)
  parser.add_argument('-nc', '--num_classes',
                      help='number of classes',
                      default=34,
                      type=int)

  args = parser.parse_args()

  trainer = Trainer(args)
  while True:
    trainer.run('train')
    trainer.run('eval')
    trainer.epoch += 1

if __name__ == '__main__':
  main()
