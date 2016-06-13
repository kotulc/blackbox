# Copyright 2016 Clayton Kotulak, All Rights Reserved
#
# nnet_trainer.py
# Functionality for initializing and training a neural network graph
# with TensorFlow.

"""
Note: Some functionality adapted from TensorFlow tutorial files
"""

import tensorflow as tf
import nnet_graph
import os.path
import pickle
import time

# Model parameters as flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('max_steps', 35000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.5, 'Output dropout probability.')


class NNetTrainer(object):

  def __init__(self, nnet_id, net_prop, feature_n, class_n):
    # Use id as the directory name to store object data
    self.id = nnet_id
    self.batch_n = net_prop['batch_n']
    self.inf_graph = nnet_graph.NNetGraph(nnet_id, net_prop, feature_n,
                                          class_n, self.batch_n)

    with self.inf_graph.nn_graph.as_default():
      # Generate placeholders for the data
      self.y = tf.placeholder(tf.float32, shape=(self.batch_n, class_n))
      self.z = tf.placeholder(tf.float32, shape=(self.batch_n))
      # Build the graph with all operations required for training
      self.loss = self.graph_loss(self.inf_graph.logits, self.y, self.z)
      self.train_op = self.graph_training(self.loss)
      self.eval_correct = self.graph_evaluation(self.inf_graph.logits,
                                                self.y)

  # Calculate the loss by comparing logits and labels
  def graph_loss(self, logits, labels, weights):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
              logits, labels, name='xentropy')
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
      return loss

  # Return the optimizer operation for this model
  def graph_training(self, loss):
      tf.scalar_summary(loss.op.name, loss)
      optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
      global_step = tf.Variable(0, name='global_step', trainable=False)
      train_op = optimizer.minimize(loss, global_step=global_step)
      return train_op

  # Return a tensor with the number of instances correctly predicted
  def graph_evaluation(self, logits, labels):
      correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
      #correct_prediction = tf.nn.in_top_k(logits, labels, 1)
      return tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

  # Fill_feed_dict from fully_connected_feed.py
  def fill_feed_dict(self, data_set, keep):
      #x_feed, y_feed, z_feed = data_set.next_batch(self.batch_n)
      x_feed, y_feed = data_set.next_batch(self.batch_n)
      feed_dict = {self.inf_graph.x: x_feed, self.y: y_feed, keep[0]: keep[1]}
      assert x_feed.shape[0] == self.batch_n
      return feed_dict

  # Runs a single evaluation against a full epoch of data
  def evaluate(self, sess, data_set, keep):
      true_count = 0
      steps_per_epoch = data_set.instance_idx // self.batch_n
      example_n = steps_per_epoch * self.batch_n

      for step in xrange(steps_per_epoch):
          feed_dict = self.fill_feed_dict(data_set, keep)
          true_count += sess.run(self.eval_correct, feed_dict=feed_dict)

      precision = float(true_count) / example_n
      print("Num examples: %d  Num correct: %d  Precision @ 1: %0.04f\n" %
            (example_n, true_count, precision))

  # Create the session, initialize variables, and run training loop
  def run_training(self, train_data, val_data=None,
                   max_steps=None, subnet_idx=0, restore=True):
    if max_steps is None:
      max_steps = FLAGS.max_steps

    with self.inf_graph.nn_graph.as_default():
      summary_op = tf.merge_all_summaries()
      saver = tf.train.Saver()
      sess = tf.Session()

      # Load the model variables if available and load_var enabled
      if os.path.isfile(self.id+'/nnet.ckpt')\
              and restore == True:
        saver.restore(sess, self.id +'/nnet.ckpt')
        print("\nnnet restored from file.\n")
      else:
        print("\nnnet generated.\n")
        sess.run(tf.initialize_all_variables())
      summary_writer = tf.train.SummaryWriter(self.id+'/temp', sess.graph)

      keep = (self.inf_graph.keep_prob, FLAGS.keep_prob)

      for step in xrange(max_steps):
        start_time = time.time()
        feed_dict = self.fill_feed_dict(train_data, keep)
        _, loss_value = sess.run([self.train_op, self.loss],
                                 feed_dict=feed_dict)
        duration = time.time() - start_time

        if step%500==0:
          # Print status to stdout
          print("Step %d: loss=%.2f (%.3f sec)" %
                (step, loss_value, duration))
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()

        # Evaluate the model after training
        if (step+1)>=max_steps:
          saver.save(sess, self.id +'/nnet.ckpt')
          print("\nTraining data eval:")
          self.evaluate(sess, train_data,
                        (self.inf_graph.keep_prob, 1.0))

          if val_data!=None:
            print("Validation data eval:")
            self.evaluate(sess, val_data,
                          (self.inf_graph.keep_prob, 1.0))

      sess.close()