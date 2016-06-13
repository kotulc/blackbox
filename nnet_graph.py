# Copyright 2016 Clayton Kotulak, All Rights Reserved
#
# nnet_graph.py
# Wraper class for generating and managing a TensorFlow neural network graph.

"""
Note: Some functionality adapted from TensorFlow tutorial files
"""

import tensorflow as tf
import tfdeploy as td
import os.path
import pickle


class NNetGraph(object):

  def __init__(self, nnet_id, net_prop,
               feature_n, output_n, batch_n):
    # Use id as the directory name to store object data
    self.id = nnet_id
    self.net_prop = net_prop
    self.feature_n = feature_n
    self.output_n = output_n
    self.batch_n = batch_n
    self.nn_graph = tf.Graph()
    self.session = None

    with self.nn_graph.as_default():
      self.x = tf.placeholder(tf.float32,
                              shape=(batch_n, feature_n), name='input')
      self.keep_prob = tf.placeholder('float')
      self.logits = self.graph_inference()

  # Iterate through the layer dimensions included in net_prop to add
  # each neural network layer operation to the graph
  def graph_inference(self):
    dropout = self.net_prop['dropout']
    layer_nodes = self.net_prop['net_dims']
    layer_features = self.feature_n
    layer_len = len(layer_nodes)
    layer_h = self.x

    for layer_n in xrange(0, layer_len):
      # Add dropout to the final layer of the network if enabled
      if (dropout) and (layer_n==layer_len-1):
        layer_h = self.fc_drop_layer(layer_n, layer_features,
                                     layer_nodes[layer_n], layer_h)
      else:
        layer_h = self.fc_layer(layer_n, layer_features,
                                layer_nodes[layer_n], layer_h)
      # Update the number of features for the next layer to match
      # the current number of layer nodes
      layer_features = layer_nodes[layer_n]

    logits = self.output_layer(layer_features, layer_h)
    return logits

  def init_session(self):
    with self.nn_graph.as_default():
      if self.session is not None:
        self.session.close()
      # Initialize a new session for this subnet
      self.session = tf.Session()
      saver = tf.train.Saver()
      # Load the model variables
      if os.path.isfile(self.id+'/nnet.ckpt'):
        saver.restore(self.session, self.id+'/nnet.ckpt')
      else:
        print("\nnnet variables have not been initialized.")
        raise AttributeError

  def get_inference(self, state, subnet_idx=0):
    if self.session is None:
      self.init_session()
    feed_dict = {self.x: state, self.keep_prob: 1.0}
    logits = self.session.run(self.logits, feed_dict=feed_dict)
    return logits

  def get_model(self):
    if self.session is None:
      self.init_session()
    td_model = td.Model()
    td_model.add(self.logits, self.session)
    return td_model

  # Return a weight variable initialized with the given shape
  def weight_variable(self, shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial, name='weights')

  # Return a bias variable initialized with the given shape
  def bias_variable(self, shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial, name='biases')

  # Return the output of a fully connected relu layer
  def fc_layer(self, id, feature_n, node_n, x):
      with tf.name_scope('fc_'+str(id)):
          w_fc = self.weight_variable([feature_n, node_n])
          b_fc = self.bias_variable([node_n])
          # Apply relu operation to x
          h_fc = tf.nn.relu(tf.matmul(x, w_fc) + b_fc)
          return h_fc

  # Return the output of a relu layer with dropout
  def fc_drop_layer(self, id, feature_n, node_n, x):
      h_fc = self.fc_layer(id, feature_n, node_n, x)
      # Apply dropout to fc_layer output
      h_fc_drop = tf.nn.dropout(h_fc, self.keep_prob)
      return h_fc_drop

  # Return the output of the readout layer
  def output_layer(self, feature_n, x):
      with tf.name_scope('readout'):
          w_out = self.weight_variable([feature_n, self.output_n])
          b_out = self.bias_variable([self.output_n])
          #y = tf.nn.softmax(tf.matmul(x, w_out) + b_out)
          y = tf.nn.relu(tf.matmul(x, w_out) + b_out, name='output')
          return y