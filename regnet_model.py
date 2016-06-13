"""
Copyright 2016 Clayton Kotulak, All Rights Reserved

RegNetModel class utilizes the RegressionModel and NNetGraph classes
to calculate an action given a state or set of features associated with
a given state. NNetTrainer facilitates training the neural network using
a DataSet object.
"""

import nnet_trainer as ntrain
import nnet_graph as ngraph
import reg_model
import pickle
import numpy
import os.path


class RegNetModel(object):

  def __init__(self, id, net_prop, class_n, state_n,
               feature_n, trained=False, score=-1e9):
    self.id = id
    self.net_prop = net_prop
    self.class_n = class_n
    self.state_n = state_n
    self.feature_n = feature_n
    self.trained = trained
    self.score = score
    self.action_counts = [0]*self.class_n

    # Initialize two different inference models: the linear regression
    # baseline model and the neural network model
    self.r_model = reg_model.RegressionModel(class_n,
                                             feature_n, "reg_coefs.txt")
    self.nnet_trainer = ntrain.NNetTrainer(self.id, self.net_prop,
                                           feature_n*state_n, self.class_n)
    # If the neural network contains a dropout layer, the NNetGraph
    # object used to generate inferences will need to be initialized
    # without this property
    if net_prop['dropout'] == True:
      net_prop_alt = net_prop.copy()
      net_prop_alt['dropout'] = False
      self.nnet_graph = ngraph.NNetGraph(self.id, net_prop_alt,
                                       feature_n*state_n, self.class_n, 1)
    else:
      self.nnet_graph = ngraph.NNetGraph(self.id, net_prop,
                                       feature_n*state_n, self.class_n, 1)

    if trained == True and os.path.isfile(id+'/nnet.ckpt'):
      self.init_model()
    else:
      self.trained = False

  @classmethod
  def load(clss, regnet_id):
    data_file = open(regnet_id+'/model.dat', "rb")
    model_data = pickle.load(data_file)
    data_file.close()
    rnet_model = RegNetModel(regnet_id, model_data[0], model_data[1],
        model_data[2], model_data[3], model_data[4], model_data[5])
    return rnet_model

  def save(self):
    model_data = [self.net_prop, self.class_n, self.state_n,
                  self.feature_n, self.trained, self.score]
    data_file = open(self.id+'/model.dat', 'wb')
    pickle.dump(model_data, data_file)
    data_file.close()

  def print_stats(self):
    for action_idx in xrange(self.class_n):
      action_count = self.action_counts[action_idx]
      print("Action %d: %d instances" % (action_idx, action_count))
    self.action_counts = [0]*self.class_n

  def init_model(self):
    self.nnet_graph.init_session()
    td_model = self.nnet_graph.get_model()
    self.x, self.y = td_model.get('input','readout/output')

  def run_training(self, train_data, max_steps=None, restore=True):
    if train_data.instance_idx > self.net_prop['batch_n']:
      self.nnet_trainer.run_training(train_data,
                                     max_steps=max_steps, restore=restore)
      self.init_model()
      self.trained = True

  def get_action(self, state):
    if self.trained == True:
      logits = self.y.eval({self.x: state})
      action = numpy.argmax(logits)
      self.action_counts[action] += 1
      return action
    else:
      return 0

  def get_lreg_action(self, state):
    return self.r_model.get_action(state)



