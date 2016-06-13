"""
Copyright 2016 Clayton Kotulak, All Rights Reserved

Train the neural network [nnet] bot on the BlackBox training level.
Start by seeding the train_data data set with samples from several run-
throughs using the linear regression model [lreg or reg_model]. After
every update_n steps, train the nnet with the new data in an attempt to
better replicate the simple lreg model. Once the nnet has come close to
matching the baseline performance of the lreg model, begin to supplament
the training data with samples predicted to bring greater rewards based
on the action_lookup function.
"""

import interface as bbox

import os.path
import numpy
import regnet_model
import data_set


# Global variables as a simple training interface
# seed_model enables the loop to generate the initial training
# data set, matching the performance of the benchmark lreg model
seed_model = False
# train_model enables play-through of additional game sessions to
# supplament the training data set using action_lookup
train_model = True
# validate_model enables play-through of the black box validation
# level to determine the bots ability to generalize
validate_model = True

# The number of full train-validate loop iterations to perform
training_iter = 1
# The number of play-throughs in each training iteration
session_iter = 1
# The number of steps to elapse before training the nnet with new data
update_n = 80000
# The number of training steps to perform at each nnet update
update_nnet = 5000
# The initial size of the data set state/action/reward arrays
instance_count = 20000
# The size of the data set buffer used to temporarily store states
buffer_size = 5000
# The number of random float values to calculate in a single batch
rand_n = 50000
# The probability of saving a given state/action/reward to the data set
sample_prob = 0.002
# The probability of performing n random actions to better sample the
# state space distribution
perturb_prob = 0.02
# The minimum number of random actions to perform
rand_min = 20
# The maximum number of random actions to perform
rand_max = 150

# The number of features contained in each Black Box game state
feature_n = 36
# The number of available actions in each state
action_n = 4


def prepare_bbox(train_level=True):
  """
  Load the training level by default, use 'test_level' as a means
  of validating the bots generalization performance.
  :param train_level: boolean, load the training level if True
  :return: None
  """
  if train_level:
    bbox.load_level("../levels/train_level.data", verbose=1)
  else:
    bbox.load_level("../levels/test_level.data", verbose=1)


def action_lookup(model, train_data, step_inc):
  """
  At any given point, use action_lookup to determine the ideal action
  from the current state. Use the behavior of the model following each
  possible action to determine that which brings the greatest reward.
  :param model: object with a get_action method for action inference
  :param train_data: DataSet object used for bbox state buffering
  :param step_inc: int, the number of state steps to increment for each
    possible action of action_n total actions
  :return: (int, float), the tuple representing the highest scoring
    action
  """

  # Create a checkpoint to revert to after each action lookup
  start_checkpoint = bbox.create_checkpoint()
  # Similarly, create a backup of the DataSet object state buffer
  train_data.backup_buffer()
  best_score = -1e9
  best_action = -1

  # Perform the forward lookup for all valid actions
  for action_idx in xrange(action_n):
    start_score = bbox.get_score()
    bbox.do_action(action_idx)
    train_data.update_buffer(bbox.get_state())

    # After the initial action selection, use the model inference to
    # continue step_inc states into the future
    for _ in xrange(step_inc):
      action = model.get_action(train_data.get_buffer())
      bbox.do_action(action)
      train_data.update_buffer(bbox.get_state())

    # Check the score delta step_inc steps after the initial aciton
    end_score = bbox.get_score()
    score_delta = end_score - start_score
    if score_delta > best_score:
      best_score = score_delta
      best_action = action_idx
    bbox.load_from_checkpoint(start_checkpoint)
    train_data.restore_buffer()

  return best_action, best_score


def learn_bbox(rnet_model, train_data, update_inc=5000,
               lookup_inc=250, seed_data=False):
  """
  Add training instances to train_data from a single run-through of a
  bbox session.
  :param rnet_model: model object with get_lreg_action and get_action
    methods
  :param train_data: DataSet object used to buffer states and append
    new training instances
  :param update_inc: int, number of steps between each nnet model update
  :param lookup_inc: int, number of forward action lookup steps
  :param seed_data: boolean, sets best_action is the action returned by
    the lreg model.
  :return: int, the number of action errors, or differences between
    actions produced by the rnet_model and the ideal or seed model.
  """
  has_next = 1
  error_count = 0
  rand_count = 0
  rand_idx = rand_n

  prepare_bbox()
  # For each new state in the session, add it to the data set's state
  # buffer so that historical states are included in a commit event
  train_data.clear_buffer()
  current_state = bbox.get_state()
  train_data.update_buffer(current_state)

  while has_next:
    # If all random values have been used, generate a new batch
    if rand_idx >= (rand_n-1):
      rand_vals = numpy.random.random_sample(size=(rand_n))
      rand_idx = 0

    step_count = bbox.get_time()
    # Get the next action from the model based on the current set of
    # buffered states
    action = rnet_model.get_action(train_data.get_buffer())

    # Every update_inc steps train the model's network with newly
    # acquired training data
    if step_count % update_inc == 0:
      rn_model.run_training(train_data, max_steps=update_nnet, restore=True)
      error_count = 0
      rand_count = 0
    # If the random value is less than or equal to the sample
    # probability, sample the current session state and determine the
    # best action, adding it to the training set if necessary
    elif rand_vals[rand_idx] <= sample_prob:
      if seed_data:
        best_action = rnet_model.get_lreg_action(current_state)
        score_delta = 0.1
      else:
        best_action, score_delta = action_lookup(rnet_model,
                                                 train_data, lookup_inc)
      if action != best_action:
        train_data.commit_buffer(best_action, score_delta)
        error_count += 1
      rand_count += 1
    # Add random variation to the session by performing a random action
    # if less than or equal to perturb probability
    if rand_vals[rand_idx+1] <= perturb_prob:
      action = numpy.random.randint(0,4)
      step_inc = numpy.random.randint(rand_min, rand_max)
      for _ in xrange(step_inc):
        has_next = bbox.do_action(action)
        current_state = bbox.get_state()
        train_data.update_buffer(current_state)
    else:
      has_next = bbox.do_action(action)
      current_state = bbox.get_state()
      train_data.update_buffer(current_state)

    rand_idx += 2
    if step_count % 5000 == 0:
      print ("time = %d, score = %f" % (step_count, bbox.get_score()))
      print ("errors = %d, samples = %d" % (error_count, rand_count))
      #rn_model.print_stats()

  bbox.finish(verbose=1)
  return error_count


def run_bbox(rnet_model, train_data,
             train_level=True, verbose=True):
  """
  Run a single session of the black box training or test environments
  :param rnet_model: model with a get_action(state) method
  :param train_data: a DataSet object used to buffer each state
  :param train_level: boolean, run the training level if True
  :param verbose: boolean, display additional information if True
  :return: float, the final session score
  """
  has_next = 1
  prepare_bbox(train_level)
  train_data.clear_buffer()

  while has_next:
    step_count = bbox.get_time()
    train_data.update_buffer(bbox.get_state())
    state = train_data.get_buffer()
    action = rnet_model.get_action(state)
    has_next = bbox.do_action(action)

    if step_count % 5000 == 0 and verbose:
      print ("time = %d, score = %f" % (step_count, bbox.get_score()))

  final_score = bbox.finish(verbose=1)
  return final_score


if __name__ == "__main__":
  # Network properties
  net_prop = {'net_dims': [250,100],
              'batch_n': 250,
              'ensemble': False,
              'dropout': True}
  # Model data will be stored under the directory named by rnet_id
  rnet_id = 'rnet_model'
  # Mask steps is the pattern of historical states stored in each istance
  mask_steps = ([2,10],[2,5],[2,2],[2,1])
  state_n = 8

  # Check to see if the model data exists
  if os.path.isfile(rnet_id+'/model.dat'):
    rn_model = regnet_model.RegNetModel.load(rnet_id)
  else:
    rn_model = regnet_model.RegNetModel(rnet_id, net_prop,
                                        action_n, state_n, feature_n)

  # Check to see if the training data exists
  file_name = rnet_id+'/data'
  if os.path.isfile(file_name+'.h5'):
    train_data = data_set.DataSet.load(file_name)
  else:
    train_data = data_set.DataSet(file_name, mask_steps, action_n,
                                  feature_n, instance_count, buffer_size)

  # If the model has not yet been trained, do so now
  if rn_model.trained == False:
    rn_model.run_training(train_data, max_steps=85000, restore=False)
    rn_model.save()

  if seed_model:
    # Seed the training data: match performance of linear reg. model
    error_count = 1
    while error_count > 0:
      error_count = learn_bbox(rn_model, train_data,
                               update_inc=update_n, seed_data=True)
      print("train data stats:")
      train_data.print_stats()
      train_data.save()

    print("No nnet action errors.")
    new_score = run_bbox(rn_model, train_data,
                         train_level=False, verbose=True)

  # Iterate through train-validate loops
  for _ in xrange(training_iter):
    if train_model:
      for _ in xrange(session_iter):
        # Run a single bbox session to collect training data in the
        # train_data DataSet object
        learn_bbox(rn_model, train_data, update_inc=update_n, lookup_inc=250)
        print("\ntrain data stats:")
        train_data.print_stats()
        train_data.save()

    if validate_model:
      # Run a single validation session to determine the model's
      # ability to generalize
      best_score = rn_model.score
      new_score = run_bbox(rn_model, train_data,
                           train_level=False, verbose=True)
      print("\nmodel stats:")
      rn_model.print_stats()
      if new_score > best_score:
        rn_model.score = new_score
        rn_model.save()
        print("\nnew rnet score: %d" % best_score)
