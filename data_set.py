"""
Copyright 2016 Clayton Kotulak, All Rights Reserved

object class containing system states and rewards relative to an agent's
performed action. Includes functionality to record and manage state time-
series in the pattern defined by mask_steps.
"""

import numpy
import h5py
import pickle
import os.path


class DataSet(object):

  def __init__(self, file_name, mask_steps, class_n,
               feature_n, instance_n, buffer_size):
    self.file_name = file_name
    # Mask steps identify the intervals to extract buffered state
    # features from
    self.mask_steps = mask_steps
    # Save data set dimensions
    self.class_n = class_n
    self.feature_n = feature_n
    # Initialize self.data_mask, the pattern used to extract features
    # from the buffer
    self.step_n = self.build_mask(mask_steps)
    self.instance_n = instance_n
    self.buffer_size = buffer_size
    # instance idx references the smallest unallocated array index
    self.instance_idx = 0
    # working_idx contains the indices of all allocated data
    self.working_idx = None
    # Initialize the data arrays with instance_n rows, these will be
    # expanded if instance_idx exceeds their capacity
    self.states = numpy.zeros(shape=(instance_n, feature_n*self.state_n))
    self.actions = -1*numpy.ones(shape=(instance_n))
    self.rewards = numpy.zeros(shape=(instance_n))
    # The buffer stores state data temporarily
    self.buffer_idx = self.step_n
    self.state_buffer = numpy.zeros(shape=(buffer_size, feature_n))

  @classmethod
  def load(cls, file_name):
    # Load the data_set object from the .h5 and .dat files if they exist
    if os.path.isfile(file_name+'.h5'):
      print("Loading " + file_name + "...")
      data_file = open(file_name + '_obj.dat', "rb")
      obj_data = pickle.load(data_file)
      data_file.close()
      data = DataSet(file_name, obj_data[0], obj_data[1],
              obj_data[2], obj_data[3], obj_data[4])
      data.instance_idx = obj_data[5]
      h5_file = h5py.File(file_name + '.h5', 'r')
      data.states = h5_file['states'][:]
      data.actions = h5_file['actions'][:]
      data.rewards = h5_file['rewards'][:]
      h5_file.close()
      print(file_name + " loaded.")
      return data
    else:
      raise IOError

  def save(self):
    # Save the data to a .h5 and the object to a .dat file
    print("saving " + self.file_name + "...")
    obj_data = [self.mask_steps, self.class_n, self.feature_n,
                self.instance_n, self.buffer_size, self.instance_idx]
    data_file = open(self.file_name + '_obj.dat', 'wb')
    pickle.dump(obj_data, data_file)
    data_file.close()
    h5_file = h5py.File(self.file_name + '.h5', 'w')
    h5_file.create_dataset('states', data=self.states)
    h5_file.create_dataset('actions', data=self.actions)
    h5_file.create_dataset('rewards', data=self.rewards)
    h5_file.close()
    print(self.file_name + " saved.")

  def print_stats(self):
    # Print the number of actions associationed with each saved state
    for action_idx in xrange(self.class_n):
      action_count = numpy.sum(self.actions[0:self.instance_idx] == action_idx)
      print("Action %d: %d instances" % (action_idx, action_count))

  def build_mask(self, mask_steps):
    """
    Generate a feature mask to select only those states indicated by
    the tuple of pairs mask_steps
    :param mask_steps: tuple of pairs in the format
    ([step_n, increment_n]...,)
    :return: int, the value indicating the number of states to buffer
    """
    # If mask_steps is empty, step_n is equal to 1
    if len(mask_steps)<=0:
      self.state_n = 1
      return 1
    # Generate the data_mask using dimensions in mask_steps
    else:
      data_mask = []
      # Iterate through each pair in mask_steps
      for i in xrange(0,len(mask_steps)):
        step_i = mask_steps[i]
        step_n = step_i[0]*step_i[1]
        temp_mask = [0]*step_n
        temp_mask[0:step_n:step_i[1]] = [1]*step_i[0]
        data_mask.extend(temp_mask)

      self.data_mask = numpy.array(data_mask).astype(bool)
      self.state_n = numpy.sum(self.data_mask)
      return self.data_mask.size

  def dense_to_one_hot(self, labels_dense):
    """
    Adapted from Tensorflow tutorial dataset
    :param labels_dense: numpy array, numeric labels
    :return: numpy array, labels in one hot format
    """
    assert self.class_n > numpy.amax(labels_dense)
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * self.class_n
    labels_one_hot = numpy.zeros((num_labels, self.class_n))
    labels_one_hot.flat[index_offset + (labels_dense.ravel()).astype(int)] = 1
    return labels_one_hot

  def batch_shuffle(self):
    """Shuffle the order of the working indices (rows allocated to data)"""
    # Reset epoch index and completed epoch count
    self.epoch_idx = 0

    self.working_idx = numpy.arange(self.instance_idx)
    perm = numpy.arange(self.working_idx.shape[0])
    numpy.random.shuffle(perm)
    self.working_idx = self.working_idx[perm]

  def batch_states(self, start_idx, end_idx):
    # Return all states associated with the supplied range of indices
    batch_states = self.states[self.working_idx[start_idx:end_idx]]
    return batch_states

  def batch_rewards(self, start_idx, end_idx):
    # Return all rewards associated with the supplied range of indices
    batch_rewards = self.rewards[self.working_idx[start_idx:end_idx]]
    return batch_rewards

  def batch_labels(self, start_idx, end_idx):
    # Return all labels associated with the supplied range of indices
    action_vals = self.actions[self.working_idx[start_idx:end_idx]]
    labels = self.dense_to_one_hot(action_vals)
    return labels

  def next_batch(self, batch_n):
    """
    Return a batch of data instances
    :param batch_n: the number of instances in the batch
    :return: tuple, a pair of numpy arrays
    """
    if self.working_idx is None:
      self.batch_shuffle()
    assert batch_n <= self.working_idx.shape[0]
    start_idx = self.epoch_idx
    self.epoch_idx += batch_n

    if self.epoch_idx > self.working_idx.shape[0]:
      # Shuffle the data
      start_idx = 0
      self.batch_shuffle()
      self.epoch_idx = batch_n

    end_idx = self.epoch_idx
    return (self.batch_states(start_idx, end_idx),
           self.batch_labels(start_idx, end_idx))

  def expand_arrays(self):
    """double the size of the existing instance arrays"""
    expand_array = numpy.zeros(shape=(self.instance_n,
                                      self.feature_n*self.state_n))
    self.states = numpy.concatenate((self.states, expand_array), axis=0)
    expand_array = numpy.zeros(shape=(self.instance_n))
    self.actions = numpy.concatenate((self.actions, expand_array), axis=0)
    expand_array = numpy.zeros(shape=(self.instance_n))
    self.rewards = numpy.concatenate((self.rewards, expand_array), axis=0)
    # update the instance array dimension
    self.instance_n = self.states.shape[0]

  def append(self, state, action, reward):
    """Append the given state, action, reward triple to the data set"""
    if self.instance_idx >= self.instance_n:
      self.expand_arrays()
    self.states[self.instance_idx] = state
    self.actions[self.instance_idx] = action
    self.rewards[self.instance_idx] = reward
    self.instance_idx += 1

  def shift_buffer(self):
    """
    Move the last step_n elements from the end of the buffer to the
    beginning, resetting all subsequent elements to zero to make space
    """
    # Copy step_n latest elements
    state_temp = self.state_buffer[
                 self.buffer_size-self.step_n:self.buffer_size]
    # Reset all buffer state elements to zero
    self.state_buffer = self.state_buffer*0
    # Copy state elements to the beginning of the buffer
    self.state_buffer[0:self.step_n] = state_temp
    self.buffer_idx = self.step_n

  def update_buffer(self, state):
    """Add a new set of state features to the buffer"""
    # If at the end of the buffer, make some space
    if self.buffer_idx >= self.buffer_size:
      self.shift_buffer()
    # Add the state to the buffer
    self.state_buffer[self.buffer_idx] = state
    # Shift iterator to next available index
    self.buffer_idx += 1

  def get_buffer(self):
    """Return the buffer state features as selected by the data mask"""
    assert self.data_mask is not None
    start_idx = self.buffer_idx - self.step_n
    state_block = self.state_buffer[start_idx:self.buffer_idx]
    state_features = state_block[self.data_mask, :]
    state_features = numpy.reshape(state_features,
                                   (self.state_n*self.feature_n))
    return state_features

  def commit_buffer(self, action, reward):
    """Append the buffer state, action, reward triple to the data set"""
    state_features = self.get_buffer()
    self.append(state_features, action, reward)

  def clear_buffer(self):
    """Reset all buffer values to zero, reset the buffer start index"""
    self.state_buffer = self.state_buffer*0
    self.buffer_idx = self.step_n

  def backup_buffer(self):
    """Save a copy of the state buffer and buffer index"""
    self.state_buffer_bak = self.state_buffer.copy()
    self.buffer_idx_bak = self.buffer_idx

  def restore_buffer(self):
    """Restore the state buffer from an earlier save"""
    assert self.state_buffer_bak is not None
    self.state_buffer = self.state_buffer_bak
    self.buffer_idx = self.buffer_idx_bak