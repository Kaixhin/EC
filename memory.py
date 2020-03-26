import numpy as np
import torch


class ExperienceReplay():
  def __init__(self, size, observation_shape, device):
    self.device = device
    self.size = size
    self.observations = np.empty((size, ) + observation_shape, dtype=np.uint8)
    self.actions = np.empty((size, ), dtype=np.int64)
    self.returns = np.empty((size, ), dtype=np.float32) 
    self.nonterminals = np.empty((size, ), dtype=np.float32) 
    self.idx, self.steps = 0, 0
    self.full = False  # Tracks if memory has been filled/all slots are valid

  def append(self, observation, action, reward, done):
    self.observations[self.idx] = np.asarray(observation * 255, dtype=np.uint8)  # Discretise visual observations (to save memory)
    self.actions[self.idx] = action
    self.returns[self.idx] = reward  # Store 1-step reward
    self.nonterminals[self.idx] = not done
    self.idx = (self.idx + 1) % self.size
    self.steps += 1
    self.full = self.steps >= self.size

  def append_batch(self, observations, actions, returns):
    batch_size = returns.shape[0]
    idxs = np.linspace(self.idx, self.idx + batch_size - 1, batch_size, dtype=np.int64) % self.size
    self.observations[idxs] = np.asarray(observations * 255, dtype=np.uint8)  # Discretise visual observations (to save memory)
    self.actions[idxs] = actions
    self.returns[idxs] = returns
    self.idx = (self.idx + batch_size) % self.size
    self.steps += batch_size
    self.full = self.steps >= self.size

  # Returns a batch of transitions with returns, uniformly sampled from the memory
  def sample_returns(self, n):
    idxs = np.random.randint(0, self.size if self.full else self.idx, size=n)
    observations = self.observations[idxs].astype(np.float32) / 255  # Un-discretise visual observations
    return torch.as_tensor(observations).to(device=self.device), torch.as_tensor(self.actions[idxs]).to(device=self.device), torch.as_tensor(self.returns[idxs]).to(device=self.device)

  # Returns a batch of transitions with rewards, uniformly sampled from the memory
  def sample_rewards(self, n):
    valid_idxs = False
    while not valid_idxs:
      idxs = np.random.randint(0, self.size if self.full else self.idx - 1, size=n)
      valid_idxs = self.idx not in (idxs + 1) % self.size  # Avoid invalid transitions around index
    observations, next_observations = self.observations[idxs].astype(np.float32) / 255, self.observations[(idxs + 1) % self.size].astype(np.float32) / 255  # Un-discretise visual observations
    return torch.as_tensor(observations).to(device=self.device), torch.as_tensor(self.actions[idxs]).to(device=self.device), torch.as_tensor(self.returns[idxs]).to(device=self.device), torch.as_tensor(next_observations).to(device=self.device), torch.as_tensor(self.nonterminals[idxs]).to(device=self.device)


  # Set up internal state for iterator
  def __iter__(self):
    self.current_idx = 0
    return self

  # Return valid states for validation
  def __next__(self):
    if self.current_idx == self.size:
      raise StopIteration
    observation = self.observations[self.current_idx].astype(np.float32) / 255  # Un-discretise visual observations
    self.current_idx += 1
    return torch.as_tensor(observation).to(device=self.device)
