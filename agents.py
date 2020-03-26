from abc import ABC, abstractmethod
import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F 
from models import MFEC, NEC
from optimisers import RMSprop



class _Agent(ABC):
  def __init__(self, args):
    self.train_epsilon = 0
    self.eval_epsilon = args.evaluation_epsilon
    self.training = True

  # Acts based on single set of Q-values (no batch); also returns value
  def _act(self, q_values):
    value, action = map(lambda x: x.item(), q_values.max(dim=1))  # Value of argmax policy used for evaluation
    epsilon = self.train_epsilon if self.training else self.eval_epsilon   # Use ε-greedy policy
    action = np.random.randint(0, self.action_space) if np.random.random() < epsilon else action
    return action, value

  # Save model parameters on current device (don't move model between devices)
  def save(self, path):
    torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

  def train(self):
    self.training = True
    self.online_net.train()

  def eval(self):
    self.training = False
    self.online_net.eval()

  # Sets training ε for ε-greedy policy
  def set_epsilon(self, epsilon):
    self.train_epsilon = epsilon


class _EpisodicAgent(_Agent):
  def __init__(self, args):
    super().__init__(args)

  # Acts based on single state (no batch)
  def act(self, state, return_key_value=False):
    with torch.no_grad():
      q_values, key = self.online_net(state.unsqueeze(dim=0))
      action, value = super()._act(q_values)
      if return_key_value:
        return action, key.squeeze(dim=0), value
      else:
        return action

  def update_memory_batch(self, action, keys, rewards, hashes):
    self.online_net.memories[action].update_batch(keys, rewards, hashes)

  # Evaluates Q-value based on a single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return self.online_net(state.unsqueeze(0))[0].max().item()

  # Evaluates Q-values based on a batch of states
  def evaluate_qs(self, states):
    with torch.no_grad():
      return self.online_net(states)[0]


class MFECAgent(_EpisodicAgent):
  def __init__(self, args, observation_shape, action_space, hash_size):
    super().__init__(args)
    self.action_space = action_space

    self.online_net = MFEC(args, observation_shape, action_space, hash_size).to(device=args.device)
    if args.model and os.path.isfile(args.model):
      self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))  # Always load tensors onto CPU by default, will shift to GPU if necessary
    self.online_net.train()


class NECAgent(_EpisodicAgent):
  def __init__(self, args, observation_shape, action_space, hash_size):
    super().__init__(args)
    self.action_space = action_space
    self.batch_size = args.batch_size

    self.online_net = NEC(args, observation_shape, action_space, hash_size).to(device=args.device)
    if args.model and os.path.isfile(args.model):
      self.online_net.load_state_dict(torch.load(args.model, map_location='cpu'))  # Always load tensors onto CPU by default, will shift to GPU if necessary
    self.online_net.train()

    self.optimiser = RMSprop(self.online_net.parameters(), lr=args.learning_rate, alpha=args.rmsprop_decay, eps=args.rmsprop_epsilon, momentum=args.rmsprop_momentum)

  def learn(self, mem):
    # Sample transitions with returns
    states, actions, returns = mem.sample_returns(self.batch_size)
    # Calculate Q-values
    q_values, neighbours, values, idxs, _ = self.online_net(states, learning=True)
    q_values = q_values[range(self.batch_size), actions]
    # Minimise residual between Q-values and multi-step returns
    loss = F.mse_loss(q_values, returns)
    self.optimiser.zero_grad()
    # Calculate gradients and update network parameters
    loss.backward()
    self.optimiser.step()
    # Update keys and values with gradients
    for n, v, i, m in zip(neighbours, values, idxs, self.online_net.memories):
      m.gradient_update(n, v, i)
