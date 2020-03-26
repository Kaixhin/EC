import argparse
from datetime import datetime
import json
import os
import atari_py
import numpy as np
from tqdm import tqdm
import torch
from torch import optim

from agents import MFECAgent, NECAgent
from envs import AtariEnv
from memory import ExperienceReplay
from test import test


# Hyperparameters
parser = argparse.ArgumentParser(description='Episodic Control')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--game', type=str, default='pong', choices=atari_py.list_games(), help='ATARI game')
parser.add_argument('--T-max', type=int, default=int(10e6), metavar='STEPS', help='Number of training steps (4x number of frames)')
parser.add_argument('--max-episode-length', type=int, default=int(108e3), metavar='LENGTH', help='Max episode length (0 to disable)')
parser.add_argument('--history-length', type=int, default=4, metavar='T', help='Number of consecutive states processed (ATARI)')  # 1 for MFEC (originally), 4 for MFEC (in NEC paper) and NEC
parser.add_argument('--algorithm', type=str, default='MFEC', choices=['MFEC', 'NEC'], help='Algorithm')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Hidden size')
parser.add_argument('--key-size', type=int, default=64, metavar='SIZE', help='Key size')  # 64 for MFEC, 128 for NEC
parser.add_argument('--num-neighbours', type=int, default=11, metavar='p', help='Number of nearest neighbours')  # 11 for MFEC, 50 for NEC
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e5), metavar='CAPACITY', help='Experience replay memory capacity')
parser.add_argument('--dictionary-capacity', type=int, default=int(1e6), metavar='CAPACITY', help='Dictionary capacity (per action)')  # 1e6 for MFEC, 5e5 for NEC
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--episodic-multi-step', type=int, default=1e6, metavar='n', help='Number of steps for multi-step return from end of episode')  # Infinity for MFEC, 100 for NEC
parser.add_argument('--epsilon-initial', type=float, default=1, metavar='ε', help='Initial value of ε-greedy policy')
parser.add_argument('--epsilon-final', type=float, default=0.005, metavar='ε', help='Final value of ε-greedy policy')  # 0.005 for MFEC, 0.001 for NEC
parser.add_argument('--epsilon-anneal-start', type=int, default=5000, metavar='ε', help='Number of steps before annealing ε')
parser.add_argument('--epsilon-anneal-end', type=int, default=25000, metavar='ε', help='Number of steps to finish annealing ε')
parser.add_argument('--discount', type=float, default=1, metavar='γ', help='Discount factor')  # 1 for MFEC, 0.99 for NEC
parser.add_argument('--learning-rate', type=float, default=7.92468721e-6, metavar='η', help='Network learning rate')
parser.add_argument('--rmsprop-decay', type=float, default=0.95, metavar='α', help='RMSprop decay')
parser.add_argument('--rmsprop-epsilon', type=float, default=0.01, metavar='ε', help='RMSprop epsilon')
parser.add_argument('--rmsprop-momentum', type=float, default=0, metavar='M', help='RMSprop momentum')
parser.add_argument('--dictionary-learning-rate', type=float, default=0.1, metavar='α', help='Dictionary learning rate')
parser.add_argument('--kernel', type=str, default='mean_IDW', choices=['mean', 'mean_IDW'], metavar='k', help='Kernel function')  # mean for MFEC, mean_IDW for NEC
parser.add_argument('--kernel-delta', type=float, default=1e-3, metavar='δ', help='Mean IDW kernel delta')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--learn-start', type=int, default=0, metavar='STEPS', help='Number of steps before starting training')  # 0 for MFEC, 50000 for NEC
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=100000, metavar='STEPS', help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=10, metavar='N', help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=500, metavar='N', help='Number of transitions to use for validating Q')
parser.add_argument('--evaluation-epsilon', type=float, default=0, metavar='ε', help='Value of ε-greedy policy for evaluation')
parser.add_argument('--checkpoint-interval', type=int, default=0, metavar='STEPS', help='Number of training steps between saving buffers (0 to disable)')  # TODO
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


# Simple ISO 8601 timestamped logger
def log(s):
  print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)


# Setup
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, 'args.json'), 'w') as f:
  json.dump(args.__dict__, f, indent=2)
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(np.random.randint(1, 10000))
else:
  args.device = torch.device('cpu')
metrics = {'train_steps': [], 'train_episodes': [], 'train_rewards': [], 'test_steps': [], 'test_rewards': [], 'test_Qs': []}


# Environment
env = AtariEnv(args)
env.train()


# Agent and memory
if args.algorithm == 'MFEC':
  agent = MFECAgent(args, env.observation_space.shape, env.action_space.n, env.hash_space.shape[0])
elif args.algorithm == 'NEC':
  agent = NECAgent(args, env.observation_space.shape, env.action_space.n, env.hash_space.shape[0])
  mem = ExperienceReplay(args.memory_capacity, env.observation_space.shape, args.device)


# Construct validation memory
val_mem = ExperienceReplay(args.evaluation_size, env.observation_space.shape, args.device)
T, done, states = 0, True, []  # Store transition data in episodic buffers
while T < args.evaluation_size:
  if done:
    state, done = env.reset(), False
  states.append(state.cpu().numpy())  # Append transition data to episodic buffers
  state, _, done = env.step(env.action_space.sample())
  T += 1
val_mem.append_batch(np.stack(states), np.zeros((args.evaluation_size, ), dtype=np.int64), np.zeros((args.evaluation_size, ), dtype=np.float32))


if args.evaluate:
  agent.eval()  # Set agent to evaluation mode
  test_rewards, test_Qs = test(args, 0, agent, val_mem, results_dir, evaluate=True)  # Test
  print('Avg. reward: ' + str(sum(test_rewards) / args.evaluation_episodes) + ' | Avg. Q: ' + str(sum(test_Qs) / args.evaluation_size))
else:
  # Training loop
  agent.train()
  T, done, epsilon = 0, True, args.epsilon_initial
  agent.set_epsilon(epsilon)
  for T in tqdm(range(1, args.T_max + 1)):
    if done:
      state, done = env.reset(), False
      states, actions, rewards, keys, values, hashes = [], [], [], [], [], []  # Store transition data in episodic buffers

    # Linearly anneal ε over set interval
    if T > args.epsilon_anneal_start and T <= args.epsilon_anneal_end:
      epsilon -= (args.epsilon_initial - args.epsilon_final) / (args.epsilon_anneal_end - args.epsilon_anneal_start)
      agent.set_epsilon(epsilon)
    
    # Append transition data to episodic buffers (1/2)
    states.append(state.cpu().numpy())
    hashes.append(env.get_state_hash())  # Use environment state hash function
    
    # Choose an action according to the policy
    action, key, value = agent.act(state, return_key_value=True)
    state, reward, done = env.step(action)  # Step

    # Append transition data to episodic buffers (2/2); note that original NEC implementation does not recalculate keys/values at the end of the episode
    actions.append(action)
    rewards.append(reward)
    keys.append(key.cpu().numpy())
    values.append(value)

    # Calculate returns at episode to batch memory updates
    if done:
      episode_T = len(rewards)
      returns, multistep_returns = [None] * episode_T, [None] * episode_T
      returns.append(0)
      for i in range(episode_T - 1, -1, -1):  # Calculate return-to-go in reverse
        returns[i] = rewards[i] + args.discount * returns[i + 1]
        if episode_T - i > args.episodic_multi_step:  # Calculate multi-step returns (originally only for NEC)
          multistep_returns[i] = returns[i] + args.discount ** args.episodic_multi_step * (values[i + args.episodic_multi_step] - returns[i + args.episodic_multi_step])
        else:  # Calculate Monte Carlo returns (originally only for MFEC)
          multistep_returns[i] = returns[i]
      states, actions, returns, keys, hashes = np.stack(states), np.asarray(actions, dtype=np.int64), np.asarray(multistep_returns, dtype=np.float32), np.stack(keys), np.stack(hashes)
      unique_actions, unique_action_reverse_idxs = np.unique(actions, return_inverse=True)  # Find every unique action taken and indices
      for i, a in enumerate(unique_actions):
        a_idxs = (unique_action_reverse_idxs == i).nonzero()[0]
        agent.update_memory_batch(a.item(), keys[a_idxs], returns[a_idxs][:, np.newaxis], hashes[a_idxs])  # Append transition to DND of action in batch
      if args.algorithm == 'NEC':
        mem.append_batch(states, actions, returns)  # Append transition to memory in batch

      # Save metrics
      metrics['train_steps'].append(T)
      metrics['train_episodes'].append(len(metrics['train_episodes']) + 1)
      metrics['train_rewards'].append(sum(rewards))
      torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

    # Train and test
    if T >= args.learn_start:
      if args.algorithm == 'NEC' and T % args.replay_frequency == 0:
        agent.learn(mem)  # Train network
      
      if T % args.evaluation_interval == 0:
        agent.eval()  # Set agent to evaluation mode
        test_rewards, test_Qs = test(args, T, agent, val_mem, results_dir)  # Test
        log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | Avg. reward: ' + str(sum(test_rewards) / args.evaluation_episodes) + ' | Avg. Q: ' + str(sum(test_Qs) / args.evaluation_size))
        agent.train()  # Set agent back to training mode
        metrics['test_steps'].append(T)
        metrics['test_rewards'].append(test_rewards)
        metrics['test_Qs'].append(test_Qs)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

env.close()
