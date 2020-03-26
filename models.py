import faiss
import numpy as np
from sklearn import random_projection
import torch
from torch import nn
from torch.nn import functional as F


# Set up GPU scratch space for FAISS
def _setup_faiss_gpu_resources(device):
  if device == torch.device('cuda'):
    res = faiss.StandardGpuResources()
    res.setDefaultNullStreamAllDevices()
    # res.setTempMemory(64 * 1024 * 1024)  # Do not decrease memory allocation for resource object (large amount needed for example for Atari)
    return res
  else:
    return None


# Mean kernel
def _mean_kernel(squared_l2_dists, opts=None):
    return torch.ones_like(squared_l2_dists)


# Kernel that interpolates between the mean for short distances and weighted inverse distance for large distances
def _mean_IDW_kernel(squared_l2_dists, opts):
    return 1 / (squared_l2_dists + opts['delta'])


# k-nearest neighbours search
def _knn_search(queries, data, k, return_neighbours=False, res=None):
  num_queries, dim = queries.shape
  if res is None:
    dists, idxs = np.empty((num_queries, k), dtype=np.float32), np.empty((num_queries, k), dtype=np.int64)
    heaps = faiss.float_maxheap_array_t()
    heaps.k, heaps.nh = k, num_queries
    heaps.val, heaps.ids = faiss.swig_ptr(dists), faiss.swig_ptr(idxs)
    faiss.knn_L2sqr(faiss.swig_ptr(queries), faiss.swig_ptr(data), dim, num_queries, data.shape[0], heaps)
  else:
    dists, idxs = torch.empty(num_queries, k, dtype=torch.float32, device=queries.device), torch.empty(num_queries, k, dtype=torch.int64, device=queries.device)
    faiss.bruteForceKnn(res, faiss.METRIC_L2, faiss.cast_integer_to_float_ptr(data.storage().data_ptr() + data.storage_offset() * 4), data.is_contiguous(), data.shape[0], faiss.cast_integer_to_float_ptr(queries.storage().data_ptr() + queries.storage_offset() * 4), queries.is_contiguous(), num_queries, dim, k, faiss.cast_integer_to_float_ptr(dists.storage().data_ptr() + dists.storage_offset() * 4), faiss.cast_integer_to_long_ptr(idxs.storage().data_ptr() + idxs.storage_offset() * 8))
  if return_neighbours:
    neighbours = data[idxs.reshape(-1)].reshape(-1, k, dim)
    return dists, idxs, neighbours
  else:
    return dists, idxs


# Dictionary-based memory (assumes key-value associations do not change)
class StaticDictionary(nn.Module):
  def __init__(self, args, hash_size, faiss_gpu_resources=None):
    super().__init__()
    self.key_size = args.key_size
    self.faiss_gpu_resources = faiss_gpu_resources
    self.num_neighbours = args.num_neighbours
    if args.kernel == 'mean':
      self.kernel = _mean_kernel
    elif args.kernel == 'mean_IDW':
      self.kernel = _mean_IDW_kernel
    self.kernel_opts = {'delta': args.kernel_delta}
    
    self.keys = 1e6 * np.ones((args.dictionary_capacity, args.key_size), dtype=np.float32)  # Add initial keys with very large magnitude values (infinity results in kNN returning -1 as indices)
    if self.faiss_gpu_resources is not None:
      self.keys = torch.from_numpy(self.keys).to(device=args.device)
    self.values = np.zeros((args.dictionary_capacity, 1), dtype=np.float32)
    self.hashes = 1e6 * np.ones((args.dictionary_capacity, hash_size), dtype=np.float32)  # Assumes hash of 1e6 will never appear TODO: Replace with an actual dictionary?
    self.last_access = np.linspace(args.dictionary_capacity, 1, args.dictionary_capacity, dtype=np.int32)

  # Lookup function
  def forward(self, key):
    # Perform kNN search TODO: Do we do hash check here or assume kNN is fine since key-value associations don't change?
    output = torch.zeros(key.size(0), 1, device=key.device)
    dists, idxs, neighbours = _knn_search(key.detach().cpu().numpy() if self.faiss_gpu_resources is None else key.detach(), self.keys, self.num_neighbours, return_neighbours=True, res=self.faiss_gpu_resources)  # Return (squared) L2 distances and indices of nearest neighbours
    dists, idxs = (dists, idxs) if self.faiss_gpu_resources is None else (dists.cpu().numpy(), idxs.cpu().numpy())
    match_idxs, non_match_idxs = np.nonzero(dists[:, 0] == 0)[0], np.nonzero(dists[:, 0])[0]  # Detect exact matches (based on first returned distance) and non-matches
    output[match_idxs] = torch.from_numpy(self.values[idxs[match_idxs, 0]]).to(device=key.device)  # Use stored return for exact match
    
    # For non-matches, use (possibly weighted) average return over k nearest neighbours
    idxs_non_match_idxs = idxs[non_match_idxs]
    values = self.values[idxs_non_match_idxs.reshape(-1)].reshape((idxs_non_match_idxs.shape[0], self.num_neighbours, 1))  # Retrieve values
    weights = self.kernel(torch.from_numpy(dists[non_match_idxs]).to(device=key.device), self.kernel_opts)  # Apply kernel function
    weights /= weights.sum(dim=1, keepdim=True)  # Normalise weights
    output[non_match_idxs] = torch.sum(weights.unsqueeze(dim=2) * torch.from_numpy(values).to(device=key.device), dim=1)

    # Update last access (updated for all lookups: acting, return calculation and training)
    self.last_access += 1  # Increment last access for all items
    self.last_access[idxs.reshape(-1)] = 0  
    return output

  # Updates a batch of key-value pairs
  def update_batch(self, keys, values, hashes):
    # Test for matching states in batch
    sorted_idxs = np.argsort(values, axis=0)[::-1][:, 0]  # Sort values in descending order (max value first); equivalent to purely online update
    keys, values, hashes = keys[sorted_idxs], values[sorted_idxs], hashes[sorted_idxs]  # Rearrange keys, values and hashes in this order
    hashes, unique_indices = np.unique(hashes, axis=0, return_index=True)  # Retrieve unique hashes and indices of first occurences in array
    keys, values = keys[unique_indices], values[unique_indices]  # Extract corresponding keys and values

    # Perform hash check for exact matches
    dists, idxs = _knn_search(hashes, self.hashes, 1)  # TODO: Replace kNN search with real hash check
    dists, idxs = dists[:, 0], idxs[:, 0]
    match_idxs, non_match_idxs = np.nonzero(dists == 0)[0], np.nonzero(dists)[0]
    num_matches, num_non_matches = len(match_idxs), len(non_match_idxs)
    # Update last access (updated for all lookups: acting, return calculation and training)
    self.last_access += 1  # Increment last access for all items

    # Update exact match with best return (risk seeking)
    if num_matches > 0:
      idxs_match_idxs = idxs[match_idxs]
      self.values[idxs_match_idxs] = np.maximum(self.values[idxs_match_idxs], values[match_idxs])
      self.last_access[idxs_match_idxs] = 0
    
    # Otherwise add new states and Monte Carlo returns, replacing least recently updated entries
    if num_non_matches > 0:
      lru_idxs = np.argpartition(self.last_access, -num_non_matches)[-num_non_matches:]  # Find top-k LRU items
      self.keys[lru_idxs] = keys[non_match_idxs] if self.faiss_gpu_resources is None else torch.from_numpy(keys[non_match_idxs]).to(device=self.keys.device)
      self.values[lru_idxs] =  values[non_match_idxs]
      self.hashes[lru_idxs] = hashes[non_match_idxs]
      self.last_access[lru_idxs] = 0


class MFEC(nn.Module):
  def __init__(self, args, observation_shape, action_space, hash_size):
    super().__init__()
    self.register_buffer('projection', torch.tensor(random_projection.GaussianRandomProjection(n_components=args.key_size)._make_random_matrix(args.key_size, np.prod(observation_shape)), dtype=torch.float32).t())  # TODO: Check if DeepMind corrected variance for key size
    faiss_gpu_resources = _setup_faiss_gpu_resources(args.device)
    self.memories = [StaticDictionary(args, hash_size, faiss_gpu_resources) for _ in range(action_space)]

  def forward(self, observation):
    keys = observation.view(observation.size(0), -1)
    keys = torch.matmul(keys, self.projection) if hasattr(self, 'projection') else keys
    q_values = torch.cat([memory(keys) for memory in self.memories], dim=1)
    return q_values, keys


# Differentiable neural dictionary
class DND(StaticDictionary):
  def __init__(self, args, hash_size, faiss_gpu_resources=None):
    super().__init__(args, hash_size, faiss_gpu_resources=faiss_gpu_resources)
    self.key_size = args.key_size
    self.alpha = args.dictionary_learning_rate
    # RMSprop components
    self.rmsprop_learning_rate, self.rmsprop_decay, self.rmsprop_epsilon = args.learning_rate, args.rmsprop_decay, args.rmsprop_epsilon
    self.rmsprop_keys_square_avg, self.rmsprop_values_square_avg = torch.zeros(args.dictionary_capacity, args.key_size), torch.zeros(args.dictionary_capacity, 1)

  # Lookup function
  def forward(self, key, learning=False):
    # Perform kNN search
    if learning:
      _, idxs, neighbours = _knn_search(key.detach().cpu().numpy() if self.faiss_gpu_resources is None else key.detach(), self.keys, self.num_neighbours, return_neighbours=True, res=self.faiss_gpu_resources)  # Retrieve actual neighbours
      neighbours = torch.tensor(neighbours, requires_grad=True).to(device=key.device) if self.faiss_gpu_resources is None else neighbours.requires_grad_(True)
      dists = (key.unsqueeze(dim=1) - neighbours).pow(2).sum(dim=2)  # Recalculate (squared) L2 distance for differentiation
      # TODO: Check if exact match causes gradient problems
    else:
      dists, idxs = _knn_search(key.detach().cpu().numpy() if self.faiss_gpu_resources is None else key.detach(), self.keys, self.num_neighbours, res=self.faiss_gpu_resources)
      dists = torch.tensor(dists).to(device=key.device) if self.faiss_gpu_resources is None else dists
    idxs = idxs if self.faiss_gpu_resources is None else idxs.cpu()

    # Use weighted average return over k nearest neighbours
    weights = self.kernel(dists, self.kernel_opts)  # Apply kernel function
    weights /= weights.sum(dim=1, keepdim=True)  # Normalise weights
    values = self.values[idxs.reshape(-1)].reshape((idxs.shape[0], self.num_neighbours, 1))  # Retrieve values
    values = torch.tensor(values, requires_grad=True).to(device=key.device)
    output = torch.sum(weights.unsqueeze(dim=2) * values, dim=1)

    # Update last access (updated for all lookups: acting, return calculation and training)
    self.last_access += 1  # Increment last access for all items
    self.last_access[idxs.reshape(-1)] = 0  
    if learning:
      return output, neighbours, values, idxs
    else:
      return output

  # Updates a batch of key-value pairs
  def update_batch(self, keys, values, hashes):
    # Test for matching states in batch
    sorted_idxs = np.argsort(values, axis=0)[::-1][:, 0]  # Sort values in descending order (max value first) TODO: Is this the way it should be done for NEC, or average?
    keys, values, hashes = keys[sorted_idxs], values[sorted_idxs], hashes[sorted_idxs]  # Rearrange keys, values and hashes in this order
    hashes, unique_indices = np.unique(hashes, axis=0, return_index=True)  # Retrieve unique hashes and indices of first occurences in array
    keys, values = keys[unique_indices], values[unique_indices]  # Extract corresponding keys and values

    # Perform hash check for exact matches
    dists, idxs = _knn_search(hashes, self.hashes, 1)  # TODO: Replace kNN search with real hash check
    dists, idxs = dists[:, 0], idxs[:, 0]
    match_idxs, non_match_idxs = np.nonzero(dists == 0)[0], np.nonzero(dists)[0]
    num_matches, num_non_matches = len(match_idxs), len(non_match_idxs)
    # Update last access (updated for all lookups: acting, return calculation and training)
    self.last_access += 1  # Increment last access for all items

    # Update exact match with Q-learning
    if num_matches > 0:
      idxs_match_idxs = idxs[match_idxs]
      self.keys[idxs_match_idxs] = keys[match_idxs] if self.faiss_gpu_resources is None else torch.from_numpy(keys[match_idxs]).to(device=self.keys.device) # Update keys (embedding may have changed)
      self.values[idxs_match_idxs] += self.alpha * (values[match_idxs] - self.values[idxs_match_idxs])
      # self.rmsprop_keys_square_avg[idxs_match_idxs], self.rmsprop_values_square_avg[idxs_match_idxs] = 0, 0  # TODO: Reset RMSprop stats here too?
      self.last_access[idxs_match_idxs] = 0
    
    # Otherwise add new states and n-step returns, replacing least recently updated entries
    if num_non_matches > 0:
      lru_idxs = np.argpartition(self.last_access, -num_non_matches)[-num_non_matches:]  # Find top-k LRU items
      self.keys[lru_idxs] = keys[non_match_idxs] if self.faiss_gpu_resources is None else torch.from_numpy(keys[non_match_idxs]).to(device=self.keys.device)
      self.values[lru_idxs] =  values[non_match_idxs]
      self.hashes[lru_idxs] = hashes[non_match_idxs]
      self.last_access[lru_idxs] = 0
      self.rmsprop_keys_square_avg[lru_idxs], self.rmsprop_values_square_avg[lru_idxs] = 0, 0  # Reset RMSprop stats

  # Performs a sparse RMSprop update TODO: Add momentum option and gradient clipping?
  def gradient_update(self, keys, values, idxs):
    idxs, unique_idxs = np.unique(idxs.reshape(-1), return_index=True)  # Check for duplicates to remove
    keys, values = keys.reshape(-1, self.key_size)[unique_idxs], values.reshape(-1, 1)[unique_idxs]  # Remove duplicate keys and values
    if keys.grad is not None:
      grad = keys.grad.data
      square_avg = self.rmsprop_keys_square_avg[idxs]
      square_avg.mul_(self.rmsprop_decay).addcmul_(1 - self.rmsprop_decay, grad, grad)
      avg = square_avg.add(self.rmsprop_epsilon).sqrt_()
      keys.data.addcdiv_(-self.rmsprop_learning_rate, grad, avg)
      self.keys[idxs] = keys.detach().cpu().numpy()
      self.rmsprop_keys_square_avg[idxs] = square_avg
    if values.grad is not None:
      grad = values.grad.data
      square_avg = self.rmsprop_values_square_avg[idxs]
      square_avg.mul_(self.rmsprop_decay).addcmul_(1 - self.rmsprop_decay, grad, grad)
      avg = square_avg.add(self.rmsprop_epsilon).sqrt_()
      values.data.addcdiv_(-self.rmsprop_learning_rate, grad, avg)
      self.values[idxs] = values.detach().cpu().numpy()
      self.rmsprop_values_square_avg[idxs] = square_avg


class NEC(nn.Module):
  def __init__(self, args, observation_shape, action_space, hash_size):
    super().__init__()
    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc_keys = nn.Linear(3136, args.key_size)
    faiss_gpu_resources = _setup_faiss_gpu_resources(args.device)
    self.memories = [DND(args, hash_size, faiss_gpu_resources) for _ in range(action_space)]

  def forward(self, observation, learning=False):
    hidden = F.relu(self.conv1(observation))
    hidden = F.relu(self.conv2(hidden))
    hidden = F.relu(self.conv3(hidden))
    keys = self.fc_keys(hidden.view(-1, 3136))
    memory_output = [memory(keys, learning) for memory in self.memories]
    if learning:
      memory_output, neighbours, values, idxs = zip(*memory_output)
      return torch.cat(memory_output, dim=1), neighbours, values, idxs, keys  # Return Q-values, neighbours, values and keys
    else:
      return torch.cat(memory_output, dim=1), keys  # Return Q-values and keys
