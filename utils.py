import faiss
import numpy as np
import torch


def knn_search(queries, data, k, return_neighbours=False, res=None):
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
