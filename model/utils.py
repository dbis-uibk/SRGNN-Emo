import numpy as np
import time
import datetime

from sklearn.model_selection import StratifiedKFold
import torch_geometric
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import pandas as pd
import itertools
import random
import copy
import scipy
import torch
import torch.nn.functional as F
import os


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def sparse2sparse(coo_matrix):
    v1 = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = coo_matrix.shape
    sparse_matrix = torch.sparse.LongTensor(i, v, torch.Size(shape))
    return sparse_matrix


def dense2sparse(matrix):
    a_ = scipy.sparse.coo_matrix(matrix)
    v1 = a_.data
    indices = np.vstack((a_.row, a_.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = a_.shape
    sparse_matrix = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_matrix


def k_fold(dataset, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=12345)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def init_seed(seed=None):
    if seed is None or seed == 0:
        seed = int(time.time() * 1000 // 1000)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch_geometric.seed.seed_everything(seed)


def normalize_edge_weights(edge_index, edge_weight, num_nodes=None):
    # Add self-loops to the adjacency matrix.
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)

    # Compute the sum of edge weights for each node.
    row, col = edge_index
    deg = torch.zeros(edge_weight.size(0), dtype=edge_weight.dtype, device=edge_weight.device)
    deg = deg.scatter_add_(0, row, edge_weight)

    # Divide each edge weight by the sum of edge weights of its source node.
    edge_weight = edge_weight / deg[row]

    return edge_index, edge_weight

