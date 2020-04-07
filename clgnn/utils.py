import torch
import os
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import h5py
import math
from collections import defaultdict
import random


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(args, dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    if dataset_str == 'friendster':
        dataset = h5py.File("../data/friendster/friendster_25K.h5")
        adj_list = dataset["adjacency"][:]  # Adjacency list
        if args.model_choice == 'gs' or args.model_choice == 'gs_rand':
            graph = defaultdict(set)
            for i in range(len(adj_list)):
                for j in adj_list[i]:
                    graph[i].add(j)
                    graph[j].add(i)
            adj = graph
        else:
            adj = torch.zeros((len(adj_list), len(adj_list)))
            for i in range(len(adj_list)):
                for j in adj_list[i]:
                    adj[i, j] = 1
        features = dataset["features"][:]  # Feature matrix
        labels = np.load("../data/friendster/age_labels.npy", allow_pickle=True)
        features = features[:, 1:]
        mu = features.mean(0)
        sigma = features.std(0)
        sigma[sigma == 0] = 1
        features = (features - mu) / sigma
        features = torch.FloatTensor(features)
    elif dataset_str == 'fb':
        edge_list = np.load("../data/fb.edgelist.npy")
        labels = np.load("../data/fb.labels.npy")
        adj = torch.zeros((len(labels)), len(labels))
        for (i,j) in edge_list:
            adj[i, j] = 1
            adj[j, i] = 1
        features = np.load("../data/fb.attrs.npy")
        features = torch.FloatTensor(features)
        # print(labels)
    elif dataset_str == 'protein':
        edge_list = np.loadtxt("../data/proteins/edges_protein.txt")
        labels = np.loadtxt("../data/proteins/labels_protein.txt")
        features = np.load("../data/proteins/features_protein.npy")
        mu = features.mean(0)
        sigma = features.std(0)
        sigma[sigma == 0] = 1
        features = (features - mu) / sigma
        features = torch.FloatTensor(features)
        if args.model_choice == 'gs_rand':
            graph = defaultdict(set)
            for (i, j) in edge_list:
                graph[i].add(j)
                graph[j].add(i)
            graph[8890].add(8890)
            graph[11963].add(11963)
            adj = graph

        else:
            adj = torch.zeros((len(labels)), len(labels))
            for (i, j) in edge_list:
                adj[int(i), int(j)] = 1
                adj[int(j), int(i)] = 1

    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        labels = torch.LongTensor(labels)
        labels = torch.max(labels, 1)[1]
        features = normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))
    if not args.model_choice == 'gs' and not args.model_choice == 'gs_rand':
        # print(adj)
        adj = sp.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    elif args.dataset != 'friendster' and args.dataset != 'protein':
        adj = sp.coo_matrix(adj)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = np.array(adj.todense())
        graph = defaultdict(set)
        edges = set()
        for i, v in enumerate(adj):
            for j, u in enumerate(v):
                if u != 0 and frozenset([i, j]) not in edges:
                    edges.add(frozenset([i, j]))
                    graph[i].add(j)
                    graph[j].add(i)
        adj = graph
    labels = torch.LongTensor(labels)
    if args.dataset != 'protein':
        idx_train_full = torch.from_numpy(
            np.loadtxt('../data/idx_train_' + args.dataset + '_' + str(args.trial) + '.txt')).long()
        idx_test = torch.from_numpy(
            np.loadtxt('../data/idx_test_' + args.dataset + '_' + str(args.trial) + '.txt')).long()
        idx_val_full = torch.from_numpy(
            np.loadtxt('../data/idx_val_' + args.dataset + '_' + str(args.trial) + '.txt')).long()

    return adj, features, labels, idx_train_full, idx_val_full, idx_test



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels, balanced=False):
    preds = output.max(1)[1].type_as(labels)
    if balanced:
        weights = torch.bincount(labels).float()
        weights = weights.max() / weights
        weights = weights[labels]
        # Compute (balanced) accuracy
        acc = preds.eq(labels).float().mul(weights).sum()/ weights.sum()
    else:
        correct = preds.eq(labels).double()
        correct = correct.sum()
        acc = correct / len(labels)
    return acc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



