from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy, EarlyStopping
from models.gcn import GCN, GCN_RAND, GCN_RAND_LABELED
from models.stronger_gcn import snowball, snowball_rand, snowball_rand_labeled
from models.graphsage import SupervisedGraphSage, SupervisedGraphSage_Labeled
from models.encoders import Encoder
from models.aggregators import MeanAggregator
import networkx as nx
import random, math

criterion = nn.CrossEntropyLoss()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--trial', type=int, choices=[2], default=2, help='select data split')
parser.add_argument('--device', type=int, default=0, help='GPU device number')
parser.add_argument('--batch', type=int, default=100, help='Number of batches per epoch')
parser.add_argument('--iterations', type=int, default=10, help='number of iterations')
parser.add_argument('--nround', type=int, default=5, help='number of model re-trainings to test variance')
parser.add_argument('--model_choice', choices=['gcn_rand', 'tk_rand', 'gs_rand'], default='gcn_rand')
parser.add_argument('--dataset', choices=['cora', 'pubmed', 'friendster', 'protein', 'fb'], default='cora')

# early stopping
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100,
                    help='Number of epochs for val_loss to stop dropping')
parser.add_argument('--validation', action='store_true', default=True,
                    help='pick the best epoch based on validation accuracy')

# optimization parameters
parser.add_argument('--optimizer', type=str, default='RMSprop', help='Optimizer')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--layers', type=int, default=5, help='Number of hidden layers, i.e. network depth')
parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')

# random features
parser.add_argument('--rand_number', type=int, default=10, help="Number of random features")
parser.add_argument('--baseline', action='store_true', default=False,
                    help='use base GNN model')
parser.add_argument('--pred', action='store_true', default=True,
                    help='use predicted label')
parser.add_argument('--sample', action='store_true', default=True,
                    help='use sampling')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
if not args.baseline:
    # train each sample a few steps for applying random masks
    if 'gcn' not in args.model_choice:
        args.epochs = args.patience = 1
    else:
        args.epochs = args.patience = 3


# Load data

adj, features, labels, idx_train_full, idx_val, idx_test = load_data(args, args.dataset)


nnodes = labels.shape[0]


if args.cuda:
    if 'gs' not in args.model_choice:
        adj = adj.to(device)
        features = features.to(device)
    labels = labels.to(device)
    idx_train_full = idx_train_full.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)




def train(model, optimizer, epoch, features, idx_opt_new, idx_true_new, pre_train=True,
              train_prediction=None, test_prediction=None):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    class_balanced = False
    idx_opt_new = idx_opt_new.to(device)
    idx_true_new = idx_true_new.to(device)
    if 'gs' not in args.model_choice:
        output = model.forward(features, idx_true_new, pre_train=pre_train, baseline=args.baseline, predictions=train_prediction)
    else:
        output = model(torch.LongTensor(list(range(len(features)))), features, idx_true_new, pre_train=pre_train,
                       baseline=args.baseline, predictions=train_prediction)
    if args.dataset == 'friendster':
        # compute weight for each class (balanced cross entropy loss)
        train_weights = torch.bincount(labels[idx_opt_new]).float()
        train_weights = train_weights.max() / train_weights
        if args.cuda:
            train_weights = torch.cat((train_weights, torch.zeros(1).to(device)), dim=0)
        else:
            train_weights = torch.cat((train_weights, torch.zeros(1)), dim=0)
        loss_train = F.nll_loss(output[idx_opt_new], labels[idx_opt_new], weight=train_weights)
        class_balanced = True
    else:
        loss_train = F.nll_loss(output[idx_opt_new], labels[idx_opt_new])

    loss_train.backward()
    optimizer.step()

    model.eval()
    if 'gs' not in args.model_choice:
        output_infer = model(features, idx_obs_infer, pre_train=pre_train, baseline=args.baseline, predictions=test_prediction)
    else:
        output_infer = model(torch.LongTensor(list(range(len(features)))), features, idx_obs_infer, pre_train=pre_train,
                             baseline=args.baseline, predictions=test_prediction)
    acc_train = accuracy(output_infer[idx_opt_new], labels[idx_opt_new], balanced=class_balanced)
    acc_val = accuracy(output_infer[idx_val], labels[idx_val], balanced=class_balanced)
    acc_test = accuracy(output_infer[idx_test], labels[idx_test], balanced=class_balanced)


    return acc_train, acc_val, acc_test, output.data.exp(), output_infer.data.exp()


def test(model, feature_curr, idx_test_):
    model.eval()
    output = model(feature_curr, adj)
    loss_test = F.nll_loss(output[idx_test_], labels[idx_test_])
    acc_test = accuracy(output[idx_test_], labels[idx_test_])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output[idx_test], acc_test.item()


def train_k_rounds(n_round, idx_obs_all, idx_train_all):
    if args.model_choice == 'gcn_rand':
        # print("creating new model...")
        model = GCN_RAND_LABELED(nfeat=features.shape[1],
                                 nhid=args.hidden,
                                 nclass=labels.max().item() + 1,
                                 nnode=labels.shape[0],
                                 labels=labels,
                                 args=args,
                                 adj=adj)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    elif args.model_choice == 'tk_rand':
        model = snowball_rand_labeled(args=args, adj=adj, labels=labels, nnode=labels.shape[0], nfeat=features.shape[1],
                                      nlayers=args.layers, nhid=args.hidden,
                                      nclass=labels.max().item() + 1, dropout=args.dropout, activation=activation)
        class_optimizer = eval('optim.%s' % args.optimizer)
        optimizer = class_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model_choice == 'gs_rand':
        model = SupervisedGraphSage_Labeled(nfeat=features.shape[1],
                                            nclass=labels.max().item() + 1,
                                            nnode=labels.shape[0],
                                            labels=labels, args=args, adj=adj)
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    all_accs = torch.zeros(n_round)
    all_accs_val = torch.zeros(n_round)
    train_prediction = test_prediction = None

    if args.cuda:
        model.to(device)

    for k in range(n_round):
        t_total = time.time()
        if k > 0 and not args.baseline:
            train_prediction = best_train_output
            test_prediction = best_test_output
        not_use_pred = (k == 0) or (not args.pred)
        if k > 0:
            # random permute batch
            rr = np.random.permutation(len(idx_obs_all))
            idx_obs_all = idx_obs_all[rr]
            idx_train_all = idx_train_all[rr]
        best_val, best_test = 0, 0
        for idx_obs, idx_train in zip(idx_obs_all, idx_train_all):
            early_stopping = EarlyStopping(patience=args.patience, verbose=False)
            for epoch_num in range(args.epochs):
                acc_train, acc_val, acc_test, train_output_prob, test_output_prob = train(model, optimizer, epoch_num,
                                                                                           features, idx_train, idx_obs,
                                                                                           pre_train=not_use_pred,
                                                                                           train_prediction=train_prediction,
                                                                                           test_prediction=test_prediction)
                if acc_val > best_val:
                    best_val, best_test = acc_val, acc_test
                    best_train_output, best_test_output = train_output_prob, test_output_prob
                early_stopping(-acc_val, model)
                if early_stopping.early_stop:
                    break

        print('Iteration: {:02d}'.format(k),
                'acc_train: {:.4f}'.format(acc_train.item()),
                  'acc_val: {:.4f}'.format(best_val.item()),
                  'acc_test: {:.4f}'.format(best_test.item()),
              'time: {:.4f}s'.format(time.time() - t_total))

        # Testing
        if args.validation:
            all_accs[k], all_accs_val[k] = best_test, best_val
        else:
            all_accs[k], all_accs_val[k] = acc_test, acc_val
        print(all_accs)
    del model, optimizer
    torch.cuda.empty_cache()
    return all_accs, all_accs_val




if args.activation == 'identity':
    activation = lambda X: X
elif args.activation == 'tanh':
    activation = torch.tanh
else:
    activation = eval("F.%s" % args.activation)


if args.dataset == 'cora':
    percents = [0.25, 0.3, 0.4]
elif args.dataset == 'pubmed':
    percents = [0.2, 0.25, 0.4]
elif args.dataset == 'fb':
    percents = [0.2]
elif args.dataset == 'protein':
    percents = [1.0]
else:
    percents = [0.3]

n_round = args.nround
batch_size = args.batch

if args.dataset == 'protein':
    random.seed(args.trial)
    random.shuffle(idx_test)
    test_len = int(idx_test.shape[0] * 0.5)
    idx_obs_infer = idx_test[:test_len]
    idx_test = idx_test[test_len:]


if args.baseline:
    for percent in percents:
        final_res = []
        final_res_val = []
        train_len = int(idx_train_full.shape[0] * percent)
        idx_train = idx_train_full[-train_len:]
        print("Label rate: {:.2f}%".format(100 * len(idx_train) * 1. / len(features)))
        idx_obs_infer = None
        idx_opt = [idx_train]
        labeled_idx = [torch.from_numpy(np.array([]))]
        for i in range(n_round):
            acc, acc_val = train_k_rounds(1, labeled_idx, idx_opt)
            final_res.append(acc.numpy())
            final_res_val.append(acc_val.numpy())
        final_res, final_res_val = np.array(final_res), np.array(final_res_val)
        print("GNN: val: {:.4f} ({:.4f}), test: {:.4f} ({:.4f})\n".format(np.mean(final_res_val[:, 0]),
                                                                                np.std(final_res_val[:, 0]),
                                                                                np.mean(final_res[:, 0]),
                                                                                np.std(final_res[:, 0])))
    exit()

for opt_ratio in [0.7]:
    n_iterations = args.iterations
    for percent in percents:
        final_res = []
        final_res_val = []
        train_len = int(idx_train_full.shape[0] * percent)
        idx_train_obs_train = idx_train_full[-train_len:]
        print("Label rate: {:.2f}%".format(100 * len(idx_train_obs_train) * 1. / len(features)))
        if args.dataset != 'protein':
            idx_obs_infer = idx_train_full[:train_len]
        # ranomly split V_observe and V_train
        idx_obs_all, idx_train_all = [], []
        for k in range(batch_size):
            r = torch.randperm(idx_train_obs_train.shape[0])
            idx_train_obs_train = idx_train_obs_train[r]
            # idx_obs_all: use true label
            # idx_train_all: use as opt. target
            idx_obs_all.append(idx_train_obs_train[int(idx_train_obs_train.shape[0] * opt_ratio):])
            idx_train_all.append(idx_train_obs_train[:int(idx_train_obs_train.shape[0] * opt_ratio)])
        idx_obs_all = torch.stack(idx_obs_all, dim=0)
        idx_train_all = torch.stack(idx_train_all, dim=0)

        for i in range(n_round):
            acc, acc_val = train_k_rounds(n_iterations, idx_obs_all=idx_obs_all, idx_train_all=idx_train_all)
            final_res.append(acc.numpy())
            final_res_val.append(acc_val.numpy())
        final_res, final_res_val = np.array(final_res), np.array(final_res_val)
        best_val, best_test, best_test_std, best_round = 0, 0, 0, 0
        for k in range(n_iterations):
            k_val, k_test = np.mean(final_res_val[:, k]), np.mean(final_res[:, k])
            if k_val > best_val:
                best_val, best_test = k_val, k_test
                best_test_std = np.std(final_res_val[:, k])
                best_round = k
        print(
            "Best round: {} with val acc: {:.4f}, test acc: {:.4f}({:.4f})\n".format(best_round, best_val, best_test,
                                                                                     best_test_std))

