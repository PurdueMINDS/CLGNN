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
from models.gcn import GCN, GCN_RAND
from models.stronger_gcn import snowball, snowball_rand
from models.graphsage import SupervisedGraphSage, SupervisedGraphSage_Rand
from models.encoders import Encoder
from models.aggregators import MeanAggregator
import networkx as nx
import random, math

criterion = nn.CrossEntropyLoss()

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--device', type=int, default=0, help='GPU device number')
parser.add_argument('--trial', type=int, choices=[0, 1], default=0, help='select data split')
parser.add_argument('--model_choice', choices=['gcn_rand', 'tk_rand', 'gs_rand'], default='gcn_rand')
parser.add_argument('--dataset', choices=['cora', 'pubmed', 'friendster', 'protein', 'fb'], default='cora')

# early stopping
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=100,
                    help='Number of epochs for val_loss to stop dropping')

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
parser.add_argument('--layers', type=int, default=10, help='Number of hidden layers, i.e. network depth')
parser.add_argument('--activation', type=str, default="tanh", help='Activation Function')

# random features

parser.add_argument('--predicted', action='store_true', default=True,
                    help='use predicted label (y_hat) in GNN input')
parser.add_argument('--prob', action='store_true', default=False,
                    help='use output probability instead of sampling')
parser.add_argument('--iterations', type=int, default=1,
                    help='number of iterations')
parser.add_argument('--nround', type=int, default=5,
                    help='number of model re-trainings to test variance')
parser.add_argument('--rand_number', type=int, default=10, help="Number of MC samples")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



def train(model, optimizer, epoch, features):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    class_balanced = False
    idx_train_new = idx_train
    if 'gs' not in args.model_choice:
        output = model.forward(features)
    if args.dataset == 'friendster':
        # compute weight for each class (balanced cross entropy loss)
        train_weights = torch.bincount(labels[idx_train_new]).float()
        train_weights = train_weights.max() / train_weights
        if args.cuda:
            train_weights = torch.cat((train_weights, torch.zeros(1).to(device)), dim=0)
        else:
            train_weights = torch.cat((train_weights, torch.zeros(1)), dim=0)
        if 'gs' in args.model_choice:
            result_train = model(idx_train_new)
            loss_train = F.nll_loss(result_train, labels[idx_train_new], weight=train_weights)
        else:
            loss_train = F.nll_loss(output[idx_train_new], labels[idx_train_new], weight=train_weights)
        class_balanced = True
    else:
        if 'gs' not in args.model_choice:
            loss_train = F.nll_loss(output[idx_train_new], labels[idx_train_new])
        else:
            result_train = model(idx_train_new)
            loss_train = F.nll_loss(result_train, labels[idx_train_new])

    loss_train.backward()
    optimizer.step()

    model.eval()
    if 'gs' in args.model_choice:
        output = model(torch.LongTensor(list(range(len(features)))))
    else:
        output = model(features)
    acc_train = accuracy(output[idx_train_new], labels[idx_train_new], balanced=class_balanced)
    acc_val = accuracy(output[idx_val], labels[idx_val], balanced=class_balanced)
    acc_test = accuracy(output[idx_test], labels[idx_test], balanced=class_balanced)


    return acc_train, acc_val, acc_test, output.data.exp()


def test(model, feature_curr, idx_test_):
    model.eval()
    output = model(feature_curr, adj)
    loss_test = F.nll_loss(output[idx_test_], labels[idx_test_])
    acc_test = accuracy(output[idx_test_], labels[idx_test_])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    return output[idx_test], acc_test.item()


def pre_train():
    patience = args.patience
    epochs = args.epochs
    if args.model_choice == 'gcn_rand':
        model = GCN_RAND(nfeat=features.shape[1],
                         nhid=args.hidden,
                         nclass=labels.max().item() + 1,
                         nnode=labels.shape[0],
                         labels=labels,
                         args=args,
                         predictions=None,
                         adj=adj)
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
    elif args.model_choice == 'tk_rand':
        model = snowball_rand(args=args, adj=adj, labels=labels, nnode=labels.shape[0], nfeat=features.shape[1],
                              nlayers=args.layers, nhid=args.hidden,
                              nclass=labels.max().item() + 1, dropout=args.dropout,
                              activation=activation, predictions=None)
        class_optimizer = eval('optim.%s' % args.optimizer)
        args.lr = 0.05
        optimizer = class_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.model_choice == 'gs_rand':
        model = SupervisedGraphSage_Rand(labels.max().item() + 1, args, features, adj, None, 25, 12, device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.cuda:
        model.to(device)
    t_total = time.time()

    early_stopping = EarlyStopping(patience=patience, verbose=False)
    best_val, best_test, best_prob = 0, 0, None
    for epoch_num in range(epochs):
        acc_train, acc_val, acc_test, last_output_prob = train(model, optimizer, epoch_num, features)
        if acc_val >= best_val:
            best_val, best_test = acc_val, acc_test
            best_prob = last_output_prob
        early_stopping(-acc_val, model)
        if early_stopping.early_stop:
            # print("Early stopping")
            break
    print("GNN: Val acc = {:.4f}, Test acc = {:.4f} ".format(best_val, best_test))
    return best_prob, best_test


def train_k_rounds(n_round, features):
    base_accs = torch.zeros(n_round)
    all_accs = torch.zeros(n_round)
    create_new_model = True
    for k in range(n_round):
        predicted = args.predicted
        args.predicted = False
        base_prob, base_acc = pre_train()
        base_accs[k] = base_acc
        args.predicted = predicted
        if args.model_choice == 'gcn_rand' and create_new_model:
            # print("creating new model...")
            model = GCN_RAND(nfeat=features.shape[1],
                             nhid=args.hidden,
                             nclass=labels.max().item() + 1,
                             nnode=labels.shape[0],
                             labels=labels,
                             args=args,
                             predictions=base_prob,
                             adj=adj)
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_choice == 'tk_rand' and create_new_model:
            model = snowball_rand(args=args, adj=adj, labels=labels, nnode=labels.shape[0], nfeat=features.shape[1],
                                  nlayers=args.layers, nhid=args.hidden,
                                  nclass=labels.max().item() + 1, dropout=args.dropout,
                                  activation=activation, predictions=base_prob)
            class_optimizer = eval('optim.%s' % args.optimizer)
            optimizer = class_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.model_choice == 'gs_rand'and create_new_model:
            model = SupervisedGraphSage_Rand(labels.max().item() + 1, args, features, adj, base_prob, 25, 12, device)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        # Train model
        if args.cuda:
            model.to(device)

        best_val, best_test, best_prob = 0, 0, None
        for m in range(args.iterations):
            t_total = time.time()
            if m > 0:
                # print("update predicted probability")
                model.update_prob(best_prob)
            early_stopping = EarlyStopping(patience=args.patience, verbose=False)
            for epoch_num in range(args.epochs):
                acc_train, acc_val, acc_test, last_output_prob = train(model, optimizer, epoch_num, features)
                early_stopping(-acc_val, model)
                if early_stopping.early_stop:
                    # print("Early stopping")
                    break
                if acc_val > best_val:
                    best_val, best_test = acc_val, acc_test
                    best_prob = last_output_prob

            print("CL-GNN: Val acc = {:.4f}, Test acc = {:.4f}".format(best_val, best_test))

        # Testing
        all_accs[k] = best_test
    print('base GNN: {:.4f} ({:.4f})'.format(torch.mean(base_accs).item(), torch.std(base_accs).item()))
    print('CL-GNN: {:.4f} ({:.4f})'.format(torch.mean(all_accs).item(), torch.std(all_accs).item()))
    del model, optimizer
    torch.cuda.empty_cache()

    return all_accs



# Load data

adj, features, labels, idx_train_full, idx_val, idx_test = load_data(args, args.dataset)





if args.cuda:
    if 'gs' not in args.model_choice:
        adj = adj.to(device)
        features = features.to(device)
    labels = labels.to(device)
    idx_train_full = idx_train_full.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

n_round = args.nround



if args.activation == 'identity':
    activation = lambda X: X
elif args.activation == 'tanh':
    activation = torch.tanh
else:
    activation = eval("F.%s" % args.activation)




final_res = []
if args.dataset == 'cora':
    percents = [0.625, 0.75, 1.0]
elif args.dataset == 'pubmed':
    percents = [0.5, 0.625, 1.0]
else:
    percents = [1.0]

for percent in percents:
    idx_train = idx_train_full[-int(idx_train_full.shape[0] * percent):]
    print("Label rate: {:.2f}%".format(100*len(idx_train)*1./len(features)))

    train_k_rounds(n_round, features)

