import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from models.encoders import Encoder, Encoder_Labeled
from models.aggregators import MeanAggregator, MeanAggregator_Labeled

"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        scores = scores.t()
        # print(scores)
        scores = F.log_softmax(scores, dim=1)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze().cuda())



class SupervisedGraphSage_Rand(nn.Module):
    def __init__(self, num_classes, args, features, adj, predictions, hid1, hid2, device):
        super(SupervisedGraphSage_Rand, self).__init__()
        self.xent = nn.CrossEntropyLoss()
        self.features = features
        self.adj = adj
        self.hid1 = hid1
        self.hid2 = hid2
        self.device = device
        self.predictions = predictions
        self.device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")
        self.rand_nums = args.rand_number
        self.nclass = num_classes
        self.args = args
        self.weight = nn.Parameter(torch.FloatTensor(self.nclass, hid2))
        init.xavier_uniform_(self.weight)
        self.nfeat = self.features.shape[1]
        if self.predictions is not None:
            self.nfeat += self.nclass
        agg1 = MeanAggregator_Labeled(cuda=self.args.cuda, device=self.device)
        enc1 = Encoder_Labeled(self.nfeat, hid1, self.adj, agg1, gcn=True, cuda=self.args.cuda)
        agg2 = MeanAggregator_Labeled(cuda=self.args.cuda, device=self.device)
        enc2 = Encoder_Labeled(enc1.embed_dim, hid2, self.adj, agg2, base_model=enc1, gcn=True, cuda=self.args.cuda)
        enc1.num_samples = 3
        enc2.num_samples = 3
        if self.args.cuda:
            agg1 = agg1.to(self.device)
            enc1 = enc1.to(self.device)
            agg2 = agg2.to(self.device)
            enc2 = enc2.to(self.device)
        self.enc1 = enc1
        self.enc2 = enc2

    def forward(self, nodes):
        if self.predictions is not None:
            feats = np.array(
                    [np.random.choice(range(self.nclass), 1, replace=True, p=i) for i in self.predictions.cpu().numpy()])
            feats = torch.from_numpy(feats)
            feats = feats.squeeze()
            feats = self.pad_zeros(F.one_hot(feats).squeeze())
            feats = feats.unsqueeze(0)
            features_gs = torch.stack([self.features for i in range(1)], dim=0)
            if self.args.cuda:
                feats = feats.to(self.device)
                features_gs = features_gs.to(self.device)
            features_gs = torch.cat([features_gs, feats.float()], dim=2)
            features_gs = features_gs.squeeze()
            features_gs_emb = nn.Embedding(features_gs.shape[0], features_gs.shape[1])
            features_gs_emb.weight = nn.Parameter(features_gs.float(), requires_grad=False)
        else:
            features_gs_emb = nn.Embedding(self.features.shape[0], self.features.shape[1])
            features_gs_emb.weight = nn.Parameter(self.features.float(), requires_grad=False)
            if self.args.cuda:
                features_gs_emb = features_gs_emb.to(self.device)
                nodes = nodes.to(self.device)
        embeds = self.enc2(lambda nodes: self.enc1(features_gs_emb, nodes).t(), nodes)
        scores = self.weight.mm(embeds)
        scores = scores.t()
        scores = F.log_softmax(scores, dim=1)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze().cuda())

    def update_prob(self, prob):
        self.predictions = prob

    def pad_zeros(self, feat):
        feat_new = torch.zeros(feat.shape[0], self.nclass).to(self.device)
        feat_new[:, :feat.shape[1]] = feat
        return feat_new


class SupervisedGraphSage_Labeled(nn.Module):
    def __init__(self, nfeat, nclass, nnode, labels, args, adj):
        super(SupervisedGraphSage_Labeled, self).__init__()
        self.xent = nn.CrossEntropyLoss()
        if not args.baseline:
            nfeat += labels.max().item()+1
        self.nfeat = nfeat
        self.nclass = nclass
        self.labels = labels
        self.nnode = nnode
        self.args = args
        self.adj = adj
        self.device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")
        self.rand_nums = args.rand_number
        self.weight = nn.Parameter(torch.FloatTensor(nclass, 12))
        init.xavier_uniform_(self.weight)
        agg1 = MeanAggregator_Labeled(cuda=self.args.cuda, device=self.device)
        enc1 = Encoder_Labeled(self.nfeat, 25, self.adj, agg1, gcn=True, cuda=self.args.cuda)
        agg2 = MeanAggregator_Labeled(cuda=self.args.cuda, device=self.device)
        enc2 = Encoder_Labeled(enc1.embed_dim, 12, self.adj, agg2, base_model=enc1, gcn=True, cuda=self.args.cuda)
        enc1.num_samples = 3
        enc2.num_samples = 3
        if self.args.cuda:
            agg1 = agg1.to(self.device)
            enc1 = enc1.to(self.device)
            agg2 = agg2.to(self.device)
            enc2 = enc2.to(self.device)
        self.enc1 = enc1
        self.enc2 = enc2


    def forward(self, nodes, x, idx_labeled, pre_train=False, predictions=None, baseline=False):
        if baseline:
            features_gs_emb = nn.Embedding(x.shape[0], x.shape[1])
            features_gs_emb.weight = nn.Parameter(torch.FloatTensor(x), requires_grad=False)
            if self.args.cuda:
                features_gs_emb = features_gs_emb.to(self.device)
                nodes = nodes.to(self.device)
            embeds = self.enc2(lambda nodes: self.enc1(features_gs_emb, nodes).t(), nodes)
            scores = self.weight.mm(embeds)
            scores = scores.t()
            scores = F.log_softmax(scores, dim=1)
            return scores

        if pre_train:
            feats = torch.zeros(self.nnode, ).long().to(self.device)
            feats += self.nclass + 2
            feats[idx_labeled] = self.labels[idx_labeled]
            feats = F.one_hot(feats)[:, :self.nclass]
            feats = feats.unsqueeze(0)
        else:
            feats = predictions.unsqueeze(0)
            true_feat = F.one_hot(self.labels[idx_labeled])
            if true_feat.shape[1] < self.nclass:
                true_feat = self.pad_zeros(true_feat)
            feats[:, idx_labeled, :] = true_feat.float()
            feats = feats.squeeze()
            if self.args.sample:
                feats = np.array([np.random.choice(range(self.nclass), 1, replace=True, p=i) for i in predictions.cpu().numpy()])
                feats = torch.from_numpy(feats)
                feats[idx_labeled] = self.labels.cpu().unsqueeze(1)[idx_labeled]
                feats = feats.squeeze()
                feats = self.pad_zeros(F.one_hot(feats).squeeze())
                feats = feats.unsqueeze(0)
        if self.args.cuda:
            feats = feats.to(self.device)
            x = x.to(self.device)
        if pre_train or self.args.sample:
            rand_nums = 1
        else:
            rand_nums = self.rand_nums
        features_gs = torch.stack([x for i in range(rand_nums)], dim=0)
        # print(features_gs.shape, feats.shape)
        features_gs = torch.cat([features_gs, feats.float()], dim=2)
        features_gs = features_gs.squeeze()
        features_gs_emb = nn.Embedding(features_gs.shape[0], features_gs.shape[1])
        features_gs_emb.weight = nn.Parameter(features_gs.float(), requires_grad=False)
        embeds = self.enc2(lambda nodes: self.enc1(features_gs_emb, nodes).t(), nodes)
        scores = self.weight.mm(embeds)
        scores = scores.t()
        scores = F.log_softmax(scores, dim=1)
        return scores

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze().cuda())

    def pad_zeros(self, feat):
        feat_new = torch.zeros(feat.shape[0], self.nclass).to(self.device)
        feat_new[:, :feat.shape[1]] = feat
        return feat_new



