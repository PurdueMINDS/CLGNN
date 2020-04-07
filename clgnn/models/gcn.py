import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try:
    from models.layers import GraphConvolution, SpectralConv
except:
    from layers import GraphConvolution, SpectralConv




class GCN_RAND(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nnode, labels, args, predictions=None, adj=None):
        super(GCN_RAND, self).__init__()
        if args.predicted:
            nfeat += labels.max().item()+1
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = args.dropout
        self.fc = nn.Linear(nclass, nclass)
        self.nclass = nclass
        self.labels = labels
        self.rand_nums = args.rand_number
        self.nnode = nnode
        self.device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")
        self.args = args
        self.adj = adj
        if predictions is not None and self.args.cuda:
            predictions = predictions.cpu().numpy()
        elif predictions is not None:
            predictions = predictions.numpy()
        self.predictions = predictions
        if args.predicted:
            if self.args.prob:
                self.rand_nums = 1
                feats = torch.LongTensor(self.predictions).unsqueeze(0)
            else:
                feats = np.array([np.random.choice(range(self.nclass), self.rand_nums, replace=True, p=i) for i in self.predictions]).transpose()
                feats = torch.LongTensor(feats)
                feats = torch.stack([F.one_hot(feats[i]) for i in range(self.rand_nums)], dim=0)

            self.feats = feats



    def forward(self, x):
        if self.args.predicted:
            if self.args.prob:
                self.rand_nums = 1
                feats = torch.LongTensor(self.predictions).unsqueeze(0)
            else:
                feats = np.array([np.random.choice(range(self.nclass), self.rand_nums, replace=True, p=i) for i in self.predictions]).transpose()
                feats = torch.LongTensor(feats)
                feats = torch.stack([F.one_hot(feats[i]) for i in range(self.rand_nums)], dim=0)

            self.feats = feats
            if self.args.cuda:
                self.feats = self.feats.to(self.device)
                x = x.to(self.device)
            x_concat = torch.stack([x for i in range(self.rand_nums)], dim=0)
            x_concat = torch.cat([x_concat, self.feats.float()], dim=2)
            # x_concat = [torch.cat([x, f2[i], ], dim=1) for i in range(self.rand_nums)]
            x_concat = [F.relu(self.gc1(x_i, self.adj)) for x_i in x_concat]
            x_concat = [F.dropout(x_i, self.dropout, training=self.training) for x_i in x_concat]
            x_concat = [self.gc2(x_i, self.adj) for x_i in x_concat]
            x_concat = sum(x_concat) / len(x_concat)
            return F.log_softmax(x_concat, dim=1)
        else:
            x = F.relu(self.gc1(x, self.adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, self.adj)
            return F.log_softmax(x, dim=1)

    def update_prob(self, prob):
        self.predictions = prob.cpu().numpy()


class GCN_RAND_LABELED(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nnode, labels, args, adj=None):
        super(GCN_RAND_LABELED, self).__init__()
        if not args.baseline:
            nfeat += labels.max().item()+1
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = args.dropout
        self.nclass = nclass
        self.labels = labels
        self.nnode = nnode
        self.args = args
        self.adj = adj
        self.device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")
        self.rand_nums = args.rand_number


    def forward(self, x, idx_labeled, pre_train=False, predictions=None, baseline=False):
        if baseline:
            x = F.relu(self.gc1(x, self.adj))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, self.adj)
            return F.log_softmax(x, dim=1)
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
            if self.args.sample:
                feats = np.array([np.random.choice(range(self.nclass), self.rand_nums, replace=True, p=i) for i in predictions.cpu().numpy()]).transpose()
                feats = torch.from_numpy(feats)
                if self.args.cuda:
                    feats = feats.to(self.device)
                for i in range(self.rand_nums):
                    feats[i, idx_labeled] = self.labels[idx_labeled]
                feats = torch.stack([self.pad_zeros(F.one_hot(feats[i])) for i in range(self.rand_nums)], dim=0)
        if self.args.cuda:
            feats = feats.to(self.device)
            x = x.to(self.device)
        if (not self.args.sample) or pre_train:
            rand_nums = 1
        else:
            rand_nums = self.rand_nums
        x_concat = torch.stack([x for i in range(rand_nums)], dim=0)
        x_concat = torch.cat([x_concat, feats.float()], dim=2)
        x_concat = [F.relu(self.gc1(x_i, self.adj)) for x_i in x_concat]
        x_concat = [F.dropout(x_i, self.dropout, training=self.training) for x_i in x_concat]
        x_concat = [self.gc2(x_i, self.adj) for x_i in x_concat]
        x_concat = sum(x_concat) / len(x_concat)
        return F.log_softmax(x_concat, dim=1)

    def pad_zeros(self, feat):
        feat_new = torch.zeros(feat.shape[0], self.nclass).to(self.device)
        feat_new[:, :feat.shape[1]] = feat
        return feat_new





class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class LanczosNet(nn.Module):
    '''
    Spectral convolutions + MLP
    '''

    def __init__(self, k, nfeat, nclass, short_scales, long_scales, out_features=512, inner_dim=256):
        super(LanczosNet, self).__init__()
        self.spectral_conv1 = SpectralConv(nfeat, out_features, k, short_scales, long_scales, mlp_layers_number=1)
        self.spectral_conv2 = SpectralConv(out_features, inner_dim, k, short_scales, long_scales, mlp_layers_number=1)
        self.mlp = nn.Sequential(nn.Linear(inner_dim, nclass),
                                 nn.ReLU())
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.05)

    def forward(self, adj, X, S, V, R):
        Z = self.spectral_conv1(X, S, V, R)
        #         Z = self.dropout(Z)

        Z = self.spectral_conv2(Z, S, V, R)
        #         Z = self.bn(self.dropout(Z))
        Z = self.mlp(Z)

        return F.log_softmax(Z)


class LanczosConvNet(nn.Module):
    '''
    Spectral convolutions + graph convolutions
    '''

    def __init__(self, k, nfeat, nclass, short_scales, long_scales, adj, X, S, V, R, out_features=512, inner_dim=256):
        super(LanczosConvNet, self).__init__()
        self.spectral_conv = SpectralConv(nfeat, out_features, k, short_scales, long_scales, mlp_layers_number=1)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(0.05)
        self.conv1 = GraphConvolution(out_features + nfeat, inner_dim)
        self.conv2 = GraphConvolution(inner_dim, nclass)
        self.adj = adj
        self.S = S
        self.V = V
        self.R = R

    def forward(self, X):
        Z = self.spectral_conv(X, self.S, self.V, self.R)
        #         Z = self.bn(self.dropout(Z))
        Z = torch.cat((Z, X), 1)
        Z = self.conv1(Z, self.adj)
        Z = self.conv2(Z, self.adj)

        return F.log_softmax(Z)
