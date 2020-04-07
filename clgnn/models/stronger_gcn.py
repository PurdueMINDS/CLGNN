import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from models.layers import general_GCN_layer, snowball_layer, truncated_krylov_layer
except:
    from layers import general_GCN_layer, snowball_layer, truncated_krylov_layer


class graph_convolutional_network(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(graph_convolutional_network, self).__init__()
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout = dropout
        self.hidden = nn.ModuleList()

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()



class snowball(graph_convolutional_network):
    def __init__(self, args, nfeat, nlayers, nhid, nclass, dropout, activation):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid, args.cuda))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass, args.cuda)

    def forward(self, x, adj):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(
                    F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)),
                              self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
        return F.log_softmax(output, dim=1)


class snowball_rand(graph_convolutional_network):
    def __init__(self, args, adj, labels, nnode, nfeat, nlayers, nhid, nclass, dropout, activation, predictions=None):
        super(snowball_rand, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        self.args = args
        self.device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")
        self.dropout = args.dropout
        self.nclass = nclass
        self.labels = labels
        if args.prob:
            self.rand_nums = 1
        else:
            self.rand_nums = args.rand_number
        if predictions is not None and self.args.cuda:
            predictions = predictions.cpu().numpy()
        elif torch.is_tensor(predictions):
            predictions = predictions.numpy()
        self.predictions = predictions
        if args.predicted:
            nfeat += nclass
            if self.args.prob:
                self.rand_nums = 1
                feats = torch.from_numpy(self.predictions).long().unsqueeze(0)
            else:
                feats = np.array([np.random.choice(range(self.nclass), self.rand_nums, replace=True, p=i) for i in predictions]).transpose()
                feats = torch.from_numpy(feats).long()
                feats = torch.stack([self.pad_zeros(F.one_hot(feats[i])) for i in range(self.rand_nums)], dim=0)
            self.feats = feats
        self.nnode = nnode
        self.adj = adj
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid, args.cuda))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass, args.cuda)



    def forward(self, x):
        if self.args.predicted:
            if self.args.prob:
                self.rand_nums = 1
                feats = torch.from_numpy(self.predictions).long().unsqueeze(0)
            else:
                feats = np.array([np.random.choice(range(self.nclass), self.rand_nums, replace=True, p=i) for i in
                                  self.predictions]).transpose()
                feats = torch.from_numpy(feats).long()
                feats = torch.stack([self.pad_zeros(F.one_hot(feats[i])) for i in range(self.rand_nums)], dim=0)
            self.feats = feats
            if self.args.cuda:
                self.feats = self.feats.to(self.device)
                x = x.to(self.device)
            x_concat = torch.stack([x for i in range(self.rand_nums)], dim=0)
            x_concat = torch.cat([x_concat, self.feats.float()], dim=2)
            outputs = []
            for x in x_concat:
                list_output_blocks = []
                for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
                    if layer_num == 0:
                        list_output_blocks.append(
                            F.dropout(self.activation(layer(x, self.adj)), self.dropout, training=self.training))
                    else:
                        list_output_blocks.append(
                            F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), self.adj)),
                                      self.dropout, training=self.training))
                output = self.out(torch.cat([x] + list_output_blocks, 1), self.adj, eye=False)
                outputs.append(output)
            output = sum(outputs) / len(outputs)
        else:
            list_output_blocks = []
            for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(
                        F.dropout(self.activation(layer(x, self.adj)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(
                        F.dropout(self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), self.adj)),
                                  self.dropout, training=self.training))
            output = self.out(torch.cat([x] + list_output_blocks, 1), self.adj, eye=False)
        output = F.log_softmax(output, dim=1)
        return output

    def pad_zeros(self, feat):
        feat_new = torch.zeros(feat.shape[0], self.nclass).to(self.device)
        feat_new[:, :feat.shape[1]] = feat
        return feat_new

    def update_prob(self, prob):
        self.predictions = prob.cpu().numpy()


class snowball_rand_labeled(graph_convolutional_network):
    def __init__(self, args, adj, labels, nnode, nfeat, nlayers, nhid, nclass, dropout, activation):
        super(snowball_rand_labeled, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        self.activation = activation
        if not args.baseline:
            nfeat += labels.max().item()+1
        self.dropout = args.dropout
        self.nclass = nclass
        self.labels = labels
        self.device = torch.device("cuda:" + str(args.device) if args.cuda else "cpu")
        self.rand_nums = args.rand_number
        self.nnode = nnode
        self.adj = adj
        self.args = args
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid, args.cuda))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass, args.cuda)

    def forward(self, x, idx_labeled, pre_train=False, predictions=None, baseline=False):
        if baseline:
            # print("baseline")
            list_output_blocks = []
            for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(
                        F.dropout(self.activation(layer(x, self.adj)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(
                        F.dropout(
                            self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), self.adj)),
                            self.dropout, training=self.training))
            output = self.out(torch.cat([x] + list_output_blocks, 1), self.adj, eye=False)
            return F.log_softmax(output, dim=1)
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
        self.rand_nums = 1
        x_concat = torch.stack([x for i in range(self.rand_nums)], dim=0)
        x_concat = torch.cat([x_concat, feats.float()], dim=2)
        outputs = []
        for x in x_concat:
            list_output_blocks = []
            for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(
                        F.dropout(self.activation(layer(x, self.adj)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(
                        F.dropout(
                            self.activation(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), self.adj)),
                            self.dropout, training=self.training))
            output = self.out(torch.cat([x] + list_output_blocks, 1), self.adj, eye=False)
            outputs.append(output)
        output = sum(outputs) / len(outputs)
        return F.log_softmax(output, dim=1)

    def pad_zeros(self, feat):
        feat_new = torch.zeros(feat.shape[0], self.nclass).to(self.device)
        feat_new[:, :feat.shape[1]] = feat
        return feat_new
    

