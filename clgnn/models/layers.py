import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F




class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # Wx + b
        support = torch.mm(input, self.weight)
        # A(WX + b)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SpectralConv(nn.Module):
    def __init__(self, in_features, out_features, k, short_scales, long_scales,
                 mlp_layers_number=3):
        super(SpectralConv, self).__init__()
        '''
        in_features - number of in features
        out_features - number of out features
        k - number of Ritz values, k<=n
        short_scales - scales for direct filters S^qX
        long_scales - scales of spectral filters VRV^T
        mlp_layers_number - number of layers in R values transformation
        '''
        self.short_scales = short_scales
        self.long_scales = long_scales
        self.mlp = []

        for _ in range(mlp_layers_number):
            self.mlp.append(nn.modules.Linear(k, k))
            self.mlp.append(nn.modules.ReLU())
        if len(self.mlp) > 0:
            self.mlp = nn.Sequential(*self.mlp[:-1])
        else:
            self.mlp = IdentityModule()

        self.W = Parameter(torch.Tensor(in_features*(len(short_scales) +
                                                     len(long_scales)), out_features))
        stdv = 1. / (in_features*(len(short_scales) + len(long_scales)))**0.5
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, X, S, V, R):
        '''
        X - featurs
        S - affinity matrix
        V - Q@B - Q from Lanczos, B from eigendecomposition
        R - Ritz values
        '''
        Y = X
        Z = Y
        features = []
        for l in range(1, self.short_scales[-1] + 1):
            # print(S.shape, Z.shape)
            Z = torch.mm(S, Z)
            if l in self.short_scales:
                features.append(Z)

        for i in self.long_scales:
            R_h = self.mlp(R**i)
            # print(V.shape, torch.diag(R_h).shape)
            Z = torch.mm(V, R_h)
            Z = torch.mm(Z, V.t())
            Z = torch.mm(Z, Y)
            features.append(Z)

        return torch.mm(torch.cat(features, 1), self.W)



class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'
        N = input.size()[0]
        # edge = adj.nonzero().t()
        edge = adj

        h = torch.mm(input, self.W)
        # h: N x out
        if torch.isnan(h).any():
            print(input, h, self.W)

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'





class general_GCN_layer(Module):
    def __init__(self):
        super(general_GCN_layer, self).__init__()

    @staticmethod
    def multiplication(A, B):
        if str(A.layout) == 'torch.sparse_coo':
            return torch.spmm(A, B)
        else:
            return torch.mm(A, B)


class snowball_layer(general_GCN_layer):
    def __init__(self, in_features, out_features, cuda_flag):
        super(snowball_layer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        if cuda_flag:
            self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).cuda()), Parameter(
                torch.FloatTensor(self.out_features).cuda())
        else:
            self.weight, self.bias = Parameter(
                torch.FloatTensor(self.in_features, self.out_features)), Parameter(
                torch.FloatTensor(self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        torch.nn.init.uniform_(self.weight, -stdv_weight, stdv_weight)
        torch.nn.init.uniform_(self.bias, -stdv_bias, stdv_bias)

    def forward(self, input, adj, eye=False):
        XW = torch.mm(input, self.weight)
        if eye:
            return XW + self.bias
        else:
            return self.multiplication(adj, XW) + self.bias


class truncated_krylov_layer(general_GCN_layer):
    def __init__(self, in_features, n_blocks, out_features, LIST_A_EXP=None, LIST_A_EXP_X_CAT=None):
        super(truncated_krylov_layer, self).__init__()
        self.LIST_A_EXP = LIST_A_EXP
        self.LIST_A_EXP_X_CAT = LIST_A_EXP_X_CAT
        self.in_features, self.out_features, self.n_blocks = in_features, out_features, n_blocks
        self.shared_weight, self.output_bias = Parameter(
            torch.FloatTensor(self.in_features * self.n_blocks, self.out_features).cuda()), Parameter(
            torch.FloatTensor(self.out_features).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv_shared_weight, stdv_output_bias = 1. / math.sqrt(self.shared_weight.size(1)), 1. / math.sqrt(
            self.output_bias.size(0))
        torch.nn.init.uniform_(self.shared_weight, -stdv_shared_weight, stdv_shared_weight)
        torch.nn.init.uniform_(self.output_bias, -stdv_output_bias, stdv_output_bias)

    def forward(self, input, adj, eye=True):
        if self.n_blocks == 1:
            output = torch.mm(input, self.shared_weight)
        elif self.LIST_A_EXP_X_CAT is not None:
            output = torch.mm(self.LIST_A_EXP_X_CAT, self.shared_weight)
        elif self.LIST_A_EXP is not None:
            feature_output = []
            for i in range(self.n_blocks):
                AX = self.multiplication(self.LIST_A_EXP[i], input)
                feature_output.append(AX)
            output = torch.mm(torch.cat(feature_output, 1), self.shared_weight)
        if eye:
            return output + self.output_bias
        else:
            return self.multiplication(adj, output) + self.output_bias


class IdentityModule(nn.Module):
    def forward(self, inputs):
        return inputs


class ExpKernel(nn.Module):
    def __init__(self, X, A, layers=None, e=10, learn_embedding=False):
        super().__init__()
        self.mlp = []
        if learn_embedding:
            self.X = Parameter(X)
        else:
            self.X = X

        self.A = Parameter(torch.Tensor(np.sign(A)))
        shape = X.shape[-1]

        if layers is not None:
            for layer in layers:
                self.mlp.append(nn.modules.Linear(shape, layer))
                self.mlp.append(nn.modules.ReLU())
                shape = layer
        if len(self.mlp) > 0:
            self.mlp = nn.Sequential(*self.mlp[:-1])
        else:
            self.mlp = IdentityModule()

        self.e = e

    def forward(self, input):
        Y = self.mlp(self.X)
        n = Y.size(0)
        norms = torch.sum(Y**2, dim=1, keepdim=True)
        norms_squares = (norms.expand(n, n) + norms.t().expand(n, n))
        distances_squared = torch.sqrt(1e-6 + norms_squares - 2 * Y.mm(Y.t()))
        A = torch.exp(-distances_squared/self.e)
        return torch.clamp(A * self.A, 0, 10)  
