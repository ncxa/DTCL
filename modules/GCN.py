import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch import FloatTensor
import numpy as np
from scipy.spatial.distance import pdist, squareform

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=False, residual=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        if not residual:
            self.residual = lambda x: 0
        elif (in_features == out_features):
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=5, padding=2)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.1)

    def forward(self, input, adj):
        support = input.matmul(self.weight)
        output = adj.matmul(support)

        if self.bias is not None:
            output = output + self.bias
        if self.in_features != self.out_features and self.residual:
            input = input.permute(0, 2, 1)
            res = self.residual(input)
            res = res.permute(0, 2, 1)
            output = output + res
        else:
            output = output + self.residual(input)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DistanceAdj(Module):
    def __init__(self):
        super(DistanceAdj, self).__init__()
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)

    def forward(self, batch_size, max_seqlen):
        dist = torch.arange(max_seqlen, device="cuda:1").view(-1, 1)
        dist = torch.abs(dist - dist.T).float() # Cityblock distance
        dist = torch.exp(-dist / torch.exp(torch.tensor(1., device="cuda:1")))
        return dist.unsqueeze(0).repeat(batch_size, 1, 1)


def adj4(x, seq_len=None):
    soft = nn.Softmax(dim=1)
    x2 = x.matmul(x.permute(0, 2, 1))  # (B, T, T)
    x_norm = torch.norm(x, p=2, dim=2, keepdim=True)  # (B, T, 1)
    x_norm_x = x_norm.matmul(x_norm.permute(0, 2, 1))  # (B, T, T)
    x2 = x2 / (x_norm_x + 1e-20)  # normalize
    output = torch.zeros_like(x2)

    if seq_len is None:
        for i in range(x.shape[0]):
            tmp = x2[i]
            adj2 = F.threshold(tmp, 0.7, 0)  # threshold
            adj2 = soft(adj2)  # normalize
            output[i] = adj2
    else:
        for i in range(x.shape[0]):
            tmp = x2[i]
            adj2 = F.threshold(tmp, 0.7, 0)  # threshold
            adj2 = soft(adj2)  # normalize
            output[i] = adj2

    return output


class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()


        self.gc2 = GraphConvolution(in_features, out_features, residual=True).to("cuda:1")
        self.distance_adj = DistanceAdj()
        self.linear = nn.Linear(out_features * 2, out_features).to("cuda:1")

    def forward(self, x):
        seq_len = x.shape[1]
        # adj = adj4(x).to("cuda:1")  # Adjacency matrix from input
        disadj = self.distance_adj(x.shape[0], seq_len)  # Distance adjacency matrix
        # x1 = F.gelu(self.gc1(x, adj))  # GCN layer 1
        x2 = F.gelu(self.gc2(x, disadj))  # GCN layer 2
        # x = torch.cat((x1, x2), dim=2)  # Concatenate features
        # x = self.linear(x)  # Final linear layer
        return x2


if __name__ == "__main__":
    # Example usage
    x = torch.randn(50, 300, 512).to("cuda:1")
    model = GCN(in_features=512, out_features=512)  # Instantiate GCN model
    output = model(x)  # Forward pass
    print("Output shape:", output.shape)
