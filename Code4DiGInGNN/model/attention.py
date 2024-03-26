from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


# '''
# a deprecated aggregation function
# '''


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_q

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att

def att_inter_agg(att_layer, self_feats, agg_feats, to_feats, embed_dim, a, b, bn, dropout, n, cuda):
    neigh_h = torch.cat((agg_feats.transpose(0, 1), to_feats.transpose(0, 1)), dim=0)
    combined = torch.cat((self_feats.repeat(2, 1), neigh_h), dim=1)
    attention = att_layer(bn(combined).mm(a))
    attention = dropout(attention, 0.2 , training=True)
    attention = att_layer(attention.mm(b))
    attention = torch.cat((attention[0:n, :], attention[n:2 * n, :]), dim=1)
    attention = F.softmax(attention, dim=1)
    # attention = F.dropout(attention, 0.1, training=True)

	# initialize the final neighbor embedding
    if cuda:
        aggregated = torch.zeros(size=(n, embed_dim)).cuda()
    else:
        aggregated = torch.zeros(size=(n, embed_dim))

    # add neighbor embeddings in each relation together with attention weights
    for r in range(2):
        aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

    return aggregated.transpose(0, 1)



