import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, embedding_dims):
        super(SelfAttention, self).__init__()
        self.embed_dim = embedding_dims
        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.embed_dim, self.embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.linear_final = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, q, k, v, batch_size, scale=None):
        Q = self.linear_q(q)
        K = self.linear_k(k)
        V = self.linear_v(v)
        residual = Q

        Q = Q.view(batch_size, 1, self.embed_dim)
        K = K.view(batch_size, -1, self.embed_dim)
        V = V.view(batch_size, -1, self.embed_dim)

        attention = torch.bmm(Q, K.transpose(1, 2))

        if scale:
            attention = attention * scale

        attention = self.softmax(attention)
        attention = F.dropout(attention, training=self.training)
        context = torch.bmm(attention, V)

        context = context.view(batch_size, self.embed_dim)

        output = self.linear_final(context)
        output = F.dropout(output, training=self.training)
        output = self.layer_norm(residual + output)

        return output, attention
