import torch
import torch.nn as nn
from Attention import Attention


class SocialAggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(SocialAggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)
            e_u = self.u2e.weight[list(tmp_adj)]
            u_rep = self.u2e.weight[nodes[i]]
            if num_neighs == 0:
                embed_matrix[i] = u_rep
            else:
                att_w = self.att(e_u, u_rep, num_neighs)
                att_history = torch.mm(e_u.t(), att_w).t()
                embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats