import torch
import torch.nn as nn
import torch.nn.functional as F

from Attention import Attention


class UVAggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggregator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(UVAggregator, self).__init__()

        self.uv = uv  # True for user aggregator, False for item aggregator
        self.v2e = v2e  # Embedding of items
        self.r2e = r2e  # Embedding of ratings
        self.u2e = u2e  # Embedding of users
        self.device = cuda
        self.embed_dim = embed_dim  # 64
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):
        embed_matrix = torch.empty(len(history_uv), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(history_uv)):
            # List of inter-class neighbors
            history = history_uv[i]

            # Number of neighbors
            num_history_item = len(history)

            # List of ratings
            tmp_label = history_r[i]

            if self.uv:
                # Neighbor items representation
                e_uv = self.v2e.weight[history]

                # User representation
                uv_rep = self.u2e.weight[nodes[i]]
            else:
                # Neighbor users representation
                e_uv = self.u2e.weight[history]

                # Item representation
                uv_rep = self.v2e.weight[nodes[i]]

            # Ratings representation
            e_r = self.r2e.weight[tmp_label]

            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))

            # Message representation
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, uv_rep, num_history_item)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history

        to_feats = embed_matrix

        return to_feats
