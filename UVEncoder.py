import torch
import torch.nn as nn
import torch.nn.functional as F


class UVEncoder(nn.Module):
    def __init__(self, features, embed_dim, history_uv_lists, history_r_lists, aggregator, cuda="cpu", uv=True):
        super(UVEncoder, self).__init__()

        self.features = features  # User/item features
        self.uv = uv  # True for user encoder, False for item encoder
        self.history_uv_lists = history_uv_lists  # List of users/items
        self.history_r_lists = history_r_lists  # List of ratings
        self.aggregator = aggregator
        self.embed_dim = embed_dim  # 64
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        tmp_history_uv = []
        tmp_history_r = []

        for node in nodes:
            # Append a list of neighbors
            tmp_history_uv.append(self.history_uv_lists[int(node)])

            # Append a list of ratings for those neighbors
            tmp_history_r.append(self.history_r_lists[int(node)])

        neigh_feats = self.aggregator.forward(nodes, tmp_history_uv, tmp_history_r)

        self_feats = self.features.weight[nodes]
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
