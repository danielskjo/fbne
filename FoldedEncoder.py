import torch
import torch.nn as nn
import torch.nn.functional as F

from SelfAttention import SelfAttention


class FoldedEncoder(nn.Module):
    def __init__(self, features, uv2e, embed_dim, seq_len, folded_seq, cuda="cpu"):
        super(FoldedEncoder, self).__init__()

        self.features = features
        self.uv2e = uv2e  # User/item embedding
        self.seq_len = seq_len  # 5
        self.embed_dim = embed_dim  # 64
        self.device = cuda
        self.folded_seq = folded_seq  # Random walks
        self.self_attention = SelfAttention(embed_dim)
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def forward(self, nodes):
        node_seq = []

        for node in nodes.cpu().numpy():
            seq = self.folded_seq[int(node)]

            while len(seq) < self.seq_len:
                seq.append(node)

            node_seq.append(seq[::-1])

        batch_size = len(node_seq)

        node_seq = [item for sublist in node_seq for item in sublist]

        q = self.uv2e.weight[nodes].to(self.device)
        k = self.uv2e.weight[torch.LongTensor(node_seq).to(self.device)]
        v = self.uv2e.weight[torch.LongTensor(node_seq).to(self.device)]

        folded_feats, attention = self.self_attention.forward(q, k, v, batch_size, scale=0.125)

        self_feats = self.features(torch.LongTensor(nodes.cpu().numpy())).to(self.device)
        self_feats = self_feats.t()

        combined = torch.cat([self_feats, folded_feats.view(batch_size, self.embed_dim)], dim=1)
        combined = F.relu(self.linear1(combined))

        return combined
