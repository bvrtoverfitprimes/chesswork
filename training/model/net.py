import torch
import torch.nn as nn

NUM_FEATURES = 24832
NUM_OUTPUT_BUCKETS = 8
HEAD_WIDTH = 32


def _init_linear_weight(out_dim, in_dim, num_buckets):
    w = torch.empty(num_buckets, out_dim, in_dim)
    bound = (2.0 / in_dim) ** 0.5
    nn.init.normal_(w, std=bound)
    return nn.Parameter(w)


class NnueNet(nn.Module):
    def __init__(self, hidden=256, num_buckets=NUM_OUTPUT_BUCKETS):
        super().__init__()
        self.hidden = hidden
        self.num_buckets = num_buckets

        self.embedding = nn.EmbeddingBag(NUM_FEATURES, hidden, mode="sum")
        nn.init.normal_(self.embedding.weight, std=0.01)

        self.fc1_w = _init_linear_weight(HEAD_WIDTH, hidden * 2, num_buckets)
        self.fc1_b = nn.Parameter(torch.zeros(num_buckets, HEAD_WIDTH))
        self.fc2_w = _init_linear_weight(HEAD_WIDTH, HEAD_WIDTH, num_buckets)
        self.fc2_b = nn.Parameter(torch.zeros(num_buckets, HEAD_WIDTH))
        self.fc3_w = _init_linear_weight(1, HEAD_WIDTH, num_buckets)
        self.fc3_b = nn.Parameter(torch.zeros(num_buckets, 1))

    def forward(self, stm_idx, stm_offsets, ntm_idx, ntm_offsets, bucket):
        stm_acc = self.embedding(stm_idx, stm_offsets)
        ntm_acc = self.embedding(ntm_idx, ntm_offsets)
        x = torch.cat([stm_acc, ntm_acc], dim=1)

        w1 = self.fc1_w[bucket]
        b1 = self.fc1_b[bucket]
        h1 = torch.relu(torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1)

        w2 = self.fc2_w[bucket]
        b2 = self.fc2_b[bucket]
        h2 = torch.relu(torch.bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2)

        w3 = self.fc3_w[bucket]
        b3 = self.fc3_b[bucket]
        out = torch.bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3
        return out.squeeze(-1)
