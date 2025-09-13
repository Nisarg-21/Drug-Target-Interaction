import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, device=None):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert (self.head_dim * num_heads == input_dim), "Input dimension must be divisible by number of heads"

        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)
        self.W_o = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.scaling = torch.sqrt(torch.FloatTensor([self.head_dim]))
        self.device = device

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scaling = self.scaling.to(query.device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scaling
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = F.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.input_dim)

        x = self.W_o(x)
        return x, attention