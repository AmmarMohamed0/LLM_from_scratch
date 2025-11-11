import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
    """Single-head attention (explicit shapes)."""
    def __init__(self, d_model: int, key_dim: int, dropout:float = 0.0, trace_shapes: bool = True):
        super().__init__()
        self.q = nn.Linear(d_model, key_dim, bias=False)
        self.k = nn.Linear(d_model, key_dim, bias=False)
        self.v = nn.Linear(d_model, key_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.q(x)  # (B,T,key_dim)
        k = self.k(x)  # (B,T,key_dim)
        v = self.v(x)  # (B,T,key_dim)
        if self.trace_shapes:
            print(f"q {q.shape}  k {k.shape}  v {v.shape}")
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,T,T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)  # (B,T,key_dim)
        if self.trace_shapes:
            print(f"weights {w.shape}  out {out.shape}")
        return out, w
