import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_mask import causal_mask

class MultiHeadSelfAttention(nn.Module):
    """ Multi-head attention.

    Dimensions (before masking):
      x:      (B, T, d_model) B: Batch size = how many sequences processed together / T: Sequence length= number of tokens in each sequence
      qkv:    (B, T, 3*d_model)
      view→   (B, T, 3, n_head, d_head)   where d_head = d_model // n_head
      split→  q,k,v each (B, T, n_head, d_head)
      swap→   (B, n_head, T, d_head)
      scores: (B, n_head, T, T) = q @ k^T / sqrt(d_head)
      weights:(B, n_head, T, T) = softmax(scores)
      context_vector:    (B, n_head, T, d_head) = weights @ v
      merge:  (B, T, n_head*d_head) = (B, T, d_model)
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head ==0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor): # (B,T,d_model)
        B, T, C = x.shape
        qkv = self.qkv(x)   # (B,T,3*C)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head) # (B,T,3,heads,dim)
        q, k, v = qkv.unbind(dim = 2)  # each: (B,T,heads,dim)
        q = q.transpose(1, 2)  # (B,heads,T,dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 0.1 / math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,heads,T,T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask, float('-inf'))
        w = F.softmax(attn, dim=-1)
        context_vector = torch.matmul(w,v)   # (B,heads,T,dim)

        out = context_vector.transpose(1,2).contiguous().view(B,T,C)  # (B,T,d_model)
        out = self.proj(out)
        return out 