from __future__ import annotations
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from multi_head import MultiHeadSelfAttention
from feed_forward_network import FeedForward
from block import TransformerBlock

@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.block_size = config.block_size
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config.n_embd, config.n_head, config.dropout) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        _init_weights is applied to every layer
        Linear layers → Gaussian + zero bias
        Embedding layers → Gaussian
        std=0.02 comes from GPT-2; gives stability
        Without this, training can break easily
        """
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx:torch.Tensor, targets: torch.Tensor | None= None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                top_k: int | None = 50, top_p: float | None = None):
        """
        idx → token IDs of the prompt (shape: [B, T])
        max_new_tokens → how many tokens to generate
        temperature → randomness control
        top_k → sample only from top-K most likely tokens
        top_p → sample from nucleus (probability mass)"""
        from utils import top_k_top_p_filtering
        self.eval()
        # Guard: if the prompt is empty, start with a newline byte (10)
        if idx.size(1) == 0:
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


