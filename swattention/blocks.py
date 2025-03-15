from __future__ import annotations

import torch
import torch.nn as nn

from .sliding_window_attention import sliding_window_attention


class LocalAttention(nn.Module):
    """Multi-head attention that restricts each token to a local window."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        *,
        causal: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Input must have shape (batch, seq, d_model)")

        batch, seq_len, _ = x.shape
        heads = self.n_heads
        head_dim = self.d_head

        q = self.q_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, heads, head_dim).transpose(1, 2)

        out = sliding_window_attention(
            q,
            k,
            v,
            window_size=self.window_size,
            causal=self.causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        out = self.o_proj(out)
        return self.dropout(out)


class LocalTransformerBlock(nn.Module):
    """Minimal Transformer block using LocalAttention + MLP."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int,
        *,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(mlp_ratio * d_model)

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LocalAttention(
            d_model,
            n_heads,
            window_size,
            causal=True,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
