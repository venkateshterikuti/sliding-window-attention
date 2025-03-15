from __future__ import annotations

import torch
import torch.nn.functional as F


def full_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    """Reference full attention with optional causal masking."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 tensors (B, H, N, D)")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share the same shape")

    _, _, seq_len, head_dim = q.shape
    scale = head_dim ** -0.5

    scores = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale

    if causal:
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(future_mask.view(1, 1, seq_len, seq_len), float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", weights, v)
