from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _make_indices(
    n: int,
    window_size: int,
    causal: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Build gather indices (N, W) and a boolean mask marking invalid entries."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    w = min(window_size, n)

    if causal:
        offsets = torch.arange(-w + 1, 1, device=device)
    else:
        if w % 2 == 0:
            raise ValueError("Non-causal attention expects an odd window_size for symmetry.")
        half = w // 2
        offsets = torch.arange(-half, half + 1, device=device)

    base = torch.arange(n, device=device).view(-1, 1)
    idx_raw = base + offsets.view(1, -1)

    valid = (idx_raw >= 0) & (idx_raw < n)
    if causal:
        valid = valid & (idx_raw <= base)

    idx = idx_raw.clamp(0, n - 1)
    mask = ~valid

    return idx, mask


def sliding_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    *,
    causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Vectorised sliding-window attention."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must be rank-4 tensors (B, H, N, D)")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, v must share the same shape")

    bsz, n_heads, seq_len, head_dim = q.shape
    device = q.device

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    idx, mask = _make_indices(seq_len, window_size, causal, device)
    window = idx.size(1)

    gather_idx = idx.view(1, 1, seq_len, window, 1).expand(bsz, n_heads, seq_len, window, head_dim)

    k_exp = k.unsqueeze(3).expand(-1, -1, -1, window, -1)
    v_exp = v.unsqueeze(3).expand(-1, -1, -1, window, -1)
    k_win = torch.gather(k_exp, dim=2, index=gather_idx)
    v_win = torch.gather(v_exp, dim=2, index=gather_idx)

    scores = (q.unsqueeze(3) * k_win).sum(dim=-1) * scale

    if mask is not None:
        scores = scores.masked_fill(mask.view(1, 1, seq_len, window), float("-inf"))

    attn = F.softmax(scores, dim=-1)
    out = (attn.unsqueeze(-1) * v_win).sum(dim=3)

    return out
