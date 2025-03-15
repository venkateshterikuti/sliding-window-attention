import torch

from swattention.full_attention import full_attention
from swattention.sliding_window_attention import sliding_window_attention


def test_equivalence_when_window_ge_seq():
    bsz, heads, seq_len, dim = 1, 2, 32, 16
    q = torch.randn(bsz, heads, seq_len, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out_full = full_attention(q, k, v, causal=True)
    out_local = sliding_window_attention(q, k, v, window_size=seq_len, causal=True)

    assert torch.allclose(out_full, out_local, atol=1e-5, rtol=1e-5)
