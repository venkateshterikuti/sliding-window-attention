import torch

from swattention.sliding_window_attention import sliding_window_attention


def test_shapes():
    bsz, heads, seq_len, dim = 2, 4, 128, 32
    q = torch.randn(bsz, heads, seq_len, dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out = sliding_window_attention(q, k, v, window_size=64, causal=True)

    assert out.shape == (bsz, heads, seq_len, dim)
