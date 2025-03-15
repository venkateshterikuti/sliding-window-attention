import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from swattention.full_attention import full_attention
from swattention.sliding_window_attention import sliding_window_attention
from swattention.utils import seed_everything, select_device, synchronize


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full vs sliding-window attention")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    seed_everything(0)
    device = select_device()

    bsz, heads, seq_len, dim = args.batch, args.heads, args.seq_len, args.dim
    q = torch.randn(bsz, heads, seq_len, dim, device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    synchronize(device)
    t0 = time.perf_counter()
    local_out = sliding_window_attention(q, k, v, window_size=args.window, causal=True)
    synchronize(device)
    local_ms = (time.perf_counter() - t0) * 1000

    synchronize(device)
    t1 = time.perf_counter()
    full_out = full_attention(q, k, v, causal=True)
    synchronize(device)
    full_ms = (time.perf_counter() - t1) * 1000

    diff = (local_out - full_out).abs().max().item()
    speedup = full_ms / local_ms if local_ms > 0 else float("inf")

    print(f"device         : {device}")
    print(f"shape          : {tuple(local_out.shape)}")
    print(f"local time     : {local_ms:.2f} ms")
    print(f"full time      : {full_ms:.2f} ms")
    print(f"speedup (full/local): {speedup:.2f}x")
    print(f"max |diff|     : {diff:.3e}")


if __name__ == "__main__":
    main()
