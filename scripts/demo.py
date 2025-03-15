import argparse
import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from swattention.sliding_window_attention import sliding_window_attention
from swattention.utils import seed_everything, select_device, synchronize


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke test for sliding-window attention")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=64)
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
    out = sliding_window_attention(q, k, v, window_size=args.window, causal=True)
    synchronize(device)
    dt_ms = (time.perf_counter() - t0) * 1000

    print(f"device       : {device}")
    print(f"input shape  : {tuple(q.shape)}")
    print(f"output shape : {tuple(out.shape)}")
    print(f"elapsed      : {dt_ms:.2f} ms")


if __name__ == "__main__":
    main()
