import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from swattention.full_attention import full_attention
from swattention.sliding_window_attention import sliding_window_attention
from swattention.utils import seed_everything, select_device, synchronize


def _time_once(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    make_inputs: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    backward: bool,
    device: torch.device,
) -> float:
    q, k, v = make_inputs()
    synchronize(device)
    start = time.perf_counter()
    out = fn(q, k, v)
    if backward:
        loss = out.float().square().mean()
        loss.backward()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0


def bench_impl(
    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    make_inputs: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    iters: int,
    warmup: int,
    device: torch.device,
) -> tuple[float, float]:
    for _ in range(warmup):
        _ = fn(*make_inputs())

    forward_times = []
    backward_times = []

    for _ in range(iters):
        forward_times.append(_time_once(fn, make_inputs, backward=False, device=device))
        backward_times.append(_time_once(fn, make_inputs, backward=True, device=device))

    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    return avg_forward, avg_backward


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark full vs sliding-window attention")
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--window", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="benchmarks/swattn_bench.csv")
    args = parser.parse_args()

    seed_everything(0)
    device = select_device() if args.device == "auto" else torch.device(args.device)
    torch.set_grad_enabled(True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "impl",
                "seq",
                "window",
                "heads",
                "dim",
                "batch",
                "forward_ms",
                "backward_ms",
                "tokens_per_s",
                "peak_mem_MiB",
            ]
        )

        for seq_len in args.seq:
            for window in args.window:
                if window > seq_len:
                    continue

                batch, heads, dim = args.batch, args.heads, args.dim
                base_q = torch.randn(batch, heads, seq_len, dim, device=device)
                base_k = torch.randn_like(base_q)
                base_v = torch.randn_like(base_q)

                def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    q = base_q.clone().detach().requires_grad_(True)
                    k = base_k.clone().detach().requires_grad_(True)
                    v = base_v.clone().detach().requires_grad_(True)
                    return q, k, v

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                local_fwd, local_bwd = bench_impl(
                    lambda q, k, v: sliding_window_attention(
                        q,
                        k,
                        v,
                        window_size=window,
                        causal=True,
                    ),
                    make_inputs,
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                )

                peak_mem_local = 0
                if device.type == "cuda":
                    peak_mem_local = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

                tokens = batch * heads * seq_len
                tok_per_s_local = tokens / (local_fwd / 1000.0)

                writer.writerow(
                    [
                        "local",
                        seq_len,
                        window,
                        heads,
                        dim,
                        batch,
                        local_fwd,
                        local_bwd,
                        tok_per_s_local,
                        peak_mem_local,
                    ]
                )

                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

                full_fwd, full_bwd = bench_impl(
                    lambda q, k, v: full_attention(q, k, v, causal=True),
                    make_inputs,
                    iters=args.iters,
                    warmup=args.warmup,
                    device=device,
                )

                peak_mem_full = 0
                if device.type == "cuda":
                    peak_mem_full = int(torch.cuda.max_memory_allocated() / (1024 * 1024))

                tok_per_s_full = tokens / (full_fwd / 1000.0)

                writer.writerow(
                    [
                        "full",
                        seq_len,
                        window,
                        heads,
                        dim,
                        batch,
                        full_fwd,
                        full_bwd,
                        tok_per_s_full,
                        peak_mem_full,
                    ]
                )

                print(
                    f"seq={seq_len:4d} window={window:4d} | local {local_fwd:7.2f} ms | "
                    f"full {full_fwd:7.2f} ms"
                )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
