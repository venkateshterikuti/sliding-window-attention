import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_benchmarks(path: Path, outdir: Path) -> None:
    rows = _read_csv(path)
    if not rows:
        print(f"No rows in {path}")
        return

    seqs = sorted({int(row["seq"]) for row in rows})

    def collect(metric: str, impl: str) -> List[float]:
        values = []
        for seq in seqs:
            subset = [row for row in rows if int(row["seq"]) == seq and row["impl"] == impl]
            if not subset:
                values.append(0.0)
                continue
            values.append(sum(float(row[metric]) for row in subset) / len(subset))
        return values

    local_tokps = collect("tokens_per_s", "local")
    full_tokps = collect("tokens_per_s", "full")
    speedup = [l / f if f else 0.0 for l, f in zip(local_tokps, full_tokps)]

    plt.figure()
    plt.plot(seqs, local_tokps, marker="o", label="local")
    plt.plot(seqs, full_tokps, marker="o", label="full")
    plt.xlabel("Sequence length")
    plt.ylabel("Tokens / second")
    plt.title("Throughput vs sequence length")
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / "swattn_tokens_per_s.png", bbox_inches="tight")

    plt.figure()
    plt.plot(seqs, speedup, marker="o")
    plt.xlabel("Sequence length")
    plt.ylabel("Speedup (local / full)")
    plt.title("Speedup vs sequence length")
    plt.grid(True)
    plt.savefig(outdir / "swattn_speedup.png", bbox_inches="tight")

    mem_local = collect("peak_mem_MiB", "local")
    mem_full = collect("peak_mem_MiB", "full")
    if any(mem_local) or any(mem_full):
        plt.figure()
        plt.plot(seqs, mem_local, marker="o", label="local")
        plt.plot(seqs, mem_full, marker="o", label="full")
        plt.xlabel("Sequence length")
        plt.ylabel("Peak CUDA memory (MiB)")
        plt.title("Peak memory vs sequence length")
        plt.grid(True)
        plt.legend()
        plt.savefig(outdir / "swattn_memory_MiB.png", bbox_inches="tight")


def plot_training(path: Path, outdir: Path) -> None:
    rows = _read_csv(path)
    if not rows:
        print(f"No rows in {path}")
        return

    steps = [int(row["step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows]
    tokps = [float(row["tokens_per_s"]) for row in rows]

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.grid(True)
    plt.savefig(outdir / "train_loss.png", bbox_inches="tight")

    plt.figure()
    plt.plot(steps, tokps)
    plt.xlabel("Step")
    plt.ylabel("Tokens / second")
    plt.title("Training throughput")
    plt.grid(True)
    plt.savefig(outdir / "train_tokens_per_s.png", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark/training CSVs")
    parser.add_argument("--csv", type=Path, default=Path("benchmarks/swattn_bench.csv"))
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.csv and args.csv.exists():
        plot_benchmarks(args.csv, args.outdir)
    if args.train_csv and args.train_csv.exists():
        plot_training(args.train_csv, args.outdir)


if __name__ == "__main__":
    main()
