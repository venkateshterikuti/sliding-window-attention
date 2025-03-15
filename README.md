# Sliding-Window Attention (from scratch)

This project implements sliding-window (local) attention in pure PyTorch and contrasts it with a reference full-attention baseline. It is designed as a portfolio-ready mini-project: the code is readable, fully tested, benchmarkable on laptop or GPU, and ships with plots/logs that make for compelling visuals in a blog post or recruiter packet.

## Highlights
- Clean local-attention kernel (`swattention/sliding_window_attention.py`) with gather-based indexing.
- Minimal Transformer block wired to the kernel for end-to-end tasks.
- Benchmarks that compare local vs full attention across sequence lengths, window sizes, and devices (CPU, MPS, CUDA).
- Tiny character language model showcasing training speed/quality with automatic logging and plots.
- Reproducible test suite verifying shapes and equivalence to full attention when the window covers the whole sequence.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```
(Use `python` instead of `python3` on Windows.)

### Smoke test
```bash
python scripts/demo.py --seq-len 256 --dim 64 --heads 4 --window 64
```

### Full vs local comparison
```bash
python scripts/full_vs_local_demo.py --seq-len 1024 --dim 64 --heads 4 --window 128
```
Outputs include per-kernel latency, relative speed-up, and max absolute difference between the two attention types.

## Benchmarks
Run a sequence/window sweep and save results + plots:
```bash
python scripts/benchmark.py \
  --seq 128 256 512 1024 2048 \
  --window 32 64 128 256 \
  --heads 4 --dim 64 --batch 2 --iters 20 --warmup 5 --device auto \
  --out benchmarks/swattn_bench.csv
python scripts/plot_benchmarks.py --csv benchmarks/swattn_bench.csv
```
Artifacts (`benchmarks/`):
- `swattn_bench.csv` — raw timing/memory data
- `swattn_tokens_per_s.png` — throughput vs sequence length
- `swattn_speedup.png` — local/full speedup
- `swattn_memory_MiB.png` — CUDA peak memory (if available)

For CUDA-specific benchmarking on e.g. an H100 instance:
```bash
python scripts/benchmark.py --device cuda \
  --seq 512 1024 2048 4096 --window 64 128 256 512 \
  --heads 8 --dim 128 --batch 4 --out benchmarks/swattn_bench_cuda.csv
```

## Tiny char language model
Train a language model that uses the sliding-window block and log its metrics:
```bash
python scripts/train_toy_language_model.py \
  --context 128 --window 64 --steps 500 --batch 32 \
  --log-csv logs/train.csv --sample-file logs/sample.txt
python scripts/plot_benchmarks.py --train-csv logs/train.csv
```
Artifacts:
- `logs/train.csv` — per-step loss & throughput
- `benchmarks/train_loss.png` and `benchmarks/train_tokens_per_s.png`
- `logs/sample.txt` — generated text snippet for storytelling/screenshots

## Hardware notes
- **Apple Silicon (M1/M2/M3):** PyTorch’s MPS backend is auto-detected via `select_device()`. Expect noticeable speedups vs CPU.
- **CPU-only laptops:** Reduce `--seq` / `--window` / `--batch` to keep demos responsive.
- **CUDA (H100 / A100 / etc):** Install PyTorch with the right CUDA wheels and set `--device cuda` for benchmarks. The scripts reset peak memory stats so you can quote memory savings directly.

## Project layout
```
swattention/      core kernels and Transformer blocks
scripts/          demos, benchmarks, plotting, tiny LM training
tests/            unit tests (pytest)
data/             toy corpus for the LM demo
benchmarks/, logs/ generated automatically when running scripts
```

## Testing
```bash
pytest
```
The tests cover output shapes and verify that sliding-window attention matches full attention when the window spans the entire sequence.

## Blogging tips
- Use the benchmark CSV/plots to illustrate scaling trends.
- Include the `train_loss.png` / `train_tokens_per_s.png` charts to show “real” training traces.
- Quote the generated text sample as proof-of-life.
- Highlight the complexity reduction (O(N*W) vs O(N^2)) using the README’s summary.
