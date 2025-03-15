import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from swattention.blocks import LocalTransformerBlock
from swattention.utils import seed_everything, select_device


class TinyCharLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        window: int = 64,
        context: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context = context
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, context, d_model))
        self.blocks = nn.ModuleList(
            [
                LocalTransformerBlock(
                    d_model,
                    n_heads,
                    window,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :seq_len, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x)


def build_vocab(text: str) -> Tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    context: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, data.size(0) - context - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + context] for i in idx])
    y = torch.stack([data[i + 1 : i + context + 1] for i in idx])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny character LM with sliding-window attention")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--data", type=Path, default=Path("data/tiny.txt"))
    parser.add_argument("--log-csv", type=Path, default=Path("logs/train.csv"))
    parser.add_argument("--sample-file", type=Path, default=Path("logs/sample.txt"))
    parser.add_argument("--print-every", type=int, default=50)
    args = parser.parse_args()

    seed_everything(0)
    device = select_device()

    text = args.data.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)
    data_all = encode(text, stoi).to(device)

    model = TinyCharLM(
        vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        window=args.window,
        context=args.context,
        dropout=args.dropout,
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    log_path = args.log_csv
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "loss", "wall_s", "tokens_per_s"])

    model.train()
    tokens_per_step = args.batch * args.context

    for step in range(1, args.steps + 1):
        xb, yb = get_batch(data_all, args.context, args.batch, device)
        step_start = time.perf_counter()

        logits = model(xb)
        loss = loss_fn(logits.view(-1, vocab_size), yb.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        wall = time.perf_counter() - step_start
        logger.writerow([step, float(loss.item()), wall, tokens_per_step / max(wall, 1e-9)])

        if step == 1 or step % args.print_every == 0:
            print(f"step {step:04d} | loss {loss.item():.4f} | {tokens_per_step / max(wall, 1e-9):.1f} tok/s")

    log_file.close()

    model.eval()
    with torch.no_grad():
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = []
        for _ in range(200):
            logits = model(idx[:, -args.context:])[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            generated.append(next_id.item())

    sample = "".join(itos[i] for i in generated)
    print("\nSample:\n" + sample)

    sample_path = args.sample_file
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(sample, encoding="utf-8")
    print(f"Sample saved to {sample_path}")


if __name__ == "__main__":
    main()
