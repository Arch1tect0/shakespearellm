"""Quantitative evaluation: loss curves, val CE/perplexity, comparison table.

Loads training CSV logs, plots Model A vs Model B train/val curves, then
reloads both checkpoints and computes held-out validation cross-entropy
and perplexity for a printed comparison table.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MY_TRANSFORMER = PROJECT_ROOT / "my-transformer"
sys.path.insert(0, str(MY_TRANSFORMER))

from model import GPT  # noqa: E402

LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = MY_TRANSFORMER / "data"
PLOT_PATH = PROJECT_ROOT / "loss_curves.png"

MODELS = [
    {"name": "Model A", "key": "model_a", "log": "model_a.csv", "ckpt": "model_a.pt"},
    {"name": "Model B", "key": "model_b", "log": "model_b.csv", "ckpt": "model_b.pt"},
]

EVAL_BATCH_SIZE = 64


def load_log(path: Path) -> dict[str, list[float]]:
    steps, train_loss, val_loss = [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
    return {"step": steps, "train_loss": train_loss, "val_loss": val_loss}


def plot_loss_curves(logs: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    plt.figure(figsize=(9, 5.5))
    styles = {
        "Model A": {"train": ("#1f77b4", "-"), "val": ("#1f77b4", "--")},
        "Model B": {"train": ("#d62728", "-"), "val": ("#d62728", "--")},
    }
    for name, series in logs.items():
        train_color, train_ls = styles[name]["train"]
        val_color, val_ls = styles[name]["val"]
        plt.plot(
            series["step"],
            series["train_loss"],
            color=train_color,
            linestyle=train_ls,
            linewidth=2,
            label=f"{name} train",
        )
        plt.plot(
            series["step"],
            series["val_loss"],
            color=val_color,
            linestyle=val_ls,
            linewidth=2,
            label=f"{name} val",
        )

    plt.xlabel("Step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss: Model A vs Model B")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


@torch.no_grad()
def evaluate_val_loss(model: GPT, val_data: np.ndarray, device: torch.device) -> float:
    """Mean token-level cross-entropy over the full held-out val set."""
    model.eval()
    block_size = model.block_size
    total_loss = 0.0
    total_tokens = 0

    # Non-overlapping windows so every val token (except the first of each
    # window's target alignment) is scored exactly once.
    starts = range(0, len(val_data) - block_size - 1, block_size)
    batch_x, batch_y = [], []

    def flush() -> None:
        nonlocal total_loss, total_tokens, batch_x, batch_y
        if not batch_x:
            return
        x = torch.stack(batch_x).to(device)
        y = torch.stack(batch_y).to(device)
        logits = model(x)
        # logits/y are (B, T); average over tokens in this batch.
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += y.numel()
        batch_x, batch_y = [], []

    for start in starts:
        x = torch.from_numpy(val_data[start : start + block_size].astype(np.int64))
        y = torch.from_numpy(val_data[start + 1 : start + 1 + block_size].astype(np.int64))
        batch_x.append(x)
        batch_y.append(y)
        if len(batch_x) >= EVAL_BATCH_SIZE:
            flush()
    flush()

    if total_tokens == 0:
        raise RuntimeError("Validation set too short for evaluation.")
    return total_loss / total_tokens


def load_checkpoint(path: Path, device: torch.device) -> tuple[GPT, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def print_table(rows: list[dict]) -> None:
    headers = ["Model", "Params", "Final Val Loss", "Perplexity"]
    cells = [
        [
            r["name"],
            f"{r['params']:,}",
            f"{r['val_loss']:.4f}",
            f"{r['perplexity']:.2f}",
        ]
        for r in rows
    ]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]

    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print(fmt(row))


def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(1337)

    logs = {}
    for m in MODELS:
        logs[m["name"]] = load_log(LOG_DIR / m["log"])

    plot_loss_curves(logs, PLOT_PATH)
    print(f"Saved loss curves to {PLOT_PATH}", flush=True)

    val_data = np.memmap(DATA_DIR / "val.bin", dtype=np.uint8, mode="r")
    rows = []
    for m in MODELS:
        model, n_params = load_checkpoint(CHECKPOINT_DIR / m["ckpt"], device)
        val_loss = evaluate_val_loss(model, val_data, device)
        rows.append(
            {
                "name": m["name"],
                "params": n_params,
                "val_loss": val_loss,
                "perplexity": math.exp(val_loss),
            }
        )
        print(
            f"Evaluated {m['name']}: val_loss={val_loss:.4f}, "
            f"perplexity={math.exp(val_loss):.2f}",
            flush=True,
        )

    print(flush=True)
    print_table(rows)


if __name__ == "__main__":
    main()
