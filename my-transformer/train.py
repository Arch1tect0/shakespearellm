"""Training loop for the byte-level GPT models.

Usage:
    python train.py model_a
    python train.py model_b

Loads pre-tokenized train.bin/val.bin, trains with AdamW and
cross-entropy, logs train/val loss to logs/<model>.csv, and saves the
final checkpoint to checkpoints/<model>.pt.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import ModelA, ModelB
from model import GPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

CONFIGS = {"model_a": ModelA, "model_b": ModelB}

# Training hyperparameters, sized for CPU training in a few minutes.
# Model B is much slower per step on CPU, so it gets fewer steps.
MAX_STEPS = {"model_a": 3000, "model_b": 3000}
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
EVAL_ITERS = 50


def load_split(name: str) -> np.ndarray:
    return np.memmap(DATA_DIR / f"{name}.bin", dtype=np.uint8, mode="r")


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: torch.device):
    """Sample random (x, y) pairs where y is x shifted left by one token."""
    offsets = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in offsets])
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in offsets]
    )
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(data, block_size, batch_size, device)
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level GPT.")
    parser.add_argument("model", choices=sorted(CONFIGS), help="which config to train")
    args = parser.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)
    device = torch.device("cpu")

    config = CONFIGS[args.model]()
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_data = load_split("train")
    val_data = load_split("val")

    LOG_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{args.model}.csv"
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}.pt"

    max_steps = MAX_STEPS[args.model]
    parameter_count = sum(p.numel() for p in model.parameters())
    print(
        f"Training {args.model} ({parameter_count:,} parameters) on {device} "
        f"for {max_steps} steps",
        flush=True,
    )

    start_time = time.time()
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["step", "train_loss", "val_loss"])

        for step in range(1, max_steps + 1):
            x, y = get_batch(train_data, config.block_size, BATCH_SIZE, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % EVAL_INTERVAL == 0 or step == max_steps:
                train_loss = estimate_loss(model, train_data, config.block_size, BATCH_SIZE, device)
                val_loss = estimate_loss(model, val_data, config.block_size, BATCH_SIZE, device)
                elapsed = time.time() - start_time
                timestamp = datetime.now().strftime("%H:%M:%S")
                writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}"])
                log_file.flush()
                print(
                    f"[{timestamp}] step {step:5d} | train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | {elapsed:.1f}s elapsed",
                    flush=True,
                )

    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    total_time = time.time() - start_time
    print(f"Done in {total_time:.1f}s. Checkpoint saved to {checkpoint_path}", flush=True)


if __name__ == "__main__":
    main()
