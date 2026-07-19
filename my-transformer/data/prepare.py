"""Download and tokenize Tiny Shakespeare into train.bin / val.bin.

Downloads the corpus if missing, encodes it with the byte-level
tokenizer, splits 90/10 into train/val, and saves both as numpy uint8
arrays next to this script.
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer import ByteTokenizer

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).resolve().parent


def main() -> None:
    input_path = DATA_DIR / "tinyshakespeare.txt"
    if not input_path.exists():
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, input_path)
    else:
        print(f"Found existing {input_path.name}, skipping download.")

    text = input_path.read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    tokens = np.array(tokenizer.encode(text), dtype=np.uint8)

    split = int(0.9 * len(tokens))
    train, val = tokens[:split], tokens[split:]

    train.tofile(DATA_DIR / "train.bin")
    val.tofile(DATA_DIR / "val.bin")

    print(f"Total tokens: {len(tokens):,}")
    print(f"Train tokens: {len(train):,}")
    print(f"Val tokens:   {len(val):,}")
    print(f"First 20 decoded tokens: {tokenizer.decode(tokens[:20])!r}")


if __name__ == "__main__":
    main()
