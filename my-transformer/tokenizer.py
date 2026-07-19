"""Byte-level tokenizer for the Tiny Shakespeare corpus.

Encodes text to integer token IDs (one per UTF-8 byte) and decodes IDs
back to text. The vocabulary is the fixed set of 256 byte values.
"""

VOCAB_SIZE = 256


class ByteTokenizer:
    """Tokenizer over raw UTF-8 bytes with a fixed 256-token vocabulary."""

    vocab_size = VOCAB_SIZE

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")
