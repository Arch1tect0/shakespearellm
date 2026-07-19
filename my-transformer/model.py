"""A small GPT-style, decoder-only transformer language model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that cannot attend to future positions."""

    def __init__(self, config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # (block_size, block_size): lower triangle permits only current/past tokens.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # (1, 1, block_size, block_size): singleton batch/head axes broadcast.
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C), where B=batch, T=sequence length, C=embedding width.
        B, T, C = x.shape

        # Each projection preserves shape: (B, T, C) -> (B, T, C).
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split C across heads: (B, T, C) -> (B, T, n_head, head_size).
        # Transpose so each head has its own sequence: -> (B, n_head, T, head_size).
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # q @ k^T: (B, n_head, T, head_size) @ (B, n_head, head_size, T)
        # -> attention scores of shape (B, n_head, T, T).
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # Slice mask to this T and broadcast (1,1,T,T) across B and n_head.
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, n_head, T, T)

        # Weighted values: (B, n_head, T, T) @ (B, n_head, T, head_size)
        # -> one context vector per token/head: (B, n_head, T, head_size).
        out = weights @ v
        # Rejoin heads: (B, n_head, T, head_size) -> (B, T, n_head, head_size)
        # -> (B, T, C). contiguous() makes the transposed storage viewable.
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)  # (B, T, C) -> (B, T, C)


class MLP(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, T, 4C) -> GELU -> (B, T, C).
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-layernorm transformer block with residual connections."""

    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every term is (B, T, C), so each residual addition preserves shape.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Decoder-only transformer that predicts the next byte token."""

    def __init__(self, config) -> None:
        super().__init__()
        self.block_size = config.block_size
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: integer token IDs with shape (B, T).
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.block_size}")

        positions = torch.arange(T, device=idx.device)  # (T,)
        # Token lookup: (B, T) -> (B, T, C).
        token_embeddings = self.token_embedding(idx)
        # Position lookup: (T,) -> (T, C), then broadcast over B.
        position_embeddings = self.position_embedding(positions)
        x = token_embeddings + position_embeddings  # (B, T, C)

        for block in self.blocks:
            x = block(x)  # (B, T, C) -> (B, T, C)

        x = self.ln_f(x)  # (B, T, C)
        # Project each token state to all byte-token scores:
        # (B, T, C) -> (B, T, vocab_size).
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample and append ``max_new_tokens`` tokens."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        for _ in range(max_new_tokens):
            # Keep the newest context: (B, T) -> (B, min(T, block_size)).
            idx_context = idx[:, -self.block_size :]
            # (B, context_T) -> (B, context_T, vocab_size).
            logits = self(idx_context)
            # Only the final position predicts the next token: -> (B, vocab_size).
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                # top_values: (B, k); tokens outside the top k become impossible.
                top_values, _ = torch.topk(logits, k)
                cutoff = top_values[:, [-1]]  # (B, 1), broadcasts over vocabulary.
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probabilities = F.softmax(logits, dim=-1)  # (B, vocab_size)
            next_token = torch.multinomial(probabilities, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)  # (B, T) -> (B, T + 1)

        return idx
