"""
A headless transformer.

This is the part that makes the problem-5 claim concrete rather than
rhetorical. A conventional language model ends with an output head: a
V x d matrix that turns the final hidden state into a score for every
word in the vocabulary. At V=131,072 and d=8096 that matrix alone holds
about 1.06 billion parameters, and it grows linearly with vocabulary --
which is exactly why a 1M-token vocabulary is currently unaffordable.

If the embedding map is invertible, that matrix is unnecessary. The
model can emit a vector directly and the inverse map reads the word off
it. So this model has no output head at all:

    tokens -> embeddings (deterministic, untrained)
           -> transformer blocks
           -> a d-dimensional vector
           -> inverse map -> word

`parameter_report` prints the parameter count with and without the head
it does not have, so the saving is a measured number in the results
rather than a claim in the README.

Trained with cosine loss rather than MSE, because the inverse map is
scale-invariant -- it takes an argmax over inner products, so only the
direction of the predicted vector matters. Training against magnitude
would spend capacity on something the decoder discards.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadlessTransformer(nn.Module):
    def __init__(self, dim: int, context: int = 16, layers: int = 2,
                 heads: int = 8, ff_multiplier: int = 2):
        super().__init__()
        self.dim = dim
        self.context = context

        self.position = nn.Parameter(torch.zeros(1, context, dim))
        nn.init.normal_(self.position, std=0.02)

        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * ff_multiplier,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(block, num_layers=layers)
        self.final_norm = nn.LayerNorm(dim)
        # Deliberately absent: nn.Linear(dim, vocab_size)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(batch, context, dim) -> (batch, dim) predicted next embedding."""
        length = embeddings.shape[1]
        hidden = embeddings + self.position[:, :length]

        causal = torch.triu(
            torch.full((length, length), float("-inf"), device=embeddings.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal)
        return self.final_norm(hidden[:, -1])

    def parameter_report(self, vocab_size: int) -> dict:
        """Actual parameters vs. what a conventional head would have cost."""
        actual = sum(p.numel() for p in self.parameters())
        head = vocab_size * self.dim
        return {
            "model_parameters": actual,
            "output_head_parameters_avoided": head,
            "conventional_total": actual + head,
            "reduction_fraction": head / (actual + head),
            "vocab_size": vocab_size,
        }


def cosine_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity. Scale-invariant, matching the decoder."""
    return (1.0 - F.cosine_similarity(predicted, target, dim=-1)).mean()


def relative_error(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """||predicted - target|| / ||target||, after matching scale.

    This is the number that decides whether problem 5 is viable. The
    noise experiments establish a corruption threshold the inverse map
    survives; this measures where a real trained model actually sits
    relative to that threshold. Magnitude is matched first because the
    decoder ignores scale, so an unmatched norm would inflate the error
    with a difference the decoder never sees.
    """
    scale = (
        (predicted * target).sum(-1, keepdim=True)
        / (predicted * predicted).sum(-1, keepdim=True).clamp_min(1e-12)
    )
    aligned = predicted * scale
    return (
        torch.linalg.norm(aligned - target, dim=-1)
        / torch.linalg.norm(target, dim=-1).clamp_min(1e-12)
    )
