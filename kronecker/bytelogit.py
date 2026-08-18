"""
The proposed architecture: a byte-logit head.

exp3 and exp4 established why emitting a d-dimensional vector cannot
work. Under uncertainty the loss-minimising point is the average of the
plausible next words, and an average of embeddings is not an embedding.
Fix A could not help (decoding already is a projection). Fix B made it
worse (ranking objectives do not constrain the output to be valid).

Both failures share a cause: a point in R^d has no way to say "probably
p, possibly b" without producing something that is neither.

So stop asking for a point.

The manifold of valid embeddings is exactly parameterised by
32 positions x 256 byte values. A model that emits logits over that grid
gets both properties the vector formulation cannot have at once:

  - **Uncertainty is representable.** Position 3 can hold a genuine
    distribution over p and b. Nothing is averaged into a third thing.
    This is the property vector regression lacks and the one that
    actually fixes the superposition failure.

  - **Outputs were valid in every run measured here -- but NOT by
    construction.** An earlier version of this docstring claimed that
    any argmax over byte logits is a well-formed byte string
    automatically. That is wrong. UTF-8 is a grammar: a lead byte
    constrains what may follow it, and independent per-position argmax
    cannot enforce that, since position 2 does not know what position 1
    chose. `E0 A4 41` and `C3 41` are both reachable by independent
    argmax and both fail to decode.

    The measured 0% invalid rate is empirical at this scale, plausibly
    because a 5,000-word vocabulary gives sharply peaked per-slot
    distributions. `constrained.py` implements the actual fix --
    decoding position by position with each step masked by what came
    before, which makes validity structural.

    Note also that valid UTF-8 is not the same as a legal token: `qxzf`
    decodes fine and is not a word.

And the original prize survives: the head is d x 8192, constant in
vocabulary size.

    vocabulary     conventional (V x d, d=8096)    byte-logit (d x 8192)
    131,072                     1.06B                        66.3M
    1,000,000                   8.10B                        66.3M

The length problem does not disappear -- it becomes explicit. Rather
than inferring L from where peak scores fall off a cliff, position 0 of
a 33rd "length" channel is predicted directly, so the model states how
many bytes it means. That is strictly more information than the vector
formulation carried.

STATUS: validated. `experiments/exp5_bytelogit.py` and `exp6_scaled.py`
train this and report the same end-to-end metrics as exp3/exp4, so the
numbers are directly comparable: 0% invalid utf-8 at every checkpoint
against ~51% for the vector head, under matched budgets.

That 0% is measured, not guaranteed -- see the second bullet above.
`experiments/exp7_constrained.py` reports both decoders side by side
along with token-probability metrics, which show the model learning
(28x over uniform) where exact-match reporting showed almost nothing.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_VALUES = 256


class ByteLogitTransformer(nn.Module):
    """Transformer emitting per-position byte distributions, not a vector.

    Input embeddings are still the deterministic Kronecker forward map --
    unchanged, untrained, table-free. Only the output side differs.
    """

    def __init__(self, dim: int, context: int = 12, layers: int = 2,
                 heads: int = 8, ff_multiplier: int = 2,
                 max_positions: int = 32):
        super().__init__()
        self.dim = dim
        self.context = context
        self.max_positions = max_positions

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

        # The head. d x (32 * 256), independent of vocabulary size.
        self.byte_head = nn.Linear(dim, max_positions * BYTE_VALUES)
        # Predicted byte count, so length is stated rather than inferred.
        self.length_head = nn.Linear(dim, max_positions + 1)

    def forward(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(batch, context, dim) -> byte logits (batch, 32, 256), length logits."""
        length = embeddings.shape[1]
        hidden = embeddings + self.position[:, :length]

        causal = torch.triu(
            torch.full((length, length), float("-inf"), device=embeddings.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal)
        hidden = self.final_norm(hidden[:, -1])

        byte_logits = self.byte_head(hidden).view(
            -1, self.max_positions, BYTE_VALUES)
        return byte_logits, self.length_head(hidden)

    def parameter_report(self, vocab_size: int) -> dict:
        actual = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.byte_head.parameters())
        conventional_head = vocab_size * self.dim
        return {
            "model_parameters": actual,
            "byte_head_parameters": head,
            "conventional_head_parameters": conventional_head,
            "head_reduction_factor": conventional_head / max(head, 1),
            "vocab_size": vocab_size,
        }


def byte_loss(byte_logits: torch.Tensor, length_logits: torch.Tensor,
              target_bytes: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    """Cross-entropy per position, plus length.

    Positions past a word's true length are masked out rather than
    trained toward a padding symbol -- the length head already carries
    that information, and training the byte head on padding would spend
    capacity teaching it to predict a value the decoder never reads.
    """
    batch, positions, _ = byte_logits.shape
    mask = (torch.arange(positions, device=byte_logits.device)[None, :]
            < target_lengths[:, None])

    per_position = F.cross_entropy(
        byte_logits.reshape(-1, BYTE_VALUES),
        target_bytes.reshape(-1),
        reduction="none",
    ).view(batch, positions)

    byte_term = (per_position * mask).sum() / mask.sum().clamp_min(1)
    length_term = F.cross_entropy(length_logits, target_lengths)
    return byte_term + length_term


def decode_bytes(byte_logits: torch.Tensor,
                 length_logits: torch.Tensor) -> list[bytes]:
    """Read words off the logits. No vocabulary table, no search."""
    lengths = length_logits.argmax(-1)
    choices = byte_logits.argmax(-1)
    out = []
    for row in range(len(choices)):
        count = int(lengths[row])
        out.append(bytes(int(b) for b in choices[row][:count]))
    return out


def decode_words(byte_logits: torch.Tensor,
                 length_logits: torch.Tensor) -> list[str | None]:
    """As decode_bytes, but None where the bytes are not valid UTF-8.

    Kept distinct so the invalid-UTF-8 rate stays directly comparable to
    the same figure in exp3 and exp4, where it was the clearest symptom
    of the superposition failure.
    """
    out = []
    for recovered in decode_bytes(byte_logits, length_logits):
        try:
            out.append(recovered.decode("utf-8"))
        except UnicodeDecodeError:
            out.append(None)
    return out
