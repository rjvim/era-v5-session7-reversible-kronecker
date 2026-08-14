"""
Forward pass: word -> fixed 8096-dimensional vector.

Deterministic, untrained, and requiring no stored embedding table. The
same word produces byte-identical output in every process, forever.

Encoding, in order:

  1. UTF-8 encode the word to bytes. Latin characters cost 1 byte;
     Devanagari costs 3. That asymmetry is inherited from UTF-8 itself
     and is the reason `capacity_report` exists -- the 32-slot budget
     buys ~32 Latin characters but only ~10 Devanagari ones, and the
     experiments report Latin and Devanagari separately rather than
     averaging the difference away.

  2. Each byte at position i contributes the codebook direction
     C[i, byte_i].

  3. Sum, then divide by the byte count L.

Step 3 is where the reversibility problem lives. Averaging makes the
active-slot signal 1/L rather than 1, so a long word's signal sits
closer to the cross-talk floor than a short word's does. The division is
kept anyway, because it is what V4 does and dropping it would make these
results incomparable to his. `INVERSE` recovers L before decoding rather
than assuming it.

Words longer than the byte budget are truncated, and `encode` reports
that it happened rather than silently discarding the tail.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codebook import MAX_POSITIONS, Codebook


@dataclass(frozen=True)
class EncodedWord:
    """A forward pass and everything needed to audit it."""
    word: str
    vector: np.ndarray
    used_bytes: int
    total_bytes: int

    @property
    def truncated(self) -> bool:
        return self.total_bytes > self.used_bytes


def encode(word: str, codebook: Codebook) -> EncodedWord:
    """Encode one word to its fixed vector."""
    raw = word.encode("utf-8")
    budget = codebook.max_positions
    used = raw[:budget]

    if not used:
        # Empty string has no active slots. Returning zeros keeps the
        # function total; the inverse detects this as length 0.
        return EncodedWord(
            word=word,
            vector=np.zeros(codebook.dim, dtype=np.float32),
            used_bytes=0,
            total_bytes=len(raw),
        )

    accumulator = np.zeros(codebook.dim, dtype=np.float64)
    for position, byte_value in enumerate(used):
        accumulator += codebook.direction(position, byte_value)
    accumulator /= len(used)

    return EncodedWord(
        word=word,
        vector=accumulator.astype(np.float32),
        used_bytes=len(used),
        total_bytes=len(raw),
    )


def encode_many(words: list[str], codebook: Codebook) -> np.ndarray:
    """Encode a list of words. Shape (len(words), dim)."""
    out = np.zeros((len(words), codebook.dim), dtype=np.float32)
    for row, word in enumerate(words):
        out[row] = encode(word, codebook).vector
    return out


def capacity_report(word: str) -> dict:
    """How much of the 32-byte budget a word consumes, and why.

    Reported per word in the results so the Devanagari penalty is
    visible as a measured quantity rather than an assertion.
    """
    raw = word.encode("utf-8")
    return {
        "word": word,
        "characters": len(word),
        "utf8_bytes": len(raw),
        "bytes_per_character": (len(raw) / len(word)) if word else 0.0,
        "fits_in_budget": len(raw) <= MAX_POSITIONS,
        "budget": MAX_POSITIONS,
    }
