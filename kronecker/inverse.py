"""
Inverse pass: vector -> word.

This is the part V4 does not have, and the reason problem 5 is open.

The method follows directly from the codebook's construction. A word
vector is

    v = (1/L) * sum_i  C[i, b_i]

and every C[i, b] is a unit vector, near-orthogonal to every other. So
projecting v onto the 256 candidate directions at position i gives:

  - the true byte b_i:      ~1/L
  - every other byte:       cross-talk, O(1/(L*sqrt(d)))
  - any position i >= L:    cross-talk only, no active slot

Which yields the whole algorithm: score all 256 candidates at each
position, take the argmax, and stop when the winning score no longer
stands out from the field.

Two things make this less trivial than it sounds.

**Length is not known in advance.** The vector carries no explicit
length field. It is recovered by watching the peak score fall off:
active positions peak near 1/L, inactive ones peak at the cross-talk
floor, and the ratio between the two is ~sqrt(d)/1 -- a large enough gap
that a relative threshold separates them cleanly at zero noise. Under
noise this is the first thing to break, which is why `decode` returns
the margin it saw rather than only the answer.

**A recovered byte string need not be valid UTF-8.** Under noise, a
single wrong byte can produce a sequence that decodes to nothing. That
is reported as a distinct failure mode (`invalid_utf8`) rather than
being folded into "wrong word", because the two say different things
about where the method breaks -- and for Devanagari, where one character
spans three bytes, they behave very differently.

Nothing here is trained. There is no learned decoder head, which is the
entire point of the problem: if this map works, the model's final
V x d output matrix can be deleted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from codebook import Codebook

# A position counts as active when its peak score exceeds this fraction
# of the running maximum peak. Chosen from the zero-noise separation
# measured in experiments/exp1_clean.py, not tuned per test case.
# Retained only for reference: the fixed-fraction rule this replaced.
# See _infer_length for why it was dropped.
ACTIVE_THRESHOLD = 0.35


@dataclass(frozen=True)
class DecodedWord:
    """A reconstruction and the evidence behind it."""
    word: str | None
    recovered_bytes: bytes
    predicted_length: int
    peak_scores: list[float]
    length_margin: float
    invalid_utf8: bool

    @property
    def succeeded(self) -> bool:
        return self.word is not None


def _score_all_positions(vector: np.ndarray, codebook: Codebook) -> tuple[np.ndarray, np.ndarray]:
    """Best byte and its score at every position.

    Returns (best_byte per position, best_score per position).
    """
    positions = codebook.max_positions
    best_byte = np.zeros(positions, dtype=np.int32)
    best_score = np.zeros(positions, dtype=np.float64)

    for position in range(positions):
        scores = codebook.directions(position) @ vector
        winner = int(np.argmax(scores))
        best_byte[position] = winner
        best_score[position] = float(scores[winner])

    return best_byte, best_score


def _infer_length(best_score: np.ndarray) -> tuple[int, float]:
    """Recover L from where the peak scores fall off a cliff.

    Active positions all sit near 1/L. Inactive ones sit at the
    cross-talk floor, far below. The cut is taken at the largest RATIO
    drop between consecutive peaks, and that ratio is returned as the
    margin -- a margin near 1.0 means the cliff had eroded and the
    length is a guess.

    An earlier version cut at a fixed fraction of the maximum peak
    (ACTIVE_THRESHOLD). That works at d=8096 but over-runs by a byte on
    18-byte Devanagari words once cross-talk rises at lower dimension --
    caught by test_inverse_is_exact_at_zero_noise, not by inspection.
    Measured over the same seven-word probe:

        dim    fixed threshold    max ratio drop
        512          4/7               7/7
        1024         6/7               7/7
        2048         7/7               7/7
        8096         7/7               7/7

    The ratio-drop rule is also parameter-free, so the length cut no
    longer depends on a constant tuned at one dimension.
    """
    if best_score.max() <= 0:
        return 0, 0.0
    if len(best_score) < 2:
        return int(best_score[0] > 0), float("inf")

    ratios = best_score[:-1] / np.maximum(best_score[1:], 1e-12)
    cut = int(np.argmax(ratios))
    return cut + 1, float(ratios[cut])


def decode(vector: np.ndarray, codebook: Codebook) -> DecodedWord:
    """Reconstruct a word from its vector."""
    vector = np.asarray(vector, dtype=np.float64)
    best_byte, best_score = _score_all_positions(vector, codebook)
    length, margin = _infer_length(best_score)

    recovered = bytes(int(b) for b in best_byte[:length])

    try:
        word = recovered.decode("utf-8")
        invalid = False
    except UnicodeDecodeError:
        word = None
        invalid = True

    return DecodedWord(
        word=word,
        recovered_bytes=recovered,
        predicted_length=length,
        peak_scores=[float(s) for s in best_score],
        length_margin=float(margin),
        invalid_utf8=invalid,
    )


def roundtrip(word: str, codebook: Codebook) -> tuple[bool, DecodedWord]:
    """Encode then decode. Convenience for tests and experiments."""
    from forward import encode

    encoded = encode(word, codebook)
    decoded = decode(encoded.vector, codebook)
    # A truncated word can only ever round-trip to its truncated form;
    # comparing against the original would report a failure that is
    # really a capacity limit, so compare against what was encodable.
    expected = word.encode("utf-8")[:codebook.max_positions]
    return decoded.recovered_bytes == expected, decoded
