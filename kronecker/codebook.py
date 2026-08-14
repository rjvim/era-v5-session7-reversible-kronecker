"""
The codebook: a fixed, deterministic set of directions, one per
(position, byte-value) pair.

This is the single design decision the whole reversibility result rests
on, so it is worth stating plainly why it looks like this.

Original Kronecker (V4) builds a word vector as:

    v(word) = (1/L) * sum_i  P_i * M[byte_i]

where M is a fixed random matrix and P_i is a position embedding that is
ADDED to the character's vector. Adding position after projection means
the same byte at two different positions produces two vectors that share
a large common component. That is fine going forward -- the trainable
layer downstream can cope -- but it destroys the information needed to
run the map backwards: once summed, there is no way to attribute a
component of v back to a specific (position, byte) pair.

This module changes exactly one thing: instead of `project, then add
position`, every (position, byte) pair gets its OWN independent
direction, drawn once from a fixed seed and never changed:

    v(word) = (1/L) * sum_i  C[i, byte_i]

Two consequences:

  1. Forward is still fully deterministic and still needs no stored
     embedding table -- C is regenerated from a seed, not learned.

  2. Backward becomes a projection problem. In high dimension, random
     Gaussian directions are very nearly orthogonal, so C[i, b] . C[j, b']
     is ~0 for any pair that isn't identical. Taking the inner product of
     v against every candidate direction at position i therefore
     recovers byte_i by argmax, with the other L-1 characters
     contributing only cross-talk.

The cross-talk is not zero, and quantifying it is the point of the
experiments. For d dimensions and L active characters, the expected
inner product between two distinct unit directions is O(1/sqrt(d)),
while the signal at an active slot is 1/L. The margin therefore scales
as sqrt(d)/L -- which is why d=8096 with L<=32 has room, and why the
noise experiments are where this either survives or doesn't.

Nothing here is trained. `Codebook(seed=...)` is reproducible across
processes and machines; `test_invariants.py` asserts that.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Kronecker V4's limits, kept identical so results are comparable.
MAX_POSITIONS = 32
BYTE_VALUES = 256
DEFAULT_DIM = 8096
DEFAULT_SEED = 20260808  # session 7 date; arbitrary but fixed forever


class Codebook:
    """Fixed (position, byte) -> unit direction map.

    Shape is (MAX_POSITIONS, BYTE_VALUES, dim). At the default dim that
    is 32 * 256 * 8096 floats = ~265M values, which is large to hold
    densely, so directions are generated on demand per position and
    cached. Position-major access is the only pattern either the forward
    or the inverse pass needs.
    """

    def __init__(self, dim: int = DEFAULT_DIM, seed: int = DEFAULT_SEED,
                 max_positions: int = MAX_POSITIONS):
        self.dim = int(dim)
        self.seed = int(seed)
        self.max_positions = int(max_positions)
        self._cache: dict[int, np.ndarray] = {}

    # -- generation ---------------------------------------------------

    def _position_seed(self, position: int) -> int:
        """Derive a per-position seed from the master seed.

        Hashing rather than `seed + position` so that two Codebooks with
        adjacent master seeds don't share position blocks.
        """
        key = f"{self.seed}:{position}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")

    def directions(self, position: int) -> np.ndarray:
        """All 256 unit directions for one position. Shape (256, dim)."""
        if position < 0 or position >= self.max_positions:
            raise IndexError(
                f"position {position} outside [0, {self.max_positions})"
            )
        cached = self._cache.get(position)
        if cached is not None:
            return cached

        rng = np.random.default_rng(self._position_seed(position))
        block = rng.standard_normal((BYTE_VALUES, self.dim), dtype=np.float64)
        # Unit-normalise: makes the inner product a cosine, so the
        # active-slot signal is exactly 1/L and thresholds are
        # interpretable without rescaling.
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        block = block.astype(np.float32)

        self._cache[position] = block
        return block

    def direction(self, position: int, byte_value: int) -> np.ndarray:
        """The single unit direction for one (position, byte) pair."""
        if byte_value < 0 or byte_value >= BYTE_VALUES:
            raise IndexError(f"byte value {byte_value} outside [0, 256)")
        return self.directions(position)[byte_value]

    # -- diagnostics --------------------------------------------------

    def fingerprint(self) -> str:
        """Stable hash of the codebook's identity.

        Recorded alongside every result so a reported number can never be
        silently attributed to a different codebook. Covers the
        parameters plus the actual generated content of two positions.
        """
        h = hashlib.sha256()
        h.update(f"{self.dim}:{self.seed}:{self.max_positions}".encode())
        for position in (0, self.max_positions - 1):
            h.update(self.directions(position).tobytes())
        return h.hexdigest()[:16]

    def max_offdiagonal(self, position: int) -> float:
        """Largest |cosine| between two distinct directions at one position.

        This is the empirical cross-talk floor. If it approaches 1/L for
        the word lengths in use, reconstruction is not going to hold, and
        the experiments should say so rather than the README claiming
        orthogonality it doesn't have.
        """
        block = self.directions(position)
        gram = block @ block.T
        np.fill_diagonal(gram, 0.0)
        return float(np.abs(gram).max())
