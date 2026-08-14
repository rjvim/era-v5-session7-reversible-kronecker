"""
Experiment 2: how much corruption the inverse map survives, and why the
obvious way of measuring that gives the wrong answer.

Two noise models are compared, because the difference between them turns
out to be the whole point.

ISOTROPIC -- a random Gaussian direction added to the vector. This is
what anyone testing robustness reaches for first, and in high dimension
it is close to free: a random vector's projection onto any one codebook
direction is diluted by sqrt(d), so at d=8096 the decoder barely notices
noise larger than the signal itself. Reporting only this number would
support a claim of near-unbreakable reversibility that does not survive
contact with a real model.

STRUCTURED -- a random combination of other codebook directions. This
has the shape a model's output actually has: it lives in the same
subspace the decoder reads from, so it competes with the signal instead
of spreading harmlessly across 8096 dimensions.

The dimension sweep then measures where the 50% accuracy threshold sits
as d varies, by bisection. The expectation going in was that tolerance
scales as sqrt(d). It does not -- it collapses below d=512 and saturates
above d=1024, which is worth knowing before anyone spends model width
buying robustness that is not for sale.

Caveat carried into the README: structured noise is a proxy. It has the
right shape but is not a real error distribution. exp3 measures the real
one, and exp3 is where this framing turns out to be insufficient.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kronecker"))

from codebook import Codebook  # noqa: E402
from forward import encode     # noqa: E402

PROBE_WORDS = ["a", "tree", "apple", "apples", "banana", "elephant",
               "internationalization", "नमस्ते", "हिन्दी", "भारत"]
CURVE_WORDS = ["apple", "internationalization", "नमस्ते"]
CURVE_DIM = 8096
SWEEP_DIMS = [128, 256, 512, 1024, 2048, 4096]
TRIALS = 20
SEED = 0

OUT = ROOT / "submission_artifacts" / "results"


class Probe:
    """Encoder + batched decoder sharing one stacked codebook."""

    def __init__(self, dim: int, words: list[str]):
        self.codebook = Codebook(dim=dim)
        self.stack = np.stack(
            [self.codebook.directions(p) for p in range(self.codebook.max_positions)])
        self.words = words
        self.truth = [w.encode("utf-8") for w in words]
        self.vectors = np.stack(
            [encode(w, self.codebook).vector for w in words]).astype(np.float64)
        self.norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def decode_bytes(self, vectors: np.ndarray) -> list[bytes]:
        scores = np.einsum("pbd,nd->npb", self.stack, vectors)
        best_byte, best_score = scores.argmax(2), scores.max(2)
        out = []
        for row in range(len(vectors)):
            peaks = best_score[row]
            ratios = peaks[:-1] / np.maximum(peaks[1:], 1e-12)
            length = int(np.argmax(ratios)) + 1
            out.append(bytes(int(b) for b in best_byte[row][:length]))
        return out

    def isotropic(self, rng, magnitude: float) -> np.ndarray:
        noise = rng.standard_normal(self.vectors.shape)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        return self.vectors + magnitude * self.norms * noise

    def structured(self, rng, magnitude: float, components: int = 12) -> np.ndarray:
        noise = np.zeros_like(self.vectors)
        positions, bytes_ = self.stack.shape[0], self.stack.shape[1]
        for row in range(len(self.vectors)):
            for _ in range(components):
                p = rng.integers(0, positions)
                b = rng.integers(0, bytes_)
                noise[row] += rng.standard_normal() * self.stack[p, b]
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        return self.vectors + magnitude * self.norms * noise

    def accuracy(self, rng, kind: str, magnitude: float, trials: int) -> float:
        hits = total = 0
        for _ in range(trials):
            corrupted = (self.isotropic(rng, magnitude) if kind == "isotropic"
                         else self.structured(rng, magnitude))
            for got, want in zip(self.decode_bytes(corrupted), self.truth):
                total += 1
                hits += int(got == want)
        return hits / max(total, 1)


def threshold_by_bisection(probe: Probe, rng, kind: str,
                           trials: int = 12, steps: int = 14) -> float:
    """Magnitude at which accuracy crosses 50%, found geometrically."""
    low, high = 0.01, 20.0
    for _ in range(steps):
        mid = (low * high) ** 0.5
        if probe.accuracy(rng, kind, mid, trials) >= 0.5:
            low = mid
        else:
            high = mid
    return (low * high) ** 0.5


def main():
    rng = np.random.default_rng(SEED)
    results: dict = {"config": {"trials": TRIALS, "seed": SEED,
                                "curve_dim": CURVE_DIM}}

    print(f"noise curves at d={CURVE_DIM}")
    curve_probe = Probe(CURVE_DIM, CURVE_WORDS)
    curves: dict = {}
    for kind, magnitudes in (
        ("isotropic", [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]),
        ("structured", [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]),
    ):
        curves[kind] = {}
        print(f"  {kind}:")
        for magnitude in magnitudes:
            accuracy = curve_probe.accuracy(rng, kind, magnitude, TRIALS)
            curves[kind][str(magnitude)] = accuracy
            print(f"    {magnitude:>5}x  {accuracy * 100:5.1f}%")
    results["curves"] = curves

    print("\ndimension sweep (structured noise, 50% threshold)")
    sweep: dict = {}
    for dim in SWEEP_DIMS:
        probe = Probe(dim, PROBE_WORDS[:6])
        threshold = threshold_by_bisection(probe, rng, "structured")
        sweep[str(dim)] = threshold
        print(f"  dim {dim:>5}  threshold {threshold:.3f}")
    results["dimension_sweep"] = sweep

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "noise.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT / 'noise.json'}")


if __name__ == "__main__":
    main()
