"""
Experiment 1: clean round-trip across the full vocabulary.

The foundation everything else stands on. If the inverse map is not
exact at zero noise, no later result means anything, so this runs over
every word rather than a sample, and reports Latin and Devanagari
separately -- reversibility degrades with utf-8 byte count, and averaging
the two scripts together would hide exactly the effect worth seeing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kronecker"))

from codebook import Codebook          # noqa: E402
from forward import encode             # noqa: E402
from vocab import build_vocab, byte_length_stats  # noqa: E402

DIM = 8096
PER_LANE = 1500
SOURCES = ROOT.parent / "s6" / "data_pipeline" / "real_sources"
OUT = ROOT / "submission_artifacts" / "results"


def main():
    vocab = build_vocab(SOURCES, per_lane=PER_LANE)
    words, lanes = vocab["words"], vocab["lanes"]
    codebook = Codebook(dim=DIM)
    stack = np.stack([codebook.directions(p) for p in range(codebook.max_positions)])

    print(f"vocab {len(words)}  dim {DIM}  fingerprint {codebook.fingerprint()}")

    per_lane_hits: dict[str, list[int]] = {}
    margins: list[float] = []

    for start in range(0, len(words), 64):
        chunk = words[start:start + 64]
        vectors = np.stack([encode(w, codebook).vector for w in chunk]).astype(np.float64)
        scores = np.einsum("pbd,nd->npb", stack, vectors)
        best_byte, best_score = scores.argmax(2), scores.max(2)

        for row, word in enumerate(chunk):
            reference = best_score[row].max()
            length = 0
            for position in range(stack.shape[0]):
                if best_score[row][position] < 0.35 * reference:
                    break
                length += 1
            recovered = bytes(int(b) for b in best_byte[row][:length])
            lane = lanes[start + row]
            per_lane_hits.setdefault(lane, []).append(
                int(recovered == word.encode("utf-8")))
            if 0 < length < stack.shape[0]:
                margins.append(float(best_score[row][length - 1] /
                                     max(best_score[row][length], 1e-12)))

    result = {
        "config": {"dim": DIM, "vocab_size": len(words),
                   "codebook_fingerprint": codebook.fingerprint()},
        "vocab_byte_stats": byte_length_stats(words, lanes),
        "exact_roundtrip": {
            lane: {"words": len(hits), "exact": sum(hits),
                   "accuracy": sum(hits) / len(hits)}
            for lane, hits in per_lane_hits.items()
        },
        "length_margin": {
            "mean": float(np.mean(margins)),
            "min": float(np.min(margins)),
        },
        "max_offdiagonal_position_0": codebook.max_offdiagonal(0),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clean.json").write_text(json.dumps(result, indent=2))

    for lane, r in result["exact_roundtrip"].items():
        print(f"  {lane}: {r['exact']}/{r['words']} exact ({r['accuracy'] * 100:.2f}%)")
    print(f"  length margin: mean {result['length_margin']['mean']:.1f} "
          f"min {result['length_margin']['min']:.1f}")


if __name__ == "__main__":
    main()
