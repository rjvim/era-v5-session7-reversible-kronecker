"""
Experiment 7: constrained decoding, and metrics that are not exact-match.

Two revisions requested in review, both addressed here.

**Constrained decoding.** `bytelogit.py` originally claimed independent
per-position argmax always yields valid UTF-8. It does not.
`constrained.py` decodes left to right with each step masked by the
grammar state, so a lead byte forces its continuations. This experiment
reports both decoders on the same trained model, so the difference is
measured rather than argued.

Expect the gap to be small on this model -- the unconstrained decoder
already measured 0% invalid at this scale. That is the point. The
constrained decoder's guarantee holds for reasons that do not depend on
the model being well-trained, which is exactly what the earlier claim
lacked, and the accompanying invariant test stresses it against random
logits where the unconstrained version fails outright.

**Token-probability metrics.** Exact-match at a few percent says almost
nothing about whether the model is learning: a model ranking the correct
word second out of 5,000 scores identically to one ranking it last.
Reported instead:

  - probability assigned to the correct token
  - top-1 / top-5 / top-10 accuracy
  - perplexity over the byte factorisation

Scoring a token requires only its bytes, so ranking the full vocabulary
here uses the deterministic forward map and no output head -- the
V x d matrix stays deleted. Cost is O(V x 32) table lookups at eval
time only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kronecker"))

from bytelogit import (ByteLogitTransformer, byte_loss,  # noqa: E402
                       decode_words)
from codebook import Codebook          # noqa: E402
from constrained import decode_constrained_words  # noqa: E402
from forward import encode             # noqa: E402
from vocab import build_vocab          # noqa: E402

DIM = 512
CONTEXT = 8
LAYERS = 2
HEADS = 8
STEPS = 700
BATCH = 48
LR = 5e-4
PER_LANE = 800
SEED = 7
EVAL_SAMPLES = 256
MAX_POSITIONS = 32

SOURCES = ROOT.parent / "s6" / "data_pipeline" / "real_sources"
OUT = ROOT / "submission_artifacts" / "results"


def make_sequences(stream, index, context):
    ids = [index[w] for w in stream if w in index]
    arr = np.asarray(ids, dtype=np.int64)
    windows = len(arr) - context - 1
    if windows <= 0:
        return np.zeros((0, context), np.int64), np.zeros((0,), np.int64)
    inputs = np.lib.stride_tricks.sliding_window_view(arr[:-1], context)[:windows]
    return np.ascontiguousarray(inputs), np.ascontiguousarray(arr[context:context + windows])


def byte_targets(words):
    grid = torch.zeros(len(words), MAX_POSITIONS, dtype=torch.long)
    lengths = torch.zeros(len(words), dtype=torch.long)
    for row, word in enumerate(words):
        raw = word.encode("utf-8")[:MAX_POSITIONS]
        lengths[row] = len(raw)
        for position, byte_value in enumerate(raw):
            grid[row, position] = byte_value
    return grid, lengths


def score_vocabulary(byte_logits, length_logits, grid, lengths):
    """Log-probability of every vocabulary word under the byte factorisation.

    log P(word) = log P(length) + sum over positions of log P(byte).
    Needs each word's bytes, not a V x d matrix.
    """
    byte_logprobs = F.log_softmax(byte_logits, dim=-1)      # (B, 32, 256)
    length_logprobs = F.log_softmax(length_logits, dim=-1)  # (B, 33)

    batch = byte_logits.shape[0]
    vocab = grid.shape[0]
    totals = torch.zeros(batch, vocab)
    for position in range(MAX_POSITIONS):
        active = (lengths > position)             # (V,)
        if not active.any():
            break
        contribution = byte_logprobs[:, position][:, grid[:, position]]  # (B, V)
        totals += contribution * active.float()
    totals += length_logprobs[:, lengths]
    return totals


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    started = time.time()

    vocab = build_vocab(SOURCES, per_lane=PER_LANE)
    words = vocab["words"]
    codebook = Codebook(dim=DIM)
    table_t = torch.from_numpy(
        np.stack([encode(w, codebook).vector for w in words]))
    grid, lengths = byte_targets(words)

    en_in, en_tg = make_sequences(
        [w.lower() for w in vocab["english_stream"]], vocab["index"], CONTEXT)
    hi_in, hi_tg = make_sequences(vocab["hindi_stream"], vocab["index"], CONTEXT)
    inputs = np.concatenate([en_in, hi_in])
    targets = np.concatenate([en_tg, hi_tg])

    order = rng.permutation(len(inputs))
    split = int(len(inputs) * 0.9)
    train_idx, eval_idx = order[:split], order[split:]

    model = ByteLogitTransformer(dim=DIM, context=CONTEXT, layers=LAYERS,
                                 heads=HEADS, max_positions=MAX_POSITIONS)
    print(f"vocab {len(words)}  dim {DIM}  sequences {len(inputs)}", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS):
        pick = rng.choice(train_idx, BATCH)
        target_ids = torch.from_numpy(targets[pick])
        byte_logits, length_logits = model(table_t[torch.from_numpy(inputs[pick])])
        loss = byte_loss(byte_logits, length_logits,
                         grid[target_ids], lengths[target_ids])
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"    step {step:4d}  {float(loss.detach()):.4f}  "
                  f"({time.time() - started:.0f}s)", flush=True)

    model.eval()
    sample = eval_idx[:EVAL_SAMPLES]
    truth = [words[i] for i in targets[sample]]
    naive, constrained = [], []
    ranks, probabilities, nats = [], [], []

    with torch.no_grad():
        for start in range(0, len(sample), 64):
            pick = sample[start:start + 64]
            target_ids = torch.from_numpy(targets[pick])
            byte_logits, length_logits = model(
                table_t[torch.from_numpy(inputs[pick])])

            naive.extend(decode_words(byte_logits, length_logits))
            constrained.extend(decode_constrained_words(byte_logits, length_logits))

            scores = score_vocabulary(byte_logits, length_logits, grid, lengths)
            probs = F.softmax(scores, dim=-1)
            for row, target in enumerate(target_ids):
                probabilities.append(float(probs[row, target]))
                ranks.append(int((scores[row] > scores[row, target]).sum()) + 1)
                nats.append(-float(torch.log(probs[row, target].clamp_min(1e-12))))

    def rate(decoded):
        exact = sum(1 for got, want in zip(decoded, truth) if got == want)
        invalid = sum(1 for got in decoded if got is None)
        return {"exact": exact, "accuracy": exact / len(truth),
                "invalid_utf8": invalid, "invalid_fraction": invalid / len(truth)}

    ranks_arr = np.asarray(ranks)
    result = {
        "config": {"dim": DIM, "steps": STEPS, "batch": BATCH,
                   "vocab_size": len(words), "seed": SEED,
                   "eval_samples": len(truth)},
        "decoding": {"unconstrained": rate(naive), "constrained": rate(constrained)},
        "token_probability": {
            "mean_prob_of_correct_token": float(np.mean(probabilities)),
            "uniform_baseline": 1.0 / len(words),
            "lift_over_uniform": float(np.mean(probabilities) * len(words)),
            "median_rank_of_correct_token": float(np.median(ranks_arr)),
            "top1": float((ranks_arr <= 1).mean()),
            "top5": float((ranks_arr <= 5).mean()),
            "top10": float((ranks_arr <= 10).mean()),
            "top100": float((ranks_arr <= 100).mean()),
            "perplexity": float(np.exp(np.mean(nats))),
        },
        "runtime_seconds": time.time() - started,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "constrained_and_probability.json").write_text(json.dumps(result, indent=2))

    d = result["decoding"]
    t = result["token_probability"]
    print(f"\n  unconstrained: exact {d['unconstrained']['accuracy'] * 100:.1f}%  "
          f"invalid {d['unconstrained']['invalid_fraction'] * 100:.1f}%")
    print(f"  constrained:   exact {d['constrained']['accuracy'] * 100:.1f}%  "
          f"invalid {d['constrained']['invalid_fraction'] * 100:.1f}%")
    print(f"\n  P(correct token) {t['mean_prob_of_correct_token']:.5f}  "
          f"vs uniform {t['uniform_baseline']:.5f}  "
          f"({t['lift_over_uniform']:.1f}x)")
    print(f"  median rank {t['median_rank_of_correct_token']:.0f} of {len(words)}")
    print(f"  top1 {t['top1'] * 100:.1f}%  top5 {t['top5'] * 100:.1f}%  "
          f"top10 {t['top10'] * 100:.1f}%  top100 {t['top100'] * 100:.1f}%")
    print(f"  perplexity {t['perplexity']:.1f}")


if __name__ == "__main__":
    main()
