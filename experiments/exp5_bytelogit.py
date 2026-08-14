"""
Experiment 5: does the byte-logit head fix the superposition failure?

Reports the same two metrics as exp3 and exp4 -- exact words recovered
end-to-end, and the fraction of outputs that are not valid UTF-8 -- so
the three architectures are directly comparable:

    exp3  vector head, cosine regression      0.5% exact,  49.8% invalid
    exp4  vector head, InfoNCE                0.0% exact, 100.0% invalid
    exp5  byte-logit head                     this run

The prediction under test: uncertainty stops producing invalid output,
because every argmax over byte logits is a well-formed byte string by
construction. If the invalid-UTF-8 rate does not collapse toward zero,
the diagnosis in README section 4 is wrong and should be revised.

Note on what a fair comparison means here. The byte-logit model has a
larger head (d x 8192) than the vector model (none), so it is not a
parameter-matched comparison. It is matched on the thing that matters
for problem 5: neither model has a V x d output matrix, and neither
model's size grows with vocabulary. Parameter counts for both are
reported so the difference is visible rather than buried.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kronecker"))

from bytelogit import (ByteLogitTransformer, byte_loss,  # noqa: E402
                       decode_words)
from codebook import Codebook  # noqa: E402
from forward import encode     # noqa: E402
from vocab import build_vocab  # noqa: E402

DIM = 512
CONTEXT = 8
LAYERS = 2
HEADS = 8
STEPS = 400
BATCH = 32
LR = 5e-4
PER_LANE = 800
SEED = 7
EVAL_SAMPLES = 128
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


def byte_targets(words: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-word byte grid and true byte count."""
    grid = torch.zeros(len(words), MAX_POSITIONS, dtype=torch.long)
    lengths = torch.zeros(len(words), dtype=torch.long)
    for row, word in enumerate(words):
        raw = word.encode("utf-8")[:MAX_POSITIONS]
        lengths[row] = len(raw)
        for position, byte_value in enumerate(raw):
            grid[row, position] = byte_value
    return grid, lengths


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
    report = model.parameter_report(vocab_size=131072)
    print(f"vocab {len(words)}  dim {DIM}  sequences {len(inputs)}")
    print(f"model {report['model_parameters']:,} params, "
          f"byte head {report['byte_head_parameters']:,}")
    print(f"  a conventional V x d head at V=131,072 would be "
          f"{report['conventional_head_parameters']:,} "
          f"({report['head_reduction_factor']:.0f}x larger)", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    for step in range(STEPS):
        pick = rng.choice(train_idx, BATCH)
        target_ids = torch.from_numpy(targets[pick])
        byte_logits, length_logits = model(
            table_t[torch.from_numpy(inputs[pick])])
        loss = byte_loss(byte_logits, length_logits,
                         grid[target_ids], lengths[target_ids])
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"    step {step:4d}  loss {float(loss.detach()):.4f}  "
                  f"({time.time() - started:.0f}s)", flush=True)

    model.eval()
    exact = invalid = 0
    truth = [words[i] for i in targets[eval_idx[:EVAL_SAMPLES]]]
    with torch.no_grad():
        recovered: list[str | None] = []
        sample = eval_idx[:EVAL_SAMPLES]
        for start in range(0, len(sample), 64):
            pick = sample[start:start + 64]
            byte_logits, length_logits = model(
                table_t[torch.from_numpy(inputs[pick])])
            recovered.extend(decode_words(byte_logits, length_logits))

    for got, want in zip(recovered, truth):
        if got is None:
            invalid += 1
        elif got == want:
            exact += 1

    result = {
        "config": {"dim": DIM, "context": CONTEXT, "layers": LAYERS,
                   "steps": STEPS, "batch": BATCH, "lr": LR,
                   "vocab_size": len(words), "seed": SEED,
                   "codebook_fingerprint": codebook.fingerprint()},
        "parameters": report,
        "headless_decode": {
            "attempted": len(truth),
            "exact_word_recovered": exact,
            "accuracy": exact / max(len(truth), 1),
            "invalid_utf8": invalid,
            "invalid_fraction": invalid / max(len(truth), 1),
        },
        "runtime_seconds": time.time() - started,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bytelogit.json").write_text(json.dumps(result, indent=2))

    print(f"\n  exact: {exact}/{len(truth)} "
          f"({exact / len(truth) * 100:.1f}%)")
    print(f"  invalid utf-8: {invalid}/{len(truth)} "
          f"({invalid / len(truth) * 100:.1f}%)")


if __name__ == "__main__":
    main()
