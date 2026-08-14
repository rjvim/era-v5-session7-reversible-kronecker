"""
Experiment 3: what error does a real model actually make?

The noise experiments establish how much corruption the inverse map
survives. That number is only useful next to this one: the error a
trained model actually produces. If real error sits below the threshold,
problem 5 is viable. If above, it isn't, and the gap is the finding.

Runs at reduced dimension by necessity (1 CPU, 3GB). Robustness scales
with sqrt(d), so the reduced-dimension result is a LOWER bound on what
d=8096 would give -- stated explicitly rather than extrapolated
silently. exp4 measures the scaling directly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kronecker"))

from codebook import Codebook  # noqa: E402
from forward import encode  # noqa: E402
from model import HeadlessTransformer, cosine_loss, relative_error  # noqa: E402
from vocab import build_vocab, byte_length_stats  # noqa: E402

DIM = 512
CONTEXT = 12
LAYERS = 2
HEADS = 8
STEPS = 1200
BATCH = 32
LR = 3e-4
PER_LANE = 1500
SEED = 7

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT.parent / "s6" / "data_pipeline" / "real_sources"
OUT = ROOT / "submission_artifacts" / "results"


def make_sequences(stream, index, context):
    ids = [index[w] for w in stream if w in index]
    arr = np.asarray(ids, dtype=np.int64)
    windows = len(arr) - context - 1
    if windows <= 0:
        return np.zeros((0, context), np.int64), np.zeros((0,), np.int64)
    inputs = np.lib.stride_tricks.sliding_window_view(arr[:-1], context)[:windows]
    targets = arr[context:context + windows]
    return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    started = time.time()

    print("building vocabulary from Session 6 corpus...")
    vocab = build_vocab(SOURCES, per_lane=PER_LANE)
    words, lanes = vocab["words"], vocab["lanes"]
    print(f"  {len(words)} words ({lanes.count('en')} en / {lanes.count('hi')} hi)")

    stats = byte_length_stats(words, lanes)
    for lane, s in stats.items():
        print(f"  {lane}: {s['mean_characters']:.1f} chars, "
              f"{s['mean_utf8_bytes']:.1f} bytes "
              f"({s['bytes_per_character']:.2f} bytes/char)")

    codebook = Codebook(dim=DIM)
    print(f"\nencoding vocabulary (dim={DIM}, "
          f"fingerprint={codebook.fingerprint()})...")
    table = np.stack([encode(w, codebook).vector for w in words])
    table_t = torch.from_numpy(table)

    en_in, en_tg = make_sequences(
        [w.lower() for w in vocab["english_stream"]], vocab["index"], CONTEXT)
    hi_in, hi_tg = make_sequences(
        vocab["hindi_stream"], vocab["index"], CONTEXT)
    inputs = np.concatenate([en_in, hi_in])
    targets = np.concatenate([en_tg, hi_tg])
    print(f"  {len(inputs)} training sequences")

    split = int(len(inputs) * 0.9)
    order = rng.permutation(len(inputs))
    train_idx, eval_idx = order[:split], order[split:]

    model = HeadlessTransformer(dim=DIM, context=CONTEXT,
                                layers=LAYERS, heads=HEADS)
    report = model.parameter_report(vocab_size=len(words))
    print(f"\nmodel: {report['model_parameters']:,} parameters, NO output head")
    print(f"  a conventional head would add "
          f"{report['output_head_parameters_avoided']:,}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    print(f"\ntraining {STEPS} steps...")
    losses = []
    for step in range(STEPS):
        pick = rng.choice(train_idx, BATCH)
        x = table_t[torch.from_numpy(inputs[pick])]
        y = table_t[torch.from_numpy(targets[pick])]

        predicted = model(x)
        loss = cosine_loss(predicted, y)

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        losses.append(float(loss))
        if step % 100 == 0 or step == STEPS - 1:
            print(f"  step {step:4d}  loss {float(loss):.4f}  "
                  f"({time.time() - started:.0f}s)")

    print("\nmeasuring error on held-out data...")
    model.eval()
    errors, cosines = [], []
    with torch.no_grad():
        for start in range(0, min(len(eval_idx), 2000), 128):
            pick = eval_idx[start:start + 128]
            x = table_t[torch.from_numpy(inputs[pick])]
            y = table_t[torch.from_numpy(targets[pick])]
            predicted = model(x)
            errors.append(relative_error(predicted, y).numpy())
            cosines.append(
                torch.nn.functional.cosine_similarity(predicted, y, dim=-1).numpy())

    errors = np.concatenate(errors)
    cosines = np.concatenate(cosines)

    # The end-to-end test: decode the model's own predicted vectors with
    # no output head, and see how often the right word comes back. This
    # is the number problem 5 actually asks for -- everything else is
    # diagnostic.
    print("decoding predicted vectors (no output head)...")
    from inverse import decode  # noqa: E402

    exact = 0
    invalid = 0
    attempted = 0
    with torch.no_grad():
        sample = eval_idx[:400]
        for start in range(0, len(sample), 64):
            pick = sample[start:start + 64]
            x = table_t[torch.from_numpy(inputs[pick])]
            predicted = model(x).numpy()
            for row, target_id in enumerate(targets[pick]):
                result_word = decode(predicted[row], codebook)
                attempted += 1
                if result_word.invalid_utf8:
                    invalid += 1
                elif result_word.word == words[target_id]:
                    exact += 1

    decode_accuracy = exact / max(attempted, 1)
    print(f"  exact word recovered: {exact}/{attempted} "
          f"({decode_accuracy * 100:.1f}%), invalid utf-8: {invalid}")

    result = {
        "config": {"dim": DIM, "context": CONTEXT, "layers": LAYERS,
                   "steps": STEPS, "batch": BATCH, "lr": LR,
                   "vocab_size": len(words), "seed": SEED,
                   "codebook_fingerprint": codebook.fingerprint()},
        "vocab_byte_stats": stats,
        "parameters": report,
        "final_loss": float(np.mean(losses[-50:])),
        "relative_error": {
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p10": float(np.percentile(errors, 10)),
            "p90": float(np.percentile(errors, 90)),
            "min": float(errors.min()),
        },
        "cosine_to_target": {
            "mean": float(cosines.mean()),
            "max": float(cosines.max()),
        },
        "headless_decode": {
            "attempted": attempted,
            "exact_word_recovered": exact,
            "accuracy": decode_accuracy,
            "invalid_utf8": invalid,
        },
        "runtime_seconds": time.time() - started,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_run.json").write_text(json.dumps(result, indent=2))

    print(f"\n  relative error: mean {errors.mean():.3f}  "
          f"median {np.median(errors):.3f}  best-decile {np.percentile(errors, 10):.3f}")
    print(f"  cosine to target: mean {cosines.mean():.3f}")
    print(f"\nwrote {OUT / 'model_run.json'}")


if __name__ == "__main__":
    main()
