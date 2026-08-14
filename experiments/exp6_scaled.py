"""
Experiment 6: the byte-logit head at the largest scale this machine
allows, and a comparison against the vector head under identical budget.

exp5 established the structural result -- invalid utf-8 goes to zero --
at d=512 on 1,600 words after 400 steps. The obvious objection is that
400 steps proves nothing about whether the architecture can actually
learn, and that the 2.3% exact figure is just an undertrained model.

This runs both architectures for as long as the hardware permits, on a
larger vocabulary, checkpointing between chunks so training can
accumulate across separate invocations rather than being capped by any
single one. Both arms get identical data, identical budgets, identical
seeds. Only the output head differs.

Two things worth separating in the results:

  - **Structural validity** (invalid utf-8 rate). Predicted to be zero
    for byte logits at any budget, because it follows from the
    construction rather than from training. If it drifts above zero with
    more steps, the section 6 claim is wrong.

  - **Predictive quality** (exact word recovery). Expected to improve
    with budget for both arms. This is the number that says whether the
    architecture is merely valid or actually useful, and it is the one
    exp5 could not speak to.
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
from model import HeadlessTransformer, cosine_loss  # noqa: E402
from vocab import build_vocab  # noqa: E402

DIM = 512
CONTEXT = 8
LAYERS = 2
HEADS = 8
BATCH = 48
LR = 5e-4
PER_LANE = 2500
SEED = 7
EVAL_SAMPLES = 512
MAX_POSITIONS = 32

SOURCES = ROOT.parent / "s6" / "data_pipeline" / "real_sources"
OUT = ROOT / "submission_artifacts" / "results"
CKPT = ROOT / "submission_artifacts" / "checkpoints"


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


class Decoder:
    """Batched inverse map for the vector arm."""

    def __init__(self, codebook):
        self.stack = np.stack(
            [codebook.directions(p) for p in range(codebook.max_positions)])

    def words(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        scores = np.einsum("pbd,nd->npb", self.stack, vectors)
        best_byte, best_score = scores.argmax(2), scores.max(2)
        out = []
        for row in range(len(vectors)):
            peaks = best_score[row]
            ratios = peaks[:-1] / np.maximum(peaks[1:], 1e-12)
            length = int(np.argmax(ratios)) + 1
            recovered = bytes(int(b) for b in best_byte[row][:length])
            try:
                out.append(recovered.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(None)
        return out


def build_state():
    rng = np.random.default_rng(SEED)
    vocab = build_vocab(SOURCES, per_lane=PER_LANE)
    words = vocab["words"]
    codebook = Codebook(dim=DIM)
    table = torch.from_numpy(np.stack([encode(w, codebook).vector for w in words]))

    en_in, en_tg = make_sequences(
        [w.lower() for w in vocab["english_stream"]], vocab["index"], CONTEXT)
    hi_in, hi_tg = make_sequences(vocab["hindi_stream"], vocab["index"], CONTEXT)
    inputs = np.concatenate([en_in, hi_in])
    targets = np.concatenate([en_tg, hi_tg])

    order = rng.permutation(len(inputs))
    split = int(len(inputs) * 0.9)
    return {
        "words": words, "codebook": codebook, "table": table,
        "inputs": inputs, "targets": targets,
        "train_idx": order[:split], "eval_idx": order[split:],
        "grid_and_lengths": byte_targets(words),
    }


def make_model(arm):
    torch.manual_seed(SEED)
    if arm == "bytelogit":
        return ByteLogitTransformer(dim=DIM, context=CONTEXT, layers=LAYERS,
                                    heads=HEADS, max_positions=MAX_POSITIONS)
    return HeadlessTransformer(dim=DIM, context=CONTEXT, layers=LAYERS, heads=HEADS)


def train_chunk(arm, state, steps):
    """Train `steps` more steps, resuming from checkpoint if present."""
    CKPT.mkdir(parents=True, exist_ok=True)
    path = CKPT / f"{arm}.pt"

    model = make_model(arm)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    done = 0
    if path.exists():
        blob = torch.load(path, weights_only=False)
        model.load_state_dict(blob["model"])
        optimiser.load_state_dict(blob["optimiser"])
        done = blob["steps"]

    rng = np.random.default_rng(SEED + done)
    table, inputs, targets = state["table"], state["inputs"], state["targets"]
    grid, lengths = state["grid_and_lengths"]
    started = time.time()

    model.train()
    for step in range(steps):
        pick = rng.choice(state["train_idx"], BATCH)
        x = table[torch.from_numpy(inputs[pick])]
        target_ids = torch.from_numpy(targets[pick])

        if arm == "bytelogit":
            byte_logits, length_logits = model(x)
            loss = byte_loss(byte_logits, length_logits,
                             grid[target_ids], lengths[target_ids])
        else:
            loss = cosine_loss(model(x), table[target_ids])

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 200 == 0:
            print(f"    [{arm}] +{step:4d} (total {done + step})  "
                  f"{float(loss.detach()):.4f}  ({time.time() - started:.0f}s)",
                  flush=True)

    done += steps
    torch.save({"model": model.state_dict(),
                "optimiser": optimiser.state_dict(),
                "steps": done}, path)
    return model, done


def evaluate(arm, model, state):
    words, table = state["words"], state["table"]
    inputs, targets = state["inputs"], state["targets"]
    sample = state["eval_idx"][:EVAL_SAMPLES]
    truth = [words[i] for i in targets[sample]]
    decoder = Decoder(state["codebook"])

    model.eval()
    recovered = []
    with torch.no_grad():
        for start in range(0, len(sample), 64):
            pick = sample[start:start + 64]
            x = table[torch.from_numpy(inputs[pick])]
            if arm == "bytelogit":
                byte_logits, length_logits = model(x)
                recovered.extend(decode_words(byte_logits, length_logits))
            else:
                recovered.extend(decoder.words(model(x).numpy()))

    exact = sum(1 for got, want in zip(recovered, truth) if got == want)
    invalid = sum(1 for got in recovered if got is None)
    return {"attempted": len(truth), "exact": exact,
            "accuracy": exact / len(truth), "invalid_utf8": invalid,
            "invalid_fraction": invalid / len(truth)}


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "bytelogit"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    state = build_state()
    print(f"arm={arm}  vocab {len(state['words'])}  dim {DIM}  "
          f"sequences {len(state['inputs'])}", flush=True)

    model, total = train_chunk(arm, state, steps)
    result = evaluate(arm, model, state)
    result["total_steps"] = total
    result["config"] = {"dim": DIM, "batch": BATCH, "lr": LR,
                        "vocab_size": len(state["words"]), "seed": SEED,
                        "context": CONTEXT, "layers": LAYERS}

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"scaled_{arm}.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(result)
    path.write_text(json.dumps(history, indent=2))

    print(f"\n  [{arm}] after {total} steps: "
          f"exact {result['accuracy'] * 100:.1f}%  "
          f"invalid utf-8 {result['invalid_fraction'] * 100:.1f}%")


if __name__ == "__main__":
    main()
