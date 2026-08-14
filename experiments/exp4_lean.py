"""
Experiment 4 (lean): fixing the superposition failure.

Same experiment as exp4_fixes.py, sized to run on a single CPU. Smaller
dimension and fewer steps; the comparison between objectives is what
matters here, and both arms get identical budgets so the comparison
stays fair.

exp3 established the failure: a headless model emits a point, and under
uncertainty the cosine-loss-minimising point is the average of the
plausible next words. That average is off the manifold of valid
embeddings, so the inverse map reads bytes off something that is not a
word.

FIX A -- structural projection at decode time.
    Snap the prediction back onto the manifold: argmax over the 256 byte
    directions at each position, re-encode, repeat. Costs O(32 x 256)
    and uses NO vocabulary table. Projecting onto the nearest entry of a
    V x d word table would also work and would also be precisely the
    output head problem 5 is trying to delete -- so that version would
    be circular. This one is not.

FIX B -- contrastive training.
    Attack the cause. InfoNCE asks the prediction to score higher
    against its true target than against other targets in the batch,
    which is minimised by committing to a single point rather than
    averaging several. Negatives are the batch's own target embeddings,
    computed deterministically by the forward map -- still no head, still
    no table.
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

from codebook import Codebook  # noqa: E402
from forward import encode  # noqa: E402
from inverse import decode  # noqa: E402
from model import HeadlessTransformer, cosine_loss  # noqa: E402
from vocab import build_vocab  # noqa: E402

DIM = 512           # reduced to fit a single CPU; see README caveats
CONTEXT = 8
STEPS = 250
BATCH = 32
LR = 5e-4
PER_LANE = 800
SEED = 7
TEMPERATURE = 0.05
EVAL_SAMPLES = 128

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


def infonce_loss(predicted, target, temperature=TEMPERATURE):
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = (predicted @ target.T) / temperature
    return F.cross_entropy(logits, torch.arange(len(predicted)))


class Decoder:
    """Batched decode + structural projection, sharing one stacked codebook."""

    def __init__(self, codebook: Codebook):
        self.codebook = codebook
        self.stack = np.stack(
            [codebook.directions(p) for p in range(codebook.max_positions)])

    def _bytes(self, vectors, threshold=0.35):
        scores = np.einsum("pbd,nd->npb", self.stack, vectors)
        best_byte = scores.argmax(2)
        best_score = scores.max(2)
        out = []
        for row in range(len(vectors)):
            reference = best_score[row].max()
            length = 0
            for position in range(self.stack.shape[0]):
                if best_score[row][position] < threshold * reference:
                    break
                length += 1
            out.append(bytes(int(b) for b in best_byte[row][:length]))
        return out

    def project(self, vectors, iterations=2):
        """Fix A: iterate decode -> re-encode to land on the manifold."""
        current = np.asarray(vectors, dtype=np.float64)
        for _ in range(iterations):
            rebuilt = np.zeros_like(current)
            for row, recovered in enumerate(self._bytes(current)):
                if not recovered:
                    rebuilt[row] = current[row]
                    continue
                for position, byte_value in enumerate(recovered):
                    rebuilt[row] += self.stack[position, int(byte_value)]
                rebuilt[row] /= len(recovered)
            current = rebuilt
        return current

    def words(self, vectors):
        out = []
        for recovered in self._bytes(np.asarray(vectors, dtype=np.float64)):
            try:
                out.append(recovered.decode("utf-8"))
            except UnicodeDecodeError:
                out.append(None)
        return out


def train(objective, table_t, inputs, targets, train_idx, label):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = HeadlessTransformer(dim=DIM, context=CONTEXT, layers=2, heads=8)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR)
    started = time.time()
    for step in range(STEPS):
        pick = rng.choice(train_idx, BATCH)
        loss = objective(
            model(table_t[torch.from_numpy(inputs[pick])]),
            table_t[torch.from_numpy(targets[pick])],
        )
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        if step % 200 == 0 or step == STEPS - 1:
            print(f"    [{label}] {step:4d}  {float(loss.detach()):.4f}  "
                  f"({time.time() - started:.0f}s)", flush=True)
    model.eval()
    return model


def evaluate(model, decoder, table_t, inputs, targets, eval_idx, words):
    predictions = []
    with torch.no_grad():
        sample = eval_idx[:EVAL_SAMPLES]
        for start in range(0, len(sample), 64):
            pick = sample[start:start + 64]
            predictions.append(model(table_t[torch.from_numpy(inputs[pick])]).numpy())
    predicted = np.concatenate(predictions)
    truth = [words[i] for i in targets[eval_idx[:EVAL_SAMPLES]]]

    def score(vectors):
        recovered = decoder.words(vectors)
        exact = sum(1 for got, want in zip(recovered, truth) if got == want)
        invalid = sum(1 for got in recovered if got is None)
        return {
            "attempted": len(truth),
            "exact": exact,
            "accuracy": exact / len(truth),
            "invalid_utf8": invalid,
            "invalid_fraction": invalid / len(truth),
        }

    return score(predicted), score(decoder.project(predicted))


def main():
    rng = np.random.default_rng(SEED)
    vocab = build_vocab(SOURCES, per_lane=PER_LANE)
    words = vocab["words"]
    codebook = Codebook(dim=DIM)
    decoder = Decoder(codebook)
    table_t = torch.from_numpy(
        np.stack([encode(w, codebook).vector for w in words]))

    en_in, en_tg = make_sequences(
        [w.lower() for w in vocab["english_stream"]], vocab["index"], CONTEXT)
    hi_in, hi_tg = make_sequences(vocab["hindi_stream"], vocab["index"], CONTEXT)
    inputs = np.concatenate([en_in, hi_in])
    targets = np.concatenate([en_tg, hi_tg])

    order = rng.permutation(len(inputs))
    split = int(len(inputs) * 0.9)
    train_idx, eval_idx = order[:split], order[split:]

    print(f"vocab {len(words)}  dim {DIM}  sequences {len(inputs)}", flush=True)

    results = {}
    print("  cosine regression", flush=True)
    model = train(cosine_loss, table_t, inputs, targets, train_idx, "cos")
    results["baseline"], results["fix_a_projection"] = evaluate(
        model, decoder, table_t, inputs, targets, eval_idx, words)
    del model

    print("  contrastive (InfoNCE)", flush=True)
    model = train(infonce_loss, table_t, inputs, targets, train_idx, "nce")
    results["fix_b_contrastive"], results["fix_b_plus_projection"] = evaluate(
        model, decoder, table_t, inputs, targets, eval_idx, words)

    results["config"] = {
        "dim": DIM, "steps": STEPS, "batch": BATCH, "context": CONTEXT,
        "vocab_size": len(words), "temperature": TEMPERATURE, "seed": SEED,
        "codebook_fingerprint": codebook.fingerprint(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fixes.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'variant':<26}{'exact':>9}{'invalid utf8':>15}")
    for name in ("baseline", "fix_a_projection", "fix_b_contrastive",
                 "fix_b_plus_projection"):
        r = results[name]
        print(f"{name:<26}{r['accuracy'] * 100:>8.1f}%{r['invalid_fraction'] * 100:>14.1f}%")


if __name__ == "__main__":
    main()
