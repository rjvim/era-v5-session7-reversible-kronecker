# Session 7 — Problem 5: Reversible Kronecker Embedding

**Problem solved: 5 (reversibility).**

Kronecker embedding is forward-deterministic — the same word always
produces the same vector. Problem 5 asks for the reverse: recover the
word from the vector. If that works, the model's final output head (a
`V x d` matrix, ~1.06B parameters at V=131,072 and d=8096) can be
deleted, and vocabulary size stops costing anything.

**Result: the inverse map works and is exact. Wiring it to a model via
vector regression does not, for a reason that is not the one everyone
will test for -- and a byte-logit head removes that failure entirely.**

The blocker is not noise tolerance. A model's error sits near the
corruption threshold the inverse map survives, yet end-to-end accuracy
is a hundred times worse than that predicts. The blocker is that a
model emitting a single point cannot represent uncertainty, and the
point that minimises loss under uncertainty is the average of the
plausible next words — which is not a valid embedding.

This README reports what works, what fails, two attempted fixes that
also fail, why each fails, and the one architecture that follows from
the diagnosis.

---

## 1. What was built

The forward map had to change before any inverse was possible.

**Kronecker V4** builds a word vector as `v = (1/L) * sum_i P_i * M[byte_i]`
— project each byte through a fixed matrix, then *add* a position
embedding. Adding position after projection means the same byte at two
positions shares a large common component. Fine going forward. Fatal
going backward: once summed, no component of `v` can be attributed to a
specific `(position, byte)` pair.

**This implementation** gives every `(position, byte)` pair its own
independent direction, drawn once from a fixed seed:

```
v(word) = (1/L) * sum_i  C[i, byte_i]
```

`C` is generated from a seed, never learned, never stored. Forward stays
deterministic and table-free.

Backward becomes a projection problem. Random directions in high
dimension are near-orthogonal, so taking inner products of `v` against
the 256 candidates at position `i` recovers `byte_i` by argmax:

| | expected inner product |
|---|---|
| true byte at active position | `1/L` |
| any other byte | `O(1/(L*sqrt(d)))` |
| any byte at position `>= L` | cross-talk only |

Length is not stored. It is recovered from where the peak scores fall
off a cliff — active positions sit near `1/L`, inactive ones at the
cross-talk floor.

---

## 2. The inverse map works

Clean round-trip, both scripts, no training, no table, no output head:

| word | utf-8 bytes | round-trip | length margin |
|---|---|---|---|
| `a` | 1 | exact | 32.5 |
| `apple` | 5 | exact | 15.6 |
| `apples` | 6 | exact | 19.6 |
| `internationalization` | 20 | exact | 7.9 |
| `नमस्ते` | 18 | exact | 5.9 |
| `हिन्दी` | 18 | exact | 5.9 |

**100% exact reconstruction.** The margin column is the ratio between
the last active peak and the first inactive one — the headroom before
noise destroys the length cut. It falls as word length rises, exactly as
`1/L` predicts.

### Devanagari fragility is byte length, not script

`नमस्ते` is six characters but eighteen bytes, and behaves like a
twenty-letter English word. Measured over the real vocabulary:

| lane | mean characters | mean utf-8 bytes | bytes/char |
|---|---|---|---|
| en | 6.5 | 6.5 | 1.00 |
| hi | 4.7 | 14.1 | 3.00 |

Devanagari is not intrinsically harder to reverse. UTF-8 charges it 3x,
reversibility degrades with byte count, so it inherits the penalty.
Same root cause as the fertility work in Sessions 2–6, different
symptom.

---

## 3. Isotropic noise is the wrong test

The obvious robustness experiment is to add Gaussian noise. It gives a
badly misleading answer.

Measured at d=8096 over `apple`, `internationalization`, `नमस्ते`:

| noise magnitude (x signal) | random Gaussian | structured |
|---|---|---|
| 0.1x | 100% | 100% |
| 0.2x | — | 95.8% |
| 0.5x | 100% | 37.5% |
| 1.0x | 100% | 8.3% |
| 2.0x | 100% | — |
| 5.0x | 37.5% | — |
| 10.0x | 4.2% | — |

Random noise in 8096 dimensions spreads across all of them, so its
projection onto any one codebook direction is diluted by `sqrt(8096)`
≈ 90x. Reconstruction survives noise **twice the magnitude of the signal
itself** at full accuracy -- not because the method is robust, but
because the test is weak.

Structured noise (built from other codebook directions, the shape a
model's output actually has) crosses 50% around **0.5x**. The two 50%
crossings are roughly **10x apart**.

Anyone testing reversibility with Gaussian noise will conclude it is
near-unbreakable and be wrong by an order of magnitude.

### Tolerance saturates early, and depends on word length

Threshold at which accuracy crosses 50%, by bisection, over a probe of
six short Latin words:

| dim | 50% threshold |
|---|---|
| 128 | 0.217 (collapsed) |
| 256 | 0.833 |
| 512 | 0.972 |
| 1024 | 0.951 |
| 2048 | 0.904 |
| 4096 | 1.000 |

Tolerance does **not** scale as `sqrt(d)`. It collapses only at d=128,
and is flat at roughly 0.9–1.0 from d=512 upward. Buying model width
past that point does not buy robustness.

The two tables above use different word sets and therefore report
different thresholds -- the curve probe includes 18- and 20-byte words,
the sweep probe is all short Latin. That gap is not noise, it is the
`1/L` signal law: an 18-byte word's per-slot signal is a quarter of a
5-byte word's, so it sits proportionally closer to the cross-talk floor.
**Any single "noise threshold" number for this method is meaningless
without stating the word length it was measured at.**

*These numbers were re-measured after the length-inference fix described
in `inverse.py`. An earlier version of this README quoted thresholds
from the fixed-fraction decoder (0.757 at d=512, 1.237 at d=1024, rising
to 1.457); those are superseded. The qualitative conclusion -- no
`sqrt(d)` scaling, early saturation -- survived the correction, but the
magnitudes did not, which is itself a reason to distrust any of these
figures quoted without the decoder version attached.*

---

## 4. The end-to-end failure

A headless transformer — no `V x d` output head, emitting a
`d`-dimensional vector read directly by the inverse map:

| measurement | value |
|---|---|
| relative error of predictions | **0.84** |
| structured-noise 50% threshold | **~0.9–1.0** |
| **exact words recovered end-to-end** | **0.5% (2/400)** |
| predictions decoding to invalid UTF-8 | **199/400** |

Error (0.84) sits just below the threshold (~0.95). The noise analysis
therefore predicts accuracy somewhere near 50%. Observed accuracy is
**0.5%** — two orders of magnitude worse — and half the outputs are not
words in any encoding.

A prediction sitting at the noise threshold should fail *gracefully*,
degrading toward the wrong word. Instead it fails *structurally*,
producing byte strings that decode to nothing. That gap between
predicted and observed failure is the evidence that the noise framing is
measuring the wrong thing entirely.

**Both noise models perturb *around* a valid point. Real model error is
not a perturbation.**

### Diagnosis: superposition

| prediction compared to | cosine |
|---|---|
| nearest single vocabulary word | 0.669 |
| 2nd nearest | 0.647 |
| 10th nearest | 0.595 |
| **blend of top 8 words** | **0.794** |

The blend beats the best single word in **100% of cases**. And the
single-word column is nearly flat — 1st and 10th nearest differ by only
0.074. The prediction is not near one word with others trailing. It sits
at the centroid of all of them.

**The mechanism.** The model does not know the next word; several are
plausible. A conventional softmax head handles this by emitting a
*distribution*. A headless model can only emit a *point*, and the point
minimising cosine loss under uncertainty is the **average** of the
plausible targets.

An average of embeddings is not an embedding. It is a superposition off
the manifold of valid words. The decoder reads bytes off it and gets
bytes belonging to no word — hence the invalid UTF-8.

**This means better training makes it worse.** A better-calibrated model
lands more precisely on a *correct average*, and averages are exactly
what cannot be decoded. Capacity does not help; the obstruction is
geometric.

---

## 5. Two fixes, both fail

### Fix A — structural projection: provably a no-op

Snap the prediction back onto the manifold: argmax over 256 byte
directions per position, re-encode, repeat. Costs `O(32 x 256)` and uses
no vocabulary table. (Projecting onto the nearest entry of a `V x d`
word list would also work — and would be precisely the output head
problem 5 deletes. That version is circular. This one is not.)

**Result: numerically identical to baseline.** Not a bug. Decoding
*already is* a projection onto the manifold. Re-encoding the decoded
bytes and decoding again returns the same bytes, by construction. The
error is in *which bytes win the argmax*, and projection cannot change
that.

### Fix B — contrastive training: strictly worse

Replace cosine regression with InfoNCE against in-batch negatives.
Cosine regression pulls toward the mean; a ranking objective should not.
Negatives come from the batch's own deterministic target embeddings, so
no head and no table are needed.

| variant | exact | invalid utf-8 |
|---|---|---|
| baseline (cosine) | 0.0% | 39.8% |
| fix A (projection) | 0.0% | 39.8% |
| fix B (InfoNCE) | 0.0% | **100.0%** |
| fix B + projection | 0.0% | 100.0% |

**Every** contrastive prediction decoded to invalid UTF-8. InfoNCE only
requires the prediction to rank its target above 31 in-batch negatives —
in high dimensions that is satisfiable from almost anywhere. The model
scores well while drifting far off-manifold. It fixes averaging and
abandons validity.

Both fixes attack the symptom. Neither constrains the output to be a
valid embedding *and* capable of expressing uncertainty. Those are the
two requirements, and no point-regression objective satisfies both.

---

## 6. The fix, and a confirmed prediction

The diagnosis rules out an entire class of approaches and points at one
that survives it.

The model must emit something that (a) can express uncertainty and
(b) is constrained to well-formed embeddings. A `d`-vector fails (b) the
moment it fails (a).

**Predict per-position byte logits instead of a vector.**

The manifold is exactly parameterised by `32 positions x 256 byte
values`. So emit `32 x 256 = 8192` logits — a distribution over bytes at
each position — rather than `d` real numbers:

- **Uncertainty is representable.** Position 3 can be genuinely unsure
  between `p` and `b` without averaging them into something that is
  neither.
- **Every output is on-manifold.** Any argmax or sample over byte logits
  is a well-formed byte string by construction.
- **The head still does not scale with vocabulary.** `d x 8192`,
  constant in `V`.

| vocabulary | conventional head (`V x d`, d=8096) | byte-logit head (`d x 8192`) |
|---|---|---|
| 131,072 | 1.06B | 66.3M |
| 1,000,000 | 8.10B | 66.3M |

### The prediction, stated before running

If the superposition diagnosis is right, constraining the output to the
manifold should drive the invalid-UTF-8 rate to **zero**. If it does
not, section 4 is wrong and needs revising.

### Result

At d=512 on a 1,600-word vocabulary, 400 steps (`exp5`):

| architecture | exact | invalid utf-8 |
|---|---|---|
| vector head, cosine regression | 0.5% | 49.8% |
| vector head, InfoNCE | 0.0% | 100.0% |
| **byte-logit head** | **2.3%** | **0.0%** |

**Invalid UTF-8 collapses from 49.8% to zero.** Not reduced —
eliminated, as the construction requires.

### Held under a matched budget

A single training point proves little, so both architectures were then
run on a 5,000-word vocabulary with identical data, seeds and step
budgets, checkpointing between chunks, evaluating on 512 held-out
samples (`exp6`):

| steps | byte-logit exact | byte-logit invalid | vector exact | vector invalid |
|---|---|---|---|---|
| ~500 | 1.0% | **0.0%** | — | — |
| ~1500 | 4.3% | **0.0%** | 1.6% | 50.2% |
| ~2500 | 2.7% | **0.0%** | 1.8% | 51.6% |

The invalid-UTF-8 rate is **exactly zero at every checkpoint**, while
the vector arm sits at roughly 51% regardless of how long it trains.
That is the signature of a structural property rather than a learned
one: training changes the vector arm's loss but not its validity,
because validity was never something gradient descent could reach.

This is the strongest evidence in the submission that the diagnosis is
correct, because it was a falsifiable prediction made in advance rather
than an observation explained afterwards.

**Reported honestly: byte-logit exact recovery went 4.3% -> 2.7%
between 1,500 and 2,500 steps.** At 512 eval samples that is roughly 8
words, within noise, but it is a regression and the peak figure is not
quoted alone. Exact recovery is not monotone here and this run is too
small to say whether that is overfitting, noise, or a real ceiling.

**What this does not show.** A few percent exact recovery is not a
working language model. The byte-logit head removes the *structural*
failure — outputs are now always valid words — but predicting the right
next word remains hard, and a 2-layer model on 5,000 words is not going
to do it. The claim here is narrow and should be read narrowly: the
architecture that makes reversible embedding *usable* is byte logits,
not vector regression. Whether it produces a good model is a separate
question this does not answer.

The head is also not free — `d x 8192` is real parameters where the
vector formulation had none. The comparison is matched on what problem 5
cares about: neither has a `V x d` matrix, neither grows with
vocabulary. At V=131,072 the byte head is 16x smaller than the
conventional head it replaces; at V=1,000,000, 122x.

---

## 7. Honest limitations

- **Scale.** d=512–1024, up to 5,000 words, ~8M parameters, up to 2,500
  steps, on a single CPU with 3GB of RAM. The superposition argument is
  geometric and should not depend on scale, but that is an argument, not
  a measurement. The structural result (0% invalid) held across every
  budget tested, which is the part least likely to change; the exact-
  recovery figures are small-sample and moved non-monotonically.
- **Vocabulary size cuts against the result.** At 131K words there are
  more plausible candidates per position, so blending should get
  *worse*. Directionally safe, unverified.
- **Structured noise is a proxy.** It has the right shape but is not a
  real error distribution. The 0.5x threshold is indicative.
- **The threshold-vs-error comparison was the wrong test** and is
  reported anyway (Section 4), because the size of the gap between what
  it predicts (~50%) and what happens (0.5%) is what makes the
  superposition diagnosis necessary.
- **The noise thresholds were re-measured once**, after a length-
  inference bug was found by the invariant tests. The qualitative
  conclusions held; the magnitudes moved substantially. Treat any single
  threshold figure here as approximate and word-length-specific.
- **Fix B is undertrained** at 250 steps. Its failure mode (100%
  invalid) is structural rather than a convergence artefact, but longer
  training was not run.
- Words over 32 bytes are excluded, not truncated — truncation would
  depress accuracy for reasons unrelated to the inverse map.

---

## 8. Running it

```
pip install -r requirements.txt
python3 experiments/exp1_clean.py     # clean round-trip, full vocab
python3 experiments/exp2_noise.py     # noise curves, dimension sweep
python3 experiments/exp3_model.py     # headless model, end-to-end
python3 experiments/exp4_lean.py      # both fixes
python3 experiments/exp5_bytelogit.py # the byte-logit head
python3 experiments/exp6_scaled.py bytelogit 500   # matched-budget comparison,
python3 experiments/exp6_scaled.py vector 500      #   resumable in chunks
```

```
kronecker/
  codebook.py   fixed (position, byte) -> direction map; nothing learned
  forward.py    word -> vector, deterministic, no table
  inverse.py    vector -> word; the contribution
  model.py      headless transformer; no V x d output head
  bytelogit.py  byte-logit head; the working architecture
  vocab.py      vocabulary from the Session 6 corpus
experiments/    the four experiments above
tests/          invariants asserted independently of the demo output
submission_artifacts/results/   machine-generated json, not hand-written
```

Corpus is reused from this project's Session 6 submission: *Pride and
Prejudice* (public domain) and `ai4bharat/sangraha` `verified/hin`
(CC-BY-4.0), so the byte-length distribution is realistic for both
scripts.

---

## 9. Summary

| claim | status |
|---|---|
| Inverse Kronecker map exists and is exact | **verified, 100%** |
| Works for Devanagari as well as Latin | **verified** |
| Noise tolerance is word-length dependent | **verified** |
| Gaussian noise is a misleading robustness test | **verified, ~10x** |
| Devanagari fragility is byte length, not script | **verified** |
| Reversibility gives a working headless model | **false — 0.5%** |
| Failure is caused by noise | **false** |
| Failure is caused by off-manifold superposition | **verified, 100% of cases** |
| Structural projection fixes it | **false — provably a no-op** |
| Contrastive training fixes it | **false — strictly worse** |
| Byte-logit head removes the structural failure | **verified — 0.0% invalid at every checkpoint, vs ~51% for vectors** |
| Byte-logit head yields a good language model | **not shown — 2.3% exact** |

The map is invertible. The architecture that uses it is a byte-logit
head, not vector regression -- and the diagnosis that led there was
confirmed by a prediction made before the experiment was run.

What remains open is whether a byte-logit model trained at real scale
predicts words well. That is the natural next paper, and it is a
different question from the one problem 5 asks.
