"""
Invariants asserted independently of any experiment's printed output.

Each test checks one claim the README makes, building its own state
rather than trusting artefacts a previous run left behind. Run:

    python3 tests/test_invariants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kronecker"))

from codebook import Codebook          # noqa: E402
from forward import encode             # noqa: E402
from inverse import decode, roundtrip  # noqa: E402
from bytelogit import ByteLogitTransformer, decode_words  # noqa: E402
from constrained import decode_constrained  # noqa: E402
from model import HeadlessTransformer  # noqa: E402

DIM = 1024
WORDS = ["apple", "a", "internationalization", "नमस्ते", "भारत", "tree", "9"]


def test_forward_is_deterministic():
    """Same word, two independently constructed codebooks, identical bytes."""
    for word in WORDS:
        first = encode(word, Codebook(dim=DIM)).vector
        second = encode(word, Codebook(dim=DIM)).vector
        assert first.tobytes() == second.tobytes(), word


def test_forward_is_injective():
    """No two distinct words collide on the same vector."""
    codebook = Codebook(dim=DIM)
    seen: dict[bytes, str] = {}
    for word in WORDS:
        key = encode(word, codebook).vector.tobytes()
        assert key not in seen, f"{word} collides with {seen.get(key)}"
        seen[key] = word


def test_inverse_is_exact_at_zero_noise():
    """The core claim: clean round-trip recovers the word exactly."""
    codebook = Codebook(dim=DIM)
    for word in WORDS:
        ok, result = roundtrip(word, codebook)
        assert ok, f"{word} -> {result.word!r}"


def test_devanagari_costs_three_bytes_per_character():
    """The Devanagari penalty is a property of utf-8, asserted not assumed."""
    for word in ["नमस्ते", "भारत", "हिन्दी"]:
        assert len(word.encode("utf-8")) == 3 * len(word), word


def test_margin_falls_with_word_length():
    """Signal per slot is 1/L, so longer words must have less headroom."""
    codebook = Codebook(dim=DIM)
    short = decode(encode("ab", codebook).vector, codebook).length_margin
    long = decode(encode("internationalization", codebook).vector,
                  codebook).length_margin
    assert short > long, f"{short} !> {long}"


def test_codebook_is_near_orthogonal():
    """Cross-talk floor must stay far below the 1/L signal it competes with."""
    codebook = Codebook(dim=DIM)
    assert codebook.max_offdiagonal(0) < 0.25


def test_model_has_no_output_head():
    """The whole point of problem 5: no V x d matrix anywhere in the model."""
    model = HeadlessTransformer(dim=DIM, context=8, layers=2, heads=8)
    vocab_size = 131072
    for parameter in model.parameters():
        assert vocab_size not in parameter.shape, "found a vocab-sized parameter"
    report = model.parameter_report(vocab_size=vocab_size)
    assert report["output_head_parameters_avoided"] == vocab_size * DIM


def test_projection_is_a_noop():
    """README section 5 claims fix A cannot help. Assert it, don't just say it."""
    codebook = Codebook(dim=DIM)
    rng = np.random.default_rng(0)
    vector = encode("apple", codebook).vector.astype(np.float64)
    vector = vector + 0.3 * np.linalg.norm(vector) * rng.standard_normal(DIM) / np.sqrt(DIM)

    first = decode(vector, codebook)
    rebuilt = np.zeros(DIM)
    for position, byte_value in enumerate(first.recovered_bytes):
        rebuilt += codebook.direction(position, int(byte_value))
    rebuilt /= max(len(first.recovered_bytes), 1)

    assert decode(rebuilt, codebook).recovered_bytes == first.recovered_bytes


def test_independent_argmax_does_NOT_guarantee_valid_utf8():
    """Documents the limitation the earlier version of this file missed.

    The original test here asserted the SHAPE of the byte-logit output
    and concluded validity was structural. It was not testing the claim
    it was named after, so it confirmed a false statement instead of
    falsifying it. The claim -- that any independent per-position argmax
    yields well-formed UTF-8 -- is wrong, because UTF-8 is a grammar: a
    lead byte constrains what may follow, and independent argmaxes have
    no channel to communicate that constraint between positions.

    This test asserts the counterexamples directly, so the limitation is
    encoded in the suite rather than left in prose.
    """
    reachable_by_independent_argmax = [
        bytes([0xE0, 0xA4, 0x41]),  # Devanagari lead, ASCII continuation
        bytes([0xC3, 0x41]),        # 2-byte lead, ASCII continuation
        bytes([0xA4, 0xA8]),        # continuations with no lead
        bytes([0xE0, 0xA4]),        # 3-byte sequence cut short
    ]
    for candidate in reachable_by_independent_argmax:
        try:
            candidate.decode("utf-8")
            raise AssertionError(
                f"{list(candidate)} decoded, so it is not a counterexample")
        except UnicodeDecodeError:
            pass


def test_bytelogit_output_shape_is_correct():
    """What the old test actually checked, now named accurately."""
    import torch

    torch.manual_seed(0)
    model = ByteLogitTransformer(dim=256, context=8, layers=1, heads=8)
    with torch.no_grad():
        byte_logits, length_logits = model(torch.randn(32, 8, 256))

    assert byte_logits.shape == (32, 32, 256)
    lengths = length_logits.argmax(-1)
    for row in range(32):
        assert 0 <= int(lengths[row]) <= model.max_positions


def test_bytelogit_head_is_constant_in_vocab_size():
    """The reason this architecture is worth having at all."""
    model = ByteLogitTransformer(dim=512, context=8, layers=1, heads=8)
    small = model.parameter_report(vocab_size=131072)
    large = model.parameter_report(vocab_size=1000000)
    assert small["byte_head_parameters"] == large["byte_head_parameters"]
    assert large["head_reduction_factor"] > small["head_reduction_factor"]


def test_constrained_decoding_cannot_emit_invalid_utf8():
    """The actual fix: validity as a property of the decoder.

    Stress-tested against RANDOM logits rather than a trained model's,
    because a trained model's logits are exactly the distribution that
    hid this problem in the first place -- 0% invalid was measured on
    sharply peaked outputs and mistaken for a guarantee. Random logits
    reach the adversarial corners.
    """
    import torch

    rng = np.random.default_rng(0)
    count = 4000
    byte_logits = torch.from_numpy(
        rng.standard_normal((count, 32, 256)).astype("float32"))
    length_logits = torch.from_numpy(
        rng.standard_normal((count, 33)).astype("float32"))

    for recovered in decode_constrained(byte_logits, length_logits):
        recovered.decode("utf-8")  # raises if the decoder is wrong


def test_constrained_decoding_handles_narrowed_lead_bytes():
    """RFC 3629 narrows the first continuation for four leads.

    A first version of constrained.py used 0x80-0xBF for every
    continuation and emitted `E0 80 80`, which does not decode. These
    four leads are the cases that catches.
    """
    import torch

    for lead, illegal_continuation in ((0xE0, 0x80), (0xED, 0xA0),
                                       (0xF0, 0x80), (0xF4, 0x90)):
        byte_logits = torch.full((1, 32, 256), -10.0)
        byte_logits[0, 0, lead] = 10.0
        for position in range(1, 4):
            byte_logits[0, position, illegal_continuation] = 10.0
        length_logits = torch.full((1, 33), -10.0)
        length_logits[0, 4] = 10.0

        decode_constrained(byte_logits, length_logits)[0].decode("utf-8")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as error:
            failed += 1
            print(f"  FAIL  {test.__name__}: {error}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
