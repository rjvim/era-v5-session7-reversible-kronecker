"""
Vocabulary, built from the same real corpus as Session 6.

Reuses this project's own Session 6 sources rather than a synthetic word
list, so the byte-length distribution -- which is what reversibility
actually degrades against -- is realistic for both scripts:

  English   Pride and Prejudice (public domain)
  Hindi     ai4bharat/sangraha, verified/hin split (CC-BY-4.0)

Words longer than the 32-byte budget are excluded rather than truncated.
Truncation would make a word unreconstructable for a reason that has
nothing to do with the inverse map, and would quietly depress the
headline accuracy number.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

MAX_BYTES = 32

EN_PATTERN = re.compile(r"[A-Za-z]+")
HI_PATTERN = re.compile(r"[\u0900-\u097F]+")


def _fits(word: str) -> bool:
    return 0 < len(word.encode("utf-8")) <= MAX_BYTES


def build_vocab(source_dir: Path, per_lane: int = 3000) -> dict:
    """Most frequent words per lane that fit the byte budget."""
    english_text = (source_dir / "pride_and_prejudice.txt").read_text(
        encoding="utf-8", errors="ignore"
    )
    english = [w.lower() for w in EN_PATTERN.findall(english_text)]

    hindi: list[str] = []
    hindi_path = source_dir / "hindi" / "sangraha_verified_hin.jsonl"
    with hindi_path.open(encoding="utf-8") as handle:
        for line in handle:
            hindi.extend(HI_PATTERN.findall(json.loads(line)["text"]))

    en_vocab = [w for w, _ in Counter(english).most_common() if _fits(w)][:per_lane]
    hi_vocab = [w for w, _ in Counter(hindi).most_common() if _fits(w)][:per_lane]

    words = en_vocab + hi_vocab
    lanes = ["en"] * len(en_vocab) + ["hi"] * len(hi_vocab)

    return {
        "words": words,
        "lanes": lanes,
        "index": {w: i for i, w in enumerate(words)},
        "english_stream": english,
        "hindi_stream": hindi,
    }


def byte_length_stats(words: list[str], lanes: list[str]) -> dict:
    """Byte length per lane -- the quantity reversibility degrades against.

    Reported because the Devanagari result is only interpretable next to
    it: Hindi words are not intrinsically harder to reverse, they are
    simply longer in bytes, and this table is the evidence.
    """
    out: dict[str, dict] = {}
    for lane in sorted(set(lanes)):
        selected = [w for w, l in zip(words, lanes) if l == lane]
        byte_lengths = [len(w.encode("utf-8")) for w in selected]
        char_lengths = [len(w) for w in selected]
        out[lane] = {
            "words": len(selected),
            "mean_characters": sum(char_lengths) / len(char_lengths),
            "mean_utf8_bytes": sum(byte_lengths) / len(byte_lengths),
            "max_utf8_bytes": max(byte_lengths),
            "bytes_per_character": sum(byte_lengths) / sum(char_lengths),
        }
    return out
