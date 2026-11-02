"""Building the capitalisation vocabulary from the training transcripts.

The ASR emits uppercase text, so the capitals are lost before the model ever sees the
transcript. The gold transcripts carry the correct casing, and the gold string arguments
are compared with exact equality, so recovering the casing is worth real points.

This script derives the vocabulary from the 1,545 gold training transcripts and writes it
to ``src/lifemate/resources/capitalization.json``. It is a *training-derived artefact*:
the public test has no gold transcripts, so the vocabulary has to come from train.

The rule, and why:

* a word that appears **only** in capitalised form in train is a proper word, an acronym
  or a brand — it keeps its capital ("Zalo", "SMS", "Sếp", "Huế", "VOV3")
* a word that appears in **both** cases is an ordinary word — it is lowercased
  ("mai" the month, "ba" the father), and the case that survives is the context's job
  (an honorific restores a personal name: "cô Mai")
* the most frequent capitalised spelling wins, so "ZALO" is stored as "Zalo"

Run:  python research/build_capitalization.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import resolve_split  # noqa: E402

OUTPUT = REPO_ROOT / "src" / "lifemate" / "resources" / "capitalization.json"

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def build(data_dir: str, split_name: str = "train") -> dict:
    """Derive the vocabulary from a labelled split."""
    split = resolve_split(data_dir, split_name)

    lower: collections.Counter = collections.Counter()
    upper: collections.Counter = collections.Counter()
    spellings: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )

    for sample in split.iter_samples():
        text = sample.get("audio_text") or ""
        for word in WORD.findall(text):
            key = word.lower()
            if word[0].isupper():
                upper[key] += 1
                spellings[key][word] += 1
            else:
                lower[key] += 1

    proper: dict[str, str] = {}
    ordinary: list[str] = []
    for key in set(lower) | set(upper):
        if lower[key] == 0 and upper[key] > 0:
            # only ever capitalised -> a proper word, acronym or brand
            proper[key] = spellings[key].most_common(1)[0][0]
        elif lower[key] > 0 and upper[key] > 0:
            # seen in both cases -> an ordinary word
            ordinary.append(key)

    # An acronym keeps its capitals; a name keeps its capitalised spelling.
    acronyms = {
        key: value for key, value in proper.items()
        if value.isupper() and len(value) <= 5
    }
    return {
        "proper": {k: proper[k] for k in sorted(proper) if k not in acronyms},
        "acronyms": {k: acronyms[k] for k in sorted(acronyms)},
        "ordinary": sorted(ordinary),
        "stats": {
            "words": len(set(lower) | set(upper)),
            "proper": len(proper),
            "acronyms": len(acronyms),
            "ordinary": len(ordinary),
        },
    }


def main() -> int:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/train+public_test"
    payload = build(data_dir)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"wrote {OUTPUT}")
    print("stats:", payload["stats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
