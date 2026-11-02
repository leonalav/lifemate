"""A precise fix for the acronym codes, and whether the model should touch them.

The naive fix — asking ``is_acronym`` in the render path — is a disaster: it accepts any
unknown short uppercase word, so "giúp" became "GIÚP" and 6,974 tokens broke against
13 fixed.

The precise signal is the *derived* acronym set. It already holds the alphabetic part of
the codes the training transcripts spell consistently ("vov", "vtv", "ron", "rv", "e",
"s"). The codes that the gold keeps in caps are exactly those whose alphabetic part is
in that set: "VOV3" from "vov", "RON95" from "ron", "E5" from "e", "S+" from "s".

The words the naive fix broke — "nhom22", "24h", "Omega-3" — have alphabetic parts
("nhom", "h", "omega") that are not in the set, so they keep their lowercase form.

This script measures two things:

  1. the render path change: keep the uppercase form when the alphabetic part of a
     digit-bearing token is in the derived acronym set
  2. whether the context model should then be allowed to touch a digit-bearing token
     at all, since the gold's casing for those is decided by the code shape

Run:  python research/analyze_acronym_codes.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import _vocabulary, restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402

TOKEN = re.compile(r"\S+")


def _alpha_part(token: str) -> str:
    """The token's letters, with the digits removed."""
    return "".join(c for c in token if c.isalpha())


def keeps_its_case(token: str, acronyms: frozenset[str]) -> bool:
    """True when a digit-bearing token is a code the derived set recognises.

    "VOV3" -> "vov" -> in the set. "nhom22" -> "nhom" -> not in the set.
    """
    if not any(c.isdigit() for c in token):
        return False
    alpha = _alpha_part(token).lower()
    return bool(alpha) and alpha in acronyms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    proper, acronyms, _ordinary = _vocabulary()

    stats = collections.Counter()
    fixed = collections.Counter()
    broken = collections.Counter()

    for sample in split.iter_samples():
        text = sample.get("audio_text") or ""
        if not text:
            continue
        current = restore_capitals(text.upper())

        gold_tokens = TOKEN.findall(text)
        current_tokens = TOKEN.findall(current)
        if len(gold_tokens) != len(current_tokens):
            continue

        proposed_tokens = []
        for gold, now in zip(gold_tokens, current_tokens):
            bare = gold.strip(",.;:!?\"'()")
            if keeps_its_case(bare, acronyms):
                # a code the derived set recognises keeps its uppercase form
                proposed = bare
            else:
                proposed = now
            proposed_tokens.append(proposed)

        for gold, now, then in zip(gold_tokens, current_tokens, proposed_tokens):
            stats["tokens"] += 1
            if now == gold:
                stats["current_ok"] += 1
            if then == gold:
                stats["proposed_ok"] += 1
            if now != gold and then == gold:
                stats["fixed"] += 1
                fixed[(gold, now)] += 1
            elif now == gold and then != gold:
                stats["broken"] += 1
                broken[(gold, then)] += 1

    total = stats["tokens"] or 1
    report = {
        "tokens": stats["tokens"],
        "current_rule_module_accuracy": round(
            100.0 * stats["current_ok"] / total, 3),
        "proposed_accuracy": round(100.0 * stats["proposed_ok"] / total, 3),
        "fixed": stats["fixed"],
        "broken": stats["broken"],
        "net": stats["fixed"] - stats["broken"],
        "fixes": fixed.most_common(20),
        "breakages": broken.most_common(20),
        "derived_acronyms": sorted(acronyms),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
