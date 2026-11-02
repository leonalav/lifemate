"""What is left after the context model, and is any of it a cheap rule fix?

The context model raises the transcript population's argument recovery from 57.9% to
74.5%. The remaining misses fall into a few shapes, and some of them may be a rule fix
rather than a modelling gap. This script classifies them.

  compound_honorific   "Cô giáo Mai" — the honorific is two words and only the first
                       word of the honorific is capitalised in the gold ("Cô giáo"),
                       so the module capitalises the wrong word ("Cô Giáo")
  multi_word_name      a name of several words where the per-word decisions differ
  single_word          one word, decided wrongly
  not_in_transcript    the value's words are absent, so no transcript spelling exists

Run:  python research/analyze_case_remaining.py --data-dir data/train+public_test
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

from lifemate.capitalize import HONORIFICS, restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def classify(gold: str, rule_spelling: str | None) -> str:
    if rule_spelling is None:
        return "not_in_transcript"
    words = gold.split()
    if len(words) == 1:
        return "single_word"
    lowered = gold.lower()
    for honorific in HONORIFICS:
        if " " in honorific and lowered.startswith(honorific + " "):
            return "compound_honorific"
    return "multi_word_name"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())

    kinds = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    slot_of = collections.Counter()

    for sample in split.iter_samples():
        audio_text = sample.get("audio_text") or ""
        if not audio_text:
            continue
        restored = restore_capitals(audio_text.upper())

        for call in sample.get("label", {}).get("tool_calls") or []:
            tool = call.get("name") or ""
            fuzzy_keys = fuzzy.get(tool, set())
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not _has_upper(value):
                    continue
                if _looks_like_timestamp(value):
                    continue
                if key in fuzzy_keys:
                    continue
                if value.lower() not in audio_text.lower():
                    continue
                at = restored.lower().find(value.lower())
                spelling = restored[at:at + len(value)] if at >= 0 else None
                if spelling == value:
                    continue      # the rules already get it right
                kind = classify(value, spelling)
                kinds[kind] += 1
                slot_of[f"{tool}.{key}"] += 1
                if len(examples[kind]) < 20:
                    examples[kind].append({
                        "id": sample["id"], "slot": f"{tool}.{key}",
                        "gold": value, "rule": spelling,
                    })

    report = {
        "remaining_misses": dict(kinds.most_common()),
        "per_slot": slot_of.most_common(12),
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
