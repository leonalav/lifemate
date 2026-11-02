"""The true capitalisation population: which exact-compared arguments really carry case.

The first pass over-counted, because an ISO timestamp ("2026-11-29T09:00:00+07:00")
contains an uppercase "T" and was counted as a capitalised value. This script separates
the four populations that matter:

* **timestamps** — the uppercase letter is the ISO "T"; nothing to restore.
* **case-only differences** — the value differs from its lowercase form only in case,
  so getting the case wrong *does* fail an exact comparison.
* **true proper nouns** — "Sếp Dũng", "VOV3", "Zalo": the word's identity depends on
  the convention, and no lowercase fallback preserves it.
* **fuzzy arguments** — never compared byte-exactly, so their case is irrelevant.

The last one is the decisive number: it bounds what a perfect capitalisation model
could ever buy.

Run:  python research/analyze_capitalisation.py --data-dir data/train+public_test
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

from lifemate.data import resolve_split  # noqa: E402

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
def _has_upper(text: str) -> bool:
    """True when the string contains a genuinely uppercase letter.

    A regex character class over the Unicode uppercase block is the obvious approach and
    it is wrong: ``À-Ỷ`` spans lowercase Vietnamese letters too (``ă`` is U+0103, inside
    the range), so it matches unaccented text. Python's ``str.isupper`` per character is
    the correct test.
    """
    return any(ch.isupper() for ch in text)
ISO_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _tool_specs(catalog: dict) -> dict[str, dict]:
    """Every tool: its fuzzy argument keys and its property specs."""
    out: dict[str, dict] = {}
    for app in catalog.get("apps", []) or []:
        for tool in app.get("tools", []) or []:
            props = (tool.get("parameters") or {}).get("properties") or {}
            fuzzy = {
                key for key, spec in props.items()
                if (spec.get("x-match") or "").lower() == "fuzzy"
            }
            out[tool["name"]] = {"props": props, "fuzzy": fuzzy}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    specs = _tool_specs(split.catalog())

    buckets: dict[str, collections.Counter] = {
        name: collections.Counter() for name in
        ("exact_case_only", "exact_proper", "exact_other", "fuzzy", "timestamp")
    }
    proper_values: collections.Counter = collections.Counter()
    case_only_values: collections.Counter = collections.Counter()
    exact_slots: collections.Counter = collections.Counter()

    for sample in split.iter_samples():
        for call in sample.get("label", {}).get("tool_calls") or []:
            tool = call.get("name") or ""
            spec = specs.get(tool) or {"props": {}, "fuzzy": set()}
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str):
                    continue
                if ISO_LIKE.match(value):
                    buckets["timestamp"][f"{tool}.{key}"] += 1
                    continue
                if not _has_upper(value):
                    continue
                if key in spec["fuzzy"]:
                    buckets["fuzzy"][f"{tool}.{key}"] += 1
                    continue
                buckets["exact_other"][f"{tool}.{key}"] += 1
                exact_slots[f"{tool}.{key}"] += 1
                # case-only: lowercase preserves the identity but not the bytes
                if value.lower() != value:
                    proper_values[value] += 1
                else:
                    case_only_values[value] += 1

    # A proper-noun value's tokens: which are ambiguous (seen both ways in transcripts)?
    lower_seen: collections.Counter = collections.Counter()
    upper_seen: collections.Counter = collections.Counter()
    for sample in split.iter_samples():
        for word in WORD.findall(sample.get("audio_text") or ""):
            if word[0].isupper():
                upper_seen[word.lower()] += 1
            else:
                lower_seen[word.lower()] += 1

    ambiguous: collections.Counter = collections.Counter()
    for value, count in proper_values.items():
        for word in WORD.findall(value):
            if lower_seen[word.lower()] and upper_seen[word.lower()]:
                ambiguous[word] += count

    report = {
        "populations": {k: sum(v.values()) for k, v in buckets.items()},
        "exact_slots": exact_slots.most_common(20),
        "top_proper_noun_values": proper_values.most_common(30),
        "top_case_only_values": case_only_values.most_common(10),
        "ambiguous_tokens_inside_proper_nouns": ambiguous.most_common(25),
        "distinct_transcript_tokens": len(set(lower_seen) | set(upper_seen)),
        "transcript_tokens_ever_lowercase": len(lower_seen),
        "transcript_tokens_only_capitalised": sum(
            1 for k in set(lower_seen) | set(upper_seen)
            if upper_seen[k] and not lower_seen[k]
        ),
        "transcript_tokens_seen_in_both_cases": sum(
            1 for k in set(lower_seen) | set(upper_seen)
            if lower_seen[k] and upper_seen[k]
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
