"""How much headroom is there for a context-aware case-restoration model?

The upper bound was measured: perfect capitalisation is worth 9.26 points over
lowercasing everything. This script measures the part of that bound a *case-restoration
model* could actually capture, and separates it from the part that only a catalogue
lookup can capture.

For each exactly-compared capitalised gold argument we ask:

  A. words_in_transcript_ci
     the value's words occur in the utterance, case-insensitively. Only the case is
     wrong. A case-restoration model operates on exactly this population.

  B. current_rules_hit
     ``restore_capitals`` applied to the uppercased transcript already produces the
     exact gold string. The current pipeline gets this one for free.

  C. needs_catalogue
     the words do not occur in the utterance at all ("USD", "Đồng Nai", "SGN"). No
     amount of case modelling can produce these; they must be looked up.

A - B is the model's headroom. C is out of reach for a case model by construction.

Run:  python research/analyze_case_headroom.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())

    buckets = collections.Counter()
    misses: list[dict] = []
    by_slot = collections.defaultdict(collections.Counter)

    for sample in split.iter_samples():
        sample_id = sample["id"]
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

                slot = f"{tool}.{key}"
                words_present = value.lower() in audio_text.lower()
                hit = value in restored

                if hit:
                    bucket = "current_hit"
                elif words_present:
                    bucket = "model_headroom"
                else:
                    bucket = "needs_catalogue"

                buckets[bucket] += 1
                by_slot[slot][bucket] += 1
                if bucket == "model_headroom" and len(misses) < 40:
                    misses.append({
                        "id": sample_id, "slot": slot, "gold": value,
                        "restored_context": restored[:160],
                    })

    total = sum(buckets.values())
    report = {
        "exact_compared_capitalised_arguments": total,
        "buckets": dict(buckets.most_common()),
        "percent": {k: round(100.0 * v / total, 1) for k, v in buckets.most_common()},
        "model_headroom_count": buckets["model_headroom"],
        "per_slot": {
            slot: dict(counter.most_common())
            for slot, counter in sorted(
                by_slot.items(), key=lambda kv: -sum(kv[1].values()))[:15]
        },
        "model_headroom_examples": misses[:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
