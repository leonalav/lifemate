"""What is each capitalisation population worth, in points?

The previous script measured the *ceiling* (perfect case vs. all-lowercase: 9.26 points).
This script prices each population separately, holding the trajectory fixed at gold and
changing only the case of the affected argument. That isolates case as the single
variable and turns each population into a point estimate.

For every exactly-compared capitalised gold argument the case is replaced by:

  as_spoken     the spelling that ``restore_capitals`` produces from the uppercased
                transcript — what the pipeline actually submits today
  lowercased    the all-lowercase variant
  left_alone    the gold spelling (the ceiling)

Three submissions are scored per sample: gold-with-``as_spoken``-case,
gold-with-``lowercased``-case, and untouched gold. Each is replayed on a fresh session
so the state gate behaves exactly as the organiser's would.

Run:  python research/analyze_case_price.py --data-dir data/train+public_test
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
from lifemate.metric import (  # noqa: E402
    fuzzy_arguments,
    replay_hash,
    sample_score,
)


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _as_spoken(gold_value: str, restored: str, audio_text: str) -> str | None:
    """The spelling the current pipeline would submit for this value.

    The pipeline restores the case on the transcript and the model copies from it. So
    the value the model would emit is the gold value's words as they appear in the
    restored transcript, keeping the gold's capitalisation where the transcript agrees.
    If the transcript does not contain the words at all, the model has nothing to copy
    and the value is unknowable from the utterance alone.
    """
    words_present = gold_value.lower() in audio_text.lower()
    if not words_present:
        return None
    # find the phrase in the restored transcript, case-insensitively
    index = restored.lower().find(gold_value.lower())
    if index < 0:
        return None
    return restored[index:index + len(gold_value)]


def _variant(calls: list[dict], fuzzy: dict[str, set[str]],
             restored: str, audio_text: str, mode: str) -> list[dict]:
    out = []
    for call in calls:
        arguments = dict(call.get("arguments") or {})
        fuzzy_keys = fuzzy.get(call.get("name") or "", set())
        for key, value in list(arguments.items()):
            if not isinstance(value, str) or key in fuzzy_keys:
                continue
            if _looks_like_timestamp(value) or not _has_upper(value):
                continue
            if mode == "lower":
                arguments[key] = value.lower()
            elif mode == "as_spoken":
                spoken = _as_spoken(value, restored, audio_text)
                if spoken is not None:
                    arguments[key] = spoken
        out.append({"name": call.get("name"), "arguments": arguments})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())

    totals = collections.Counter()
    samples = 0
    lost_samples: list[tuple[float, float, str]] = []

    for sample in split.iter_samples():
        sample_id = sample["id"]
        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue
        audio_text = sample.get("audio_text") or ""
        restored = restore_capitals(audio_text.upper())
        samples += 1

        gold_hash = replay_hash(split, sample_id, calls)
        gold_score = sample_score(calls, calls, gold_hash, gold_hash, fuzzy)
        totals["gold"] += gold_score

        for mode in ("as_spoken", "lower"):
            variant = _variant(calls, fuzzy, restored, audio_text, mode)
            pred_hash = replay_hash(split, sample_id, variant)
            score = sample_score(variant, calls, pred_hash, gold_hash, fuzzy)
            totals[mode] += score
            if mode == "as_spoken" and score < gold_score:
                lost_samples.append((gold_score, score, sample_id))

    n = samples or 1
    report = {
        "samples": samples,
        "score": {k: round(100.0 * v / n, 3) for k, v in totals.items()},
        "price_of_current_pipeline_case": round(
            100.0 * (totals["gold"] - totals["as_spoken"]) / n, 3),
        "price_of_lowercasing_everything": round(
            100.0 * (totals["gold"] - totals["lower"]) / n, 3),
        "samples_lost_by_current_pipeline": len(lost_samples),
        "worst": sorted(lost_samples, key=lambda r: r[1] - r[0])[:12],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
