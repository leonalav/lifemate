"""What is perfect capitalisation worth on the metric?

The question a learned case-restoration model has to answer: if the restoration were
perfect, how much score would it buy? This script measures it directly by perturbing the
gold trajectories and re-scoring them with the official metric.

Three perturbations of the gold, each affecting only the exactly-compared arguments
that carry capitals:

  exact     the gold itself (the ceiling)
  lower     every capitalised value lowercased — what a "no restoration" pipeline scores
  upper     every capitalised value uppercased — what a naive "keep the ASR casing" does

Fuzzy arguments are untouched, timestamps are untouched, and the state hash is recomputed
for each variant so the state gate behaves exactly as the organiser's would.

Run:  python research/analyze_case_ceiling.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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


def _perturb(calls: list[dict], fuzzy: dict[str, set[str]], mode: str) -> list[dict]:
    out = []
    for call in calls:
        arguments = dict(call.get("arguments") or {})
        fuzzy_keys = fuzzy.get(call.get("name") or "", set())
        for key, value in list(arguments.items()):
            if not isinstance(value, str) or key in fuzzy_keys:
                continue
            if _looks_like_timestamp(value):
                continue
            if mode == "lower":
                arguments[key] = value.lower()
            elif mode == "upper":
                arguments[key] = value.upper()
        out.append({"name": call.get("name"), "arguments": arguments})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())

    totals = {"exact": 0.0, "lower": 0.0, "upper": 0.0}
    perfect = {"exact": 0, "lower": 0, "upper": 0}
    samples = 0
    per_sample_diff: list[tuple[float, float, str]] = []

    for sample in split.iter_samples():
        sample_id = sample["id"]
        gold_calls = sample.get("label", {}).get("tool_calls") or []
        if not gold_calls:
            continue
        samples += 1
        gold_hash = replay_hash(split, sample_id, gold_calls)

        scores = {}
        for mode in ("exact", "lower", "upper"):
            calls = gold_calls if mode == "exact" else _perturb(gold_calls, fuzzy, mode)
            pred_hash = replay_hash(split, sample_id, calls)
            score = sample_score(calls, gold_calls, pred_hash, gold_hash, fuzzy)
            scores[mode] = score
            totals[mode] += score
            perfect[mode] += int(score == 1.0)

        if scores["exact"] != scores["lower"]:
            per_sample_diff.append(
                (scores["exact"], scores["lower"], sample_id))

    n = samples or 1
    report = {
        "samples": samples,
        "score": {mode: round(100.0 * total / n, 3) for mode, total in totals.items()},
        "perfect_samples": perfect,
        "value_of_perfect_capitalisation": round(
            100.0 * (totals["exact"] - totals["lower"]) / n, 3),
        "value_of_wrong_case_upper": round(
            100.0 * (totals["exact"] - totals["upper"]) / n, 3),
        "samples_where_case_matters": len(per_sample_diff),
        "worst_hits": sorted(per_sample_diff, key=lambda r: r[1] - r[0])[:15],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
