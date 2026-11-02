"""Measure the argument-level capitalisation rule against the shipped pipeline.

The review proposed a rule instead of a context model: where the gold capitalises the
first letter of an argument in *every* case, capitalise it unconditionally. The
measurement in ``analyze_argument_case.py`` found 13 such slots, the largest being
``contacts.search.query`` with 204 arguments — over half of the capitalised
exactly-compared population.

This script measures the rule where it matters: through the official metric, on top of
each transcript the pipeline can produce.

    rule module           capitalize.restore_capitals, the rule-only pipeline
    context model         the shipped case model, 98.626 points
    rule + capital slot   the rule, then the slot rule on the argument
    model + capital slot  the shipped model, then the slot rule on the argument

Only exactly-compared string arguments are affected: a fuzzy argument is matched on
token F1, which lowercases both sides, so a capital changes nothing there.

The rule is applied to the *argument the pipeline would emit*, which is the spelling
the transcript gives it. A slot rule that broke an argument it was not meant to fix
would show up here as a loss.

Run:  python research/measure_capital_slot.py --data-dir data/train+public_test
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
from lifemate.case_model import load_case_model, refine_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import (  # noqa: E402
    fuzzy_arguments,
    replay_hash,
    sample_score,
)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _spelling(text: str, value: str) -> str | None:
    """How the given text spells the value, if it contains it at all."""
    at = text.lower().find(value.lower())
    return text[at:at + len(value)] if at >= 0 else None


def _capitalise_first(text: str) -> str:
    for position, char in enumerate(text):
        if char.isalpha():
            return text[:position] + char.upper() + text[position + 1:]
    return text


# The slots where the gold capitalises the first letter in every case. Measured over
# the train split by analyze_argument_case.py (n >= 5, p_upper = 1.0 for each).
CAPITAL_SLOTS = {
    "contacts.search.query",
    "household_chores.assign.assignee",
    "library.search.query",
    "exchange_rate.get_rate.from_currency",
    "exchange_rate.get_rate.to_currency",
    "exchange_rate.convert.from_currency",
    "exchange_rate.convert.to_currency",
    "lottery.check_ticket.province",
    "lottery.get_results.province",
    "loyalty.get_points.program",
    "flights.search.destination",
    "flights.search.origin",
    "package_tracking.reschedule_delivery.tracking_id",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    model = load_case_model()
    samples = [s for s in split.iter_samples() if s.get("audio_text")]

    totals = collections.Counter()
    stats = collections.Counter()
    scored = 0

    for sample in samples:
        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue
        scored += 1

        upper = sample["audio_text"].upper()
        transcripts = {
            "rule": restore_capitals(upper),
            "model": refine_capitals(upper, model) if model else restore_capitals(upper),
        }

        gold_hash = replay_hash(split, sample["id"], calls)
        totals["gold"] += sample_score(calls, calls, gold_hash, gold_hash, fuzzy)

        for base, text in transcripts.items():
            for use_slot_rule in (False, True):
                label = base + ("+slot" if use_slot_rule else "")
                variant = []
                for call in calls:
                    tool = call.get("name") or ""
                    fuzzy_keys = fuzzy.get(tool, set())
                    arguments = dict(call.get("arguments") or {})
                    for key, value in arguments.items():
                        if not isinstance(value, str) or key in fuzzy_keys:
                            continue
                        if _looks_like_timestamp(value) or not _has_upper(value):
                            continue
                        spoken = _spelling(text, value)
                        if spoken is None:
                            continue
                        if use_slot_rule and f"{tool}.{key}" in CAPITAL_SLOTS:
                            spoken = _capitalise_first(spoken)
                        arguments[key] = spoken
                    variant.append({"name": tool, "arguments": arguments})

                pred_hash = replay_hash(split, sample["id"], variant)
                totals[label] += sample_score(variant, calls, pred_hash, gold_hash, fuzzy)

                # the argument-level accounting, for the shipped pipeline's own output
                if base == "model":
                    for call in calls:
                        tool = call.get("name") or ""
                        fuzzy_keys = fuzzy.get(tool, set())
                        for key, value in (call.get("arguments") or {}).items():
                            if not isinstance(value, str) or key in fuzzy_keys:
                                continue
                            if _looks_like_timestamp(value) or not _has_upper(value):
                                continue
                            current = _spelling(text, value)
                            if current is None:
                                continue
                            if current == value:
                                stats["already_right"] += 1
                                continue
                            stats["currently_wrong"] += 1
                            fixed = _capitalise_first(current)
                            if f"{tool}.{key}" in CAPITAL_SLOTS:
                                if fixed == value:
                                    stats["slot_rule_recovers"] += 1
                                else:
                                    stats["slot_rule_misses"] += 1

    n = scored or 1
    report = {
        "samples_scored": scored,
        "metric_score": {
            key: round(100.0 * totals[key] / n, 3)
            for key in ("gold", "rule", "model", "rule+slot", "model+slot")
        },
        "argument_level_on_the_shipped_model": {
            "already_right": stats["already_right"],
            "currently_wrong": stats["currently_wrong"],
            "recovered_by_the_slot_rule": stats["slot_rule_recovers"],
            "slot_rule_did_not_recover": stats["slot_rule_misses"],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
