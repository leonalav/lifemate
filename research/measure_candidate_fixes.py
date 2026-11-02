"""Two candidate fixes, measured end to end through the metric.

**A. Argument-level capitalisation.** The review proposed a rule instead of a model:
if the first letter of an argument is capitalised in 100% of the gold for that
(tool, argument), capitalise it unconditionally. The measurement in
``analyze_argument_case.py`` found 13 slots where the capital is universal, the
largest being ``contacts.search.query`` with 204 arguments — more than half of the
capitalised exactly-compared population.

This script measures what the rule *recovers*: the pipeline's current spelling of the
argument (the rule module's restoration of the transcript) against the gold, and then
the same with the rule applied. Only exactly-compared arguments are counted, because a
fuzzy argument is matched on token F1 and a capital changes nothing there.

**B. Tone-mark canonicalisation.** "hòa" and "hoà" are pronounced identically, so an ASR
cannot tell them apart, and the metric compares them as different tokens
(``token_f1("hòa", "hoà")`` is 0.0). The gold transcripts prefer one spelling — the
measurement says "hoà", 52 times against 2 — so the other spelling is a pure loss.
This measures how many exactly-compared arguments carry a tone-movable word, and how
many are recovered by rewriting the pipeline's spelling to the gold's dominant one.

Both measurements run the argument through the official metric, so a rule that breaks
an argument it was not meant to fix shows up as a loss rather than as a silent gain.

Run:  python research/measure_candidate_fixes.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import (  # noqa: E402
    fuzzy_arguments,
    replay_hash,
    sample_score,
)
from train_case_model import _is_cap  # noqa: E402,F401


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")

# The words whose tone mark can sit on either vowel, and the spelling the gold uses.
# The gold's preference is measured, not assumed: see analyze_argument_case.py.
TONE_GOLD = {
    "hòa": "hoà",
    "thủy": "thuỷ",
    "qúy": "quý",
    "lí": "lý",
    "sác": "sắc",
    "tông": "tổng",
    "gía": "giá",
}

# The slots where the capital is universal in the gold. Measured by
# analyze_argument_case.py over the train split; each is n >= 5 with p_upper = 1.0.
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


def _spelling(text: str, value: str) -> str | None:
    """How the given text spells the value, if it contains it at all."""
    at = text.lower().find(value.lower())
    return text[at:at + len(value)] if at >= 0 else None


def _canonical_tone(text: str) -> str:
    """Rewrite every tone-movable word to the spelling the gold uses."""
    out = text
    for wrong, right in TONE_GOLD.items():
        out = re.sub(rf"(?<![^\W_]){re.escape(wrong)}(?![^\W_])", right, out,
                     flags=re.IGNORECASE)
    return out


def _capitalise_first(text: str) -> str:
    for position, char in enumerate(text):
        if char.isalpha():
            return text[:position] + char.upper() + text[position + 1:]
    return text


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = [s for s in split.iter_samples() if s.get("audio_text")]

    stats = collections.Counter()
    totals = collections.Counter()
    scored = 0
    gains: list[tuple[float, str, str, str, str, str]] = []

    for sample in samples:
        audio_text = sample["audio_text"]
        upper = audio_text.upper()
        rule_text = restore_capitals(upper)

        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue
        scored += 1

        gold_hash = replay_hash(split, sample["id"], calls)
        totals["gold"] += sample_score(calls, calls, gold_hash, gold_hash, fuzzy)

        for label, transform in (
            ("rule", lambda t: t),
            ("capital_slot", None),
            ("tone", _canonical_tone),
            ("capital_slot+tone", None),
        ):
            if label == "capital_slot":
                def transform(text, _calls=calls):
                    return text
            if label == "capital_slot+tone":
                def transform(text, _calls=calls):
                    return text

            variant = []
            for call in calls:
                arguments = dict(call.get("arguments") or {})
                tool = call.get("name") or ""
                fuzzy_keys = fuzzy.get(tool, set())
                for key, value in arguments.items():
                    if not isinstance(value, str) or key in fuzzy_keys:
                        continue
                    if _looks_like_timestamp(value) or not _has_upper(value):
                        continue
                    spoken = _spelling(transform(rule_text), value)
                    if spoken is None:
                        continue
                    if label in ("capital_slot", "capital_slot+tone"):
                        if f"{tool}.{key}" in CAPITAL_SLOTS:
                            spoken = _capitalise_first(spoken)
                    arguments[key] = spoken
                variant.append({"name": tool, "arguments": arguments})

            pred_hash = replay_hash(split, sample["id"], variant)
            score = sample_score(variant, calls, pred_hash, gold_hash, fuzzy)
            totals[label] += score
            if label in ("capital_slot", "capital_slot+tone") and score > 0:
                for call in calls:
                    tool = call.get("name") or ""
                    fuzzy_keys = fuzzy.get(tool, set())
                    for key, value in (call.get("arguments") or {}).items():
                        if not isinstance(value, str) or key in fuzzy_keys:
                            continue
                        if _looks_like_timestamp(value) or not _has_upper(value):
                            continue
                        current = _spelling(rule_text, value)
                        if current is None or current == value:
                            continue
                        if label == "capital_slot":
                            fixed = _capitalise_first(current)
                            if fixed == value:
                                stats["capital_slot_recovers"] += 1
                                if len(gains) < 20:
                                    gains.append((score, sample["id"], f"{tool}.{key}",
                                                  current, value, "capital_slot"))
                        else:
                            fixed = _canonical_tone(current)
                            if fixed == value:
                                stats["tone_recovers"] += 1
                                if len(gains) < 20:
                                    gains.append((score, sample["id"], f"{tool}.{key}",
                                                  current, value, "tone"))

            # how many arguments each variant gets wrong that the rule version gets right
            if label == "capital_slot":
                for call in calls:
                    tool = call.get("name") or ""
                    fuzzy_keys = fuzzy.get(tool, set())
                    for key, value in (call.get("arguments") or {}).items():
                        if not isinstance(value, str) or key in fuzzy_keys:
                            continue
                        if _looks_like_timestamp(value) or not _has_upper(value):
                            continue
                        current = _spelling(rule_text, value)
                        if current is None:
                            continue
                        if current == value:
                            stats["already_right"] += 1
                        else:
                            stats["currently_wrong"] += 1
                            fixed = _capitalise_first(current)
                            if f"{tool}.{key}" in CAPITAL_SLOTS and fixed == value:
                                stats["capital_slot_recovers"] += 1
                            tone_fixed = _canonical_tone(current)
                            if tone_fixed == value:
                                stats["tone_recovers"] += 1

    n = scored or 1
    report = {
        "samples_scored": scored,
        "metric_score": {
            "gold": round(100.0 * totals["gold"] / n, 3),
            "current_pipeline": round(100.0 * totals["rule"] / n, 3),
            "with_capital_slot_rule": round(100.0 * totals["capital_slot"] / n, 3),
            "with_tone_canonicalisation": round(100.0 * totals["tone"] / n, 3),
        },
        "argument_level": {
            "already_right": stats["already_right"],
            "currently_wrong": stats["currently_wrong"],
            "recovered_by_the_capital_slot_rule": stats["capital_slot_recovers"],
            "recovered_by_tone_canonicalisation": stats["tone_recovers"],
        },
        "examples": gains[:20],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
