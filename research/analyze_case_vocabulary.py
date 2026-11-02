"""Can a train-derived vocabulary close the capitalisation gap, and does it transfer?

The measurement so far says the remaining case-model misses are 90/96 "ambiguous" — a
token seen in both cases in train ("Minh" the contact vs "minh" the pronoun). A *context*
model could resolve that. But there is a cheaper candidate: a **vocabulary of the exact
strings the gold trajectories use**, derived from train, and applied as a snap.

The experiment:

  1. build a snap table from train's gold arguments: fold(canonical) -> canonical, per
     slot family (a contact name, a currency code, an airport code, ...)
  2. apply it to the *restored transcript* and to the proposed argument, and measure
     how many of the 396 exactly-compared capitalised arguments it recovers
  3. **held-out check**: build the table on one half of train, evaluate on the other,
     because the public test has no gold transcripts and the table must transfer
  4. **transfer check**: compare the contact names, currency codes and airport codes in
     the *public test* environment with the ones the train-derived table knows. If the
     public test uses names train never saw, the table cannot be the fix.

The environments are read here for analysis only. The shipped agent never reads them:
it gets everything through tool calls (docs/VERIFIED_FACTS.md, the read/write rules).

Run:  python research/analyze_case_vocabulary.py --data-dir data/train+public_test
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

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _fold(text: str) -> str:
    return " ".join(text.lower().split())


def _slot_family(slot: str) -> str:
    """The slot family: the tool, without the argument name.

    "contacts.search.query" -> "contacts.search". The argument name rarely changes the
    spelling, and grouping by tool keeps the table small enough to transfer.
    """
    return slot.rsplit(".", 1)[0]


def build_table(samples) -> dict[str, dict[str, str]]:
    """{slot_family: {folded_value: canonical_value}} from gold arguments."""
    table: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for sample in samples:
        for call in sample.get("label", {}).get("tool_calls") or []:
            tool = call.get("name") or ""
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not _has_upper(value):
                    continue
                if _looks_like_timestamp(value):
                    continue
                family = _slot_family(f"{tool}.{key}")
                table[family].setdefault(_fold(value), value)
    return dict(table)


def apply_table(table, slot: str, value: str) -> str | None:
    """The canonical spelling for a value, or None when the table has no opinion."""
    family = _slot_family(slot)
    return (table.get(family) or {}).get(_fold(value))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = list(split.iter_samples())

    # --- 1. the table, built on ALL of train -----------------------------------
    table = build_table(samples)

    # --- 2. recovery on train, and the held-out check ---------------------------
    half = len(samples) // 2
    table_a = build_table(samples[:half])
    table_b = build_table(samples[half:])

    stats = collections.Counter()
    for index, sample in enumerate(samples):
        audio_text = sample.get("audio_text") or ""
        restored = restore_capitals(audio_text.upper())
        held_out = table_b if index < half else table_a
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
                stats["total"] += 1

                # what the pipeline produces today
                at = restored.lower().find(value.lower())
                current = restored[at:at + len(value)] if at >= 0 else None
                if current == value:
                    stats["current_hit"] += 1
                else:
                    stats["current_miss"] += 1

                # the full-table snap: does the table even contain this value?
                if apply_table(table, slot, value) is not None:
                    stats["table_knows_value"] += 1
                # the held-out snap: the table built without this sample
                if apply_table(held_out, slot, value) is not None:
                    stats["heldout_table_knows_value"] += 1

                # the realistic snap: the pipeline's own spelling, snapped through the
                # held-out table. The table maps a folded value, so the pipeline's
                # spelling of the same words lands on the same key.
                if current is not None:
                    snapped = apply_table(held_out, slot, current)
                    if snapped == value:
                        stats["heldout_snap_hit"] += 1
                    elif snapped is not None:
                        stats["heldout_snap_wrong"] += 1
                    else:
                        stats["heldout_snap_miss"] += 1

    # --- 3. transfer check: the public test environment vs the train table ------
    transfer = {}
    public_root = REPO_ROOT / "data" / "train+public_test" / "public_test"
    train_root = REPO_ROOT / "data" / "train+public_test" / "train"
    for label, root in (("train", train_root), ("public_test", public_root)):
        env_dir = root / f"{label}_environment"
        names: set[str] = set()
        codes: set[str] = set()
        if env_dir.is_dir():
            for path in env_dir.glob("*/state.json"):
                try:
                    text = path.read_text(encoding="utf-8")
                except Exception:
                    continue
                for match in re.finditer(r'"name"\s*:\s*"([^"]*[A-Z][^"]*)"', text):
                    names.add(match.group(1))
                for match in re.finditer(r'"(?:code|currency|airport|province)"\s*:\s*"([^"]+)"', text):
                    codes.add(match.group(1))
        known = set()
        for family_map in table.values():
            known.update(family_map.values())
        transfer[label] = {
            "capitalised_names_in_environment": len(names),
            "codes_in_environment": len(codes),
            "names_not_in_train_table": len(names - known),
            "codes_not_in_train_table": len(codes - known),
            "examples_names_unknown": sorted(names - known)[:20],
            "examples_codes_unknown": sorted(codes - known)[:20],
        }

    total = stats["total"] or 1
    report = {
        "train_split": {
            "exactly_compared_capitalised_arguments": stats["total"],
            "current_rules_hit": stats["current_hit"],
            "current_rules_miss": stats["current_miss"],
            "current_rules_hit_percent": round(100.0 * stats["current_hit"] / total, 1),
            "table_knows_the_value": stats["table_knows_value"],
            "heldout_table_knows_the_value": stats["heldout_table_knows_value"],
            "heldout_snap_hit": stats["heldout_snap_hit"],
            "heldout_snap_wrong": stats["heldout_snap_wrong"],
            "heldout_snap_miss": stats["heldout_snap_miss"],
            "heldout_snap_hit_plus_current_hit": stats["heldout_snap_hit"] + stats["current_hit"],
            "heldout_snap_hit_plus_current_hit_percent": round(
                100.0 * (stats["heldout_snap_hit"] + stats["current_hit"]) / total, 1),
        },
        "table_size": {family: len(m) for family, m in sorted(
            table.items(), key=lambda kv: -len(kv[1]))[:12]},
        "transfer_to_public_test": transfer,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
