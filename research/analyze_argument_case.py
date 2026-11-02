"""Three measurements the review asked for, settled against the gold data.

  1. argument-level capitalisation: is a capitalised first letter the *default* for a
     given (tool, argument)? If it is, the rule "capitalise the first character of this
     argument" is worth more than a context model that guesses from ±2 words.

  2. tone-mark placement: the gold transcripts write "hòa" or "hoà", "thủy" or "thuỷ"?
     The fuzzy comparison folds case but not tone-mark position, so a mismatch in an
     exactly-compared argument is a zero. The style has to be counted before it can be
     normalised.

  3. the compliance check: does any shipped artefact carry a name that only the
     *public test* environment knows? The rules forbid tuning on the public test, so a
     shipped table that contains a public-test-only string is a leak, not a feature.

Run:  python research/analyze_argument_case.py --data-dir data/train+public_test
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

from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


# --- 1. argument-level capitalisation ---------------------------------------------

def measure_argument_case(split, samples) -> dict:
    """P(the first letter is uppercase | tool, argument), over exactly-compared strings."""
    fuzzy = fuzzy_arguments(split.catalog())
    per_slot: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for sample in samples:
        for call in sample.get("label", {}).get("tool_calls") or []:
            tool = call.get("name") or ""
            fuzzy_keys = fuzzy.get(tool, set())
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not value.strip():
                    continue
                if key in fuzzy_keys:
                    continue
                if _looks_like_timestamp(value):
                    continue
                first = next((c for c in value if c.isalpha()), "")
                if not first:
                    continue
                slot = f"{tool}.{key}"
                per_slot[slot]["n"] += 1
                per_slot[slot]["upper" if first.isupper() else "lower"] += 1

    rows = []
    for slot, counter in per_slot.items():
        n = counter["n"]
        if n < 5:
            continue
        rows.append({
            "slot": slot,
            "n": n,
            "p_upper": round(counter["upper"] / n, 3),
        })
    rows.sort(key=lambda r: (-r["n"], r["slot"]))
    # A slot where the capital is near-universal is a slot where the rule is safe.
    safe = [r for r in rows if r["p_upper"] >= 0.95]
    return {
        "slots_with_at_least_5_arguments": len(rows),
        "slots_where_the_capital_is_universal": len(safe),
        "arguments_covered_by_those_slots": sum(r["n"] for r in safe),
        "safe_slots": safe[:40],
        "largest_slots": rows[:30],
    }


# --- 2. tone-mark placement ----------------------------------------------------------

# Words whose two spellings differ only in where the tone mark sits. The gold picks one.
TONE_PAIRS = [
    ("hòa", "hoà"),
    ("thủy", "thuỷ"),
    ("quý", "qúy"),
    ("lý", "lí"),
    ("sắc", "sác"),
    ("tổng", "tông"),
    ("giá", "gía"),
    ("học", "hoc"),
]


def measure_tone_marks(samples) -> dict:
    """How the gold spells each word whose tone mark can move."""
    counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for sample in samples:
        text = sample.get("audio_text") or ""
        folded = unicodedata.normalize("NFC", text.lower())
        for a, b in TONE_PAIRS:
            for word in (a, b):
                for match in re.finditer(rf"(?<![^\W_]){re.escape(word)}(?![^\W_])",
                                         folded):
                    counts[a + "|" + b][word] += 1

    rows = []
    for pair, counter in counts.items():
        total = sum(counter.values())
        if total < 3:
            continue
        winner = counter.most_common(1)[0][0]
        rows.append({
            "pair": pair.replace("|", " vs "),
            "n": total,
            "counts": dict(counter),
            "dominant": winner,
            "share": round(counter[winner] / total, 3),
        })
    rows.sort(key=lambda r: -r["n"])
    return {"words_measured": rows}


# --- 3. the compliance check -----------------------------------------------------------

def measure_leak(split) -> dict:
    """Does a shipped artefact carry a string only the public test environment knows?"""
    resources = REPO_ROOT / "src" / "lifemate" / "resources"
    shipped: set[str] = set()
    capitalization = resources / "capitalization.json"
    if capitalization.is_file():
        payload = json.loads(capitalization.read_text(encoding="utf-8"))
        shipped.update((payload.get("proper") or {}).values())
        shipped.update((payload.get("acronyms") or {}).values())

    def environment_names(label: str) -> set[str]:
        root = REPO_ROOT / "data" / "train+public_test" / label
        names: set[str] = set()
        for path in (root / f"{label}_environment").glob("*/state.json"):
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            for match in re.finditer(r'"name"\s*:\s*"([^"]*[A-Z][^"]*)"', text):
                names.add(match.group(1))
        return names

    train_names = environment_names("train")
    public_names = environment_names("public_test")
    public_only = public_names - train_names
    return {
        "shipped_table_entries": len(shipped),
        "public_test_only_environment_names": len(public_only),
        "shipped_entries_matching_a_public_test_only_name": sorted(
            shipped & public_only
        )[:20],
        "shipped_entries_matching_any_public_test_name": len(shipped & public_names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    samples = list(split.iter_samples())

    report = {
        "argument_level_capitalisation": measure_argument_case(split, samples),
        "tone_mark_placement": measure_tone_marks(samples),
        "compliance_shipped_artefacts": measure_leak(split),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
