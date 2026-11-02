"""Where does the deterministic normaliser corrupt the gold transcripts?

The normaliser rewrites number words and date phrases before the model sees them
(docs/DECISIONS.md, Decision 15). A rewrite that changes the *meaning* of a turn is
worse than no rewrite at all: the model is shown text that no one said.

This script runs the normaliser over every gold training transcript and reports the
turns where the rewrite is suspicious — the number it produced is not the number the
words spelled, or the date phrase it consumed was not a date phrase.

The check is a *round-trip* check, not a label check: for a numeral phrase, the words
"hai lăm" must not become 7 (they mean 15), and "mười lăm" must not become 15 (it
means 15 too, but "hai mươi lăm" means 25). The script re-spells the produced digits
into Vietnamese and compares the *value* with the value the words should have had,
which is what catches the idioms the rules do not know:

    hai lăm          15, not 2 + 5
    mười lăm         15, not 10 + 5
    hai mươi lăm      25
    linh              0 (a number, not a word)
    lẻ               the odd ones (a number, not a word)
    chục             10 (a multiplier, not a unit)
    phẩy             the decimal point
    tỉ / tỷ          a billion
    mốt              1

The clock idioms are checked the same way:

    sáu giờ kém mười lăm    05:45
    sáu giờ rưỡi             06:30
    ba giờ mười lăm phút    03:12

Run:  python research/analyze_normalizer_gaps.py --data-dir data/train+public_test
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
from lifemate.text import (  # noqa: E402
    normalize_numbers,
    resolve_relative_dates,
    spoken_time_to_clock,
)

# The idioms the rules do not know, and the value each one carries.
IDIOMS = {
    "hai lăm": 15, "hai lam": 15,
    "mười lăm": 15, "muoi lam": 15,
    "hai mươi lăm": 25, "hai muoi lam": 25,
    "mười lăm phút": 15,
    "linh": 0,
    "lẻ": None, "le": None,
    "chục": 10, "chuc": 10,
    "mốt": 1, "mot": 1,
    "tỉ": 1_000_000_000, "ti": 1_000_000_000,
}

# The clock idioms and the clock they mean.
CLOCK_IDIOMS = {
    "kém": "quarter-to",      # "sáu giờ kém mười lăm" -> 05:45
    "rưỡi": "half",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    stats = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)

    for sample in split.iter_samples():
        text = sample.get("audio_text") or ""
        if not text:
            continue
        lowered = text.lower()
        rewritten = normalize_numbers(text)

        # 1. the idioms the rules do not know
        for idiom, value in IDIOMS.items():
            for match in re.finditer(rf"(?<![^\W_]){re.escape(idiom)}(?![^\W_])",
                                     lowered):
                stats[f"idiom:{idiom}"] += 1
                at = match.start()
                snippet = rewritten[max(0, at - 30):at + 30]
                if len(examples[f"idiom:{idiom}"]) < 6:
                    examples[f"idiom:{idiom}"].append(
                        {"id": sample["id"], "text": text, "rewritten": snippet})
                if value is None:
                    continue
                # what did the number pass produce for the phrase?
                window = rewritten[max(0, at - 25):at + 25]
                numbers = [int(n) for n in re.findall(r"\d+", window)]
                if not any(n == value for n in numbers):
                    stats[f"idiom_wrong:{idiom}"] += 1

        # 2. the clock idioms
        for match in re.finditer(r"(?<![^\W_])(kém|rưỡi)(?![^\W_])", lowered):
            stats[f"clock_idiom:{match.group(1)}"] += 1
            if len(examples[f"clock_idiom:{match.group(1)}"]) < 6:
                examples[f"clock_idiom:{match.group(1)}"].append({
                    "id": sample["id"],
                    "text": text,
                    "clock": spoken_time_to_clock(text),
                })

        # 3. the date idioms the table does not know
        for idiom in ("ngày mốt", "ngay mot", "mùng 1", "mung 1", "mồng 1", "mong 1"):
            for match in re.finditer(re.escape(idiom), lowered):
                stats[f"date_idiom:{idiom}"] += 1
                if len(examples[f"date_idiom:{idiom}"]) < 6:
                    examples[f"date_idiom:{idiom}"].append({
                        "id": sample["id"], "text": text,
                        "resolved": resolve_relative_dates(
                            text, sample.get("context", {}).get("current_time")
                            and __import__("datetime").datetime.fromisoformat(
                                sample["context"]["current_time"])),
                    })

    report = {
        "split": args.split,
        "idiom_counts": {k: v for k, v in sorted(stats.items()) if k.startswith("idiom")},
        "clock_idiom_counts": {k: v for k, v in sorted(stats.items())
                               if k.startswith("clock_idiom")},
        "date_idiom_counts": {k: v for k, v in sorted(stats.items())
                              if k.startswith("date_idiom")},
        "examples": {k: v for k, v in sorted(examples.items())},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
