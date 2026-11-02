"""What the gold says about the idioms the normaliser gets wrong.

Every fix to the normaliser needs a convention, and the only authority is the gold:
what does the gold trajectory *do* with "8 giờ kém 15", with "năm chục nghìn", with
"hai lăm"? This script pairs each idiom with the gold arguments of its sample, so the
convention is read off the data rather than assumed from intuition.

Run:  python research/analyze_idiom_conventions.py --data-dir data/train+public_test
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

IDIOMS = {
    "clock_kem": re.compile(r"(?<![^\W_])kém(?![^\W_])"),
    "clock_ruoi": re.compile(r"(?<![^\W_])rưỡi(?![^\W_])"),
    "clock_ruoi_no_gio": re.compile(
        r"(?<![^\W_])(\d{1,2}|[a-zà-ỹ]+)\s+rưỡi(?![^\W_])"),
    "number_chuc": re.compile(r"(?<![^\W_])chục(?![^\W_])"),
    "number_hai_lam": re.compile(r"(?<![^\W_])hai lăm(?![^\W_])"),
    "number_muoi_lam": re.compile(r"(?<![^\W_])mười lăm(?![^\W_])"),
    "number_ti": re.compile(r"(?<![^\W_])tỉ(?![^\W_])"),
    "number_linh": re.compile(r"(?<![^\W_])linh(?![^\W_])"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    out: dict[str, list] = collections.defaultdict(list)
    counts: collections.Counter = collections.Counter()

    for sample in split.iter_samples():
        text = sample.get("audio_text") or ""
        lowered = text.lower()
        calls = sample.get("label", {}).get("tool_calls") or []
        arguments = []
        for call in calls:
            for key, value in (call.get("arguments") or {}).items():
                if isinstance(value, (str, int, float)):
                    arguments.append(f"{key}={value!r}")
        for name, pattern in IDIOMS.items():
            if not pattern.search(lowered):
                continue
            counts[name] += 1
            if len(out[name]) < 12:
                out[name].append({
                    "id": sample["id"],
                    "text": text,
                    "gold_arguments": arguments,
                })

    report = {
        "counts": {name: counts[name] for name in IDIOMS},
        "examples": {name: out[name] for name in IDIOMS},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
