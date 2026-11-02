"""The headline numbers for the capitalisation modules.

Both modules' docstrings quote figures. This script recomputes them so the numbers can
be checked against the code rather than trusted, and it prints the figures the
docstrings should carry.

Run:  python research/measure_case_headlines.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.case_model import load_case_model, refine_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402
from lifemate.text import recase  # noqa: E402


def _similarity(a: str, b: str) -> float:
    if not a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = list(split.iter_samples())
    texts = [s["audio_text"] for s in samples if s.get("audio_text")]

    def score(fn) -> float:
        return sum(_similarity(fn(t.upper()), t) for t in texts) / len(texts)

    def exact(fn) -> int:
        return sum(1 for t in texts if fn(t.upper()) == t)

    def reproduce(fn) -> tuple[int, int]:
        found = 0
        total = 0
        for sample in samples:
            text = sample.get("audio_text") or ""
            if not text:
                continue
            restored = fn(text.upper())
            for call in sample["label"]["tool_calls"]:
                fuzzy_keys = fuzzy.get(call["name"], set())
                for key, value in (call.get("arguments") or {}).items():
                    if not isinstance(value, str) or key in fuzzy_keys:
                        continue
                    if "search" not in call["name"] and "query" not in key:
                        continue
                    total += 1
                    if value in restored:
                        found += 1
        return found, total

    model = load_case_model()
    lower_found, total = reproduce(lambda t: t.lower())
    sentence_found, _ = reproduce(recase)
    rule_found, _ = reproduce(restore_capitals)
    model_found = 0
    if model is not None:
        model_found, _ = reproduce(lambda t: refine_capitals(t, model))

    report = {
        "transcripts": len(texts),
        "similarity": {
            "lowercase": round(score(lambda t: t.lower()), 4),
            "sentence_case": round(score(recase), 4),
            "rule_module": round(score(restore_capitals), 4),
        },
        "exact_transcripts": {
            "lowercase": exact(lambda t: t.lower()),
            "sentence_case": exact(recase),
            "rule_module": exact(restore_capitals),
        },
        "query_arguments": {
            "total": total,
            "lowercase": {"found": lower_found,
                          "percent": round(100.0 * lower_found / total, 1)},
            "sentence_case": {"found": sentence_found,
                              "percent": round(100.0 * sentence_found / total, 1)},
            "rule_module": {"found": rule_found,
                            "percent": round(100.0 * rule_found / total, 1)},
            "context_model": {"found": model_found,
                              "percent": round(100.0 * model_found / total, 1)},
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
