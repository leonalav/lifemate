"""Where does the correct capitalisation live: the transcript, or the environment?

The exact-compared capitalised arguments are dominated by ``contacts.search.query``
(204 of 431). The values are names like "Sếp Dũng" and "Bác Ba". If those names are
*already* in the environment — in the contact list, the household roster, the room list —
then the correct casing is **known**: the model reads it with a tool call and copies it.
No language model, no vocabulary list, no heuristics.

This script answers that by replaying each sample's gold trajectory and comparing each
exact-compared capitalised argument against every string the environment itself returned.

Run:  python research/analyze_case_source.py --data-dir data/train+public_test
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
from lifemate.kit import load_runtime  # noqa: E402

ISO_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def _has_upper(text: str) -> bool:
    """True when the string contains a genuinely uppercase letter.

    ``str.isupper`` per character, because a Unicode uppercase character class also
    matches lowercase Vietnamese letters (``ă`` is U+0103, inside ``À-Ỷ``).
    """
    return any(ch.isupper() for ch in text)


def _tool_specs(catalog: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for app in catalog.get("apps", []) or []:
        for tool in app.get("tools", []) or []:
            props = (tool.get("parameters") or {}).get("properties") or {}
            fuzzy = {
                key for key, spec in props.items()
                if (spec.get("x-match") or "").lower() == "fuzzy"
            }
            out[tool["name"]] = {"props": props, "fuzzy": fuzzy}
    return out


def _walk_strings(node, out: list[str]) -> None:
    """Every string anywhere in a tool result."""
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _walk_strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_strings(value, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    specs = _tool_specs(split.catalog())
    runtime = load_runtime(split)

    buckets = collections.Counter()
    slots = collections.Counter()
    examples: dict[str, list] = collections.defaultdict(list)
    samples_checked = 0

    for sample in split.iter_samples():
        if args.limit and samples_checked >= args.limit:
            break
        calls = sample.get("label", {}).get("tool_calls") or []
        targets = []
        for call in calls:
            tool = call.get("name") or ""
            spec = specs.get(tool) or {"props": {}, "fuzzy": set()}
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not _has_upper(value):
                    continue
                if ISO_LIKE.match(value):
                    continue
                if key in spec["fuzzy"]:
                    buckets["fuzzy_ignored"] += 1
                    continue
                targets.append((tool, key, value))
        if not targets:
            continue

        samples_checked += 1
        # Replay the gold trajectory to collect every string the environment returned.
        env_strings: list[str] = []
        try:
            env = runtime.ToolEnv(split.name, sample["id"], root=split.root)
            for call in calls:
                name = call.get("name")
                arguments = call.get("arguments") or {}
                try:
                    result = env.call(name, arguments)
                except Exception:
                    continue
                _walk_strings(result, env_strings)
        except Exception as exc:  # pragma: no cover - environment quirks
            print(f"[warn] {sample['id']}: {exc}", file=sys.stderr)
            continue

        env_exact: set[str] = set()
        env_lower: dict[str, set[str]] = collections.defaultdict(set)
        for value in env_strings:
            env_exact.add(value)
            env_lower[value.lower()].add(value)
        for tool, key, value in targets:
            slots[f"{tool}.{key}"] += 1
            if value in env_exact:
                # the environment returned this exact string, character for character
                buckets["exact_in_environment"] += 1
            elif env_lower.get(value.lower()):
                # the environment knows the words but with a different spelling
                buckets["case_differs_in_environment"] += 1
                if len(examples["case_differs"]) < 20:
                    examples["case_differs"].append(
                        {"id": sample["id"], "slot": f"{tool}.{key}",
                         "gold": value,
                         "environment": sorted(env_lower[value.lower()])})
            else:
                buckets["absent_from_environment"] += 1
                if len(examples["absent"]) < 20:
                    examples["absent"].append(
                        {"id": sample["id"], "slot": f"{tool}.{key}", "gold": value})

    report = {
        "samples_checked": samples_checked,
        "buckets": dict(buckets),
        "top_slots": slots.most_common(15),
        "case_differs": examples["case_differs"],
        "absent": examples["absent"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
