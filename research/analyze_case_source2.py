"""Are the "absent" values absent from the environment, or merely unread by the gold?

The first pass collected only the strings the *gold trajectory* happened to read. A gold
trajectory that never calls ``news.headlines`` never sees a headline, so a headline that
exists in the environment looks absent.

This script asks the environment directly: it calls the listing tool for the app that
owns each unmatched value and searches the returned catalogue for the value.

Run:  python research/analyze_case_source2.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import resolve_split  # noqa: E402
from lifemate.kit import load_runtime  # noqa: E402

# The read tool that lists an app's catalogue, per app prefix.
CATALOGUE_TOOL = {
    "contacts": "contacts.search",
    "news": "news.headlines",
    "maps": "maps.search_place",
    "podcast_radio": "podcast_radio.search",
    "library": "library.search",
    "loyalty": "loyalty.list_vouchers",
    "lottery": "lottery.get_results",
    "smart_home": "smart_home.get_status",
    "flights": "flights.search",
    "news_": "news.headlines",
}


def _walk_strings(node, out: list[str]) -> None:
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, dict):
        for value in node.values():
            _walk_strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_strings(value, out)


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ids", nargs="*", default=[])
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    runtime = load_runtime(split)

    targets = {
        "train_0608": "Sơ cứu khi cắm trại",
        "train_0642": "hiệu thuốc phố Huế",
        "train_0701": "Phản xạ khi sơ cứu",
        "train_0772": "Kết nối Xanh",
        "train_0922": "Cao Nguyên",
        "train_1025": "Cao Nguyên",
        "train_1071": "Nhà giả kim",
        "train_1118": "TP.HCM",
        "train_1204": "MinhPoint",
        "train_1231": "Rác thải nhựa ở đô thị",
    }
    if args.ids:
        targets = {}
        for sample_id in args.ids:
            sample = split.sample(sample_id)
            for call in sample.get("label", {}).get("tool_calls") or []:
                for key, value in (call.get("arguments") or {}).items():
                    if isinstance(value, str) and _has_upper(value):
                        targets[sample_id] = value

    report: dict[str, dict] = {}
    for sample_id, value in targets.items():
        app = None
        for call in split.sample(sample_id).get("label", {}).get("tool_calls") or []:
            name = call.get("name") or ""
            for prefix, tool in CATALOGUE_TOOL.items():
                if name.startswith(prefix):
                    app = tool
                    break
            if app:
                break
        found = []
        if app:
            try:
                env = runtime.ToolEnv(split.name, sample_id, root=split.root)
                result = env.call(app, {})
                strings: list[str] = []
                _walk_strings(result, strings)
                lowered = value.lower()
                found = sorted({s for s in strings if s.lower() == lowered})
            except Exception as exc:
                found = [f"<error: {exc}>"]
        report[sample_id] = {
            "value": value,
            "catalogue_tool": app,
            "exact_matches_in_environment": found,
            "recoverable": bool(found),
        }

    recoverable = sum(1 for r in report.values() if r["recoverable"])
    print(json.dumps({
        "summary": f"{recoverable}/{len(report)} recoverable from the environment catalogue",
        "detail": report,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
