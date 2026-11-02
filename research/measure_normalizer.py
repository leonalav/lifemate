"""Measure the deterministic resolvers against the gold trajectories.

Two properties are checked on the train split:

1. **Safety** — the normaliser must be a no-op on every gold argument. If it rewrites
   a correct gold value, it is corrupting the label space and the whole design is
   wrong.
2. **Utility** — for every gold date/time argument, the spoken phrase that produced it
   must resolve to the gold value. This measures whether the deterministic layer knows
   the Vietnamese conventions as well as the annotators do.

Run:

    python -m research.measure_normalizer --data-dir data/train+public_test
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime

from lifemate.agent.guard import Guard
from lifemate.agent.loop import _sample_clock
from lifemate.data import resolve_split
from lifemate.text import (
    normalize_date_value,
    normalize_number_value,
    resolve_date_expression,
    spoken_time_to_clock,
)


def _properties(catalog: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for app in catalog.get("apps", []) or []:
        for tool in app.get("tools", []) or []:
            out[tool["name"]] = (tool.get("parameters") or {}).get("properties") or {}
    return out


_SPOKEN_NUMBER = (r"\d{1,2}|mười|một|mốt|hai|ba|bốn|năm|sáu|bảy|tám|chín|không|mươi")
_ANAPHOR = re.compile(r"hôm đó|hôm đấy|hôm ấy|ngày đó|giờ đó|lúc đó|hôm trước|"
                      r"hôm kia|ngày ấy|cùng hôm|hôm này", re.IGNORECASE)
_DATE_KEYWORD = re.compile(
    r"\b(?:mai|hôm|ngày|ngay|nay|thứ\s+\w+|chủ\s+nhật|tháng|\d{1,2}\s*/\s*\d{1,2})\b",
    re.IGNORECASE)
_TIME_EXPRESSION = re.compile(
    rf"(?:{_SPOKEN_NUMBER})(?:\s+{_SPOKEN_NUMBER}){{0,2}}\s*giờ"
    rf"(?:\s*(?:rưỡi|\d{{1,2}}\s*phút))?"
    rf"(?:\s*(?:sáng|trưa|chiều|tối|đêm|khuya))?",
    re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"[,;.]|\brồi\b|\bvới\b|\bthì\b|\bnhưng\b|\bvà\b")


def _spoken_candidates(transcript: str, now: datetime) -> list[tuple[str, str]]:
    """Every (kind, value) the resolvers can produce from one transcript.

    A transcript holds several times ("sáu giờ rưỡi sáng nay" and "bảy giờ tối mai"),
    so each time expression and each day phrase is resolved on its own.
    """
    candidates: list[tuple[str, str]] = []
    text = transcript or ""
    for match in _TIME_EXPRESSION.finditer(text):
        clock = spoken_time_to_clock(match.group(0))
        if clock:
            candidates.append(("time", clock))
    for clause in _CLAUSE_SPLIT.split(text):
        clause = clause.strip()
        if not clause:
            continue
        for match in _DATE_KEYWORD.finditer(clause):
            resolved = resolve_date_expression(clause[match.start():], now)
            if resolved:
                candidates.append(("date", resolved))
    return candidates


_UNITS_VN = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _spoken_number(value: int) -> str:
    """The Vietnamese reading of an integer: 26 -> 'hai mươi sáu'."""
    if value < 10:
        return _UNITS_VN[value]
    if value < 20:
        rest = value - 10
        return "mười" if rest == 0 else "mười " + ("mốt" if rest == 1 else _UNITS_VN[rest])
    if value < 100:
        tens, rest = divmod(value, 10)
        return f"{_UNITS_VN[tens]} mươi" + (
            "" if rest == 0 else " " + ("mốt" if rest == 1 else _UNITS_VN[rest]))
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        return f"{_UNITS_VN[hundreds]} trăm" + (
            "" if rest == 0 else " " + _spoken_number(rest))
    if value < 1_000_000:
        thousands, rest = divmod(value, 1000)
        return f"{_spoken_number(thousands)} nghìn" + (
            "" if rest == 0 else " " + _spoken_number(rest))
    millions, rest = divmod(value, 1_000_000)
    return f"{_spoken_number(millions)} triệu" + (
        "" if rest == 0 else " " + _spoken_number(rest))


def _spoken_clock(hour: int, minute: int) -> str:
    """The Vietnamese reading of a clock time: (19, 0) -> 'bảy giờ tối'."""
    if hour < 12:
        words = f"{_spoken_number(hour)} giờ sáng"
    elif hour == 12:
        words = "mười hai giờ trưa"
    elif hour < 18:
        words = f"{_spoken_number(hour - 12)} giờ chiều"
    else:
        words = f"{_spoken_number(hour - 12)} giờ tối"
    if minute == 30:
        return words.replace(" giờ", " giờ rưỡi")
    if minute:
        return f"{words} {_spoken_number(minute)} phút"
    return words


def _numeric_defects(value) -> list[str]:
    """The plausible ways a model could spell a correct numeric argument."""
    if not isinstance(value, int) or isinstance(value, bool):
        return []
    defects = []
    if 0 < value <= 100:
        defects.append(f"{value} phần trăm")
        defects.append(f"{value}%")
    if value >= 1000:
        defects.append(f"{value:,}".replace(",", "."))
        defects.append(f"{value} đồng")
    defects.append(_spoken_number(value))
    return defects


def _date_defects(value: str, fmt: str) -> list[str]:
    """The plausible ways a model could spell a correct date/time argument."""
    defects = []
    if fmt == "date-time":
        match = re.match(r"^(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})([+-]\d{2}:\d{2})$",
                         value)
        if not match:
            return defects
        day, hour, minute, _second, offset = match.groups()
        defects.append(f"{day} {hour}:{minute}")
        defects.append(f"{day}T{hour}:{minute}")
        defects.append(f"{day}T{hour}:{minute}:00")
        year, month, dom = day.split("-")
        defects.append(f"{_spoken_clock(int(hour), int(minute))} {int(dom)}/{int(month)}")
    elif fmt == "date":
        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
        if not match:
            return defects
        year, month, dom = match.groups()
        defects.append(f"{int(dom)}/{int(month)}")
        defects.append(f"{int(dom)} tháng {int(month)}")
    elif fmt == "time":
        match = re.match(r"^(\d{2}):(\d{2})$", value)
        if not match:
            return defects
        hour, minute = int(match.group(1)), int(match.group(2))
        defects.append(_spoken_clock(hour, minute))
    return defects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    properties = _properties(split.catalog())

    checked_numeric = 0
    numeric_changed = []
    checked_dates = 0
    date_changed = []
    utility_total = 0
    utility_hit = 0
    utility_misses: list[dict] = []
    time_misses: list[dict] = []
    repair_total = 0
    repair_hit = 0
    repair_misses: list[dict] = []
    time_total = 0
    time_hit = 0

    for sample in split.iter_samples():
        now, tz_offset = _sample_clock(sample)
        transcript = sample.get("audio_text") or ""
        calls = sample.get("label", {}).get("tool_calls") or []
        for call in calls:
            tool = call.get("name") or ""
            props = properties.get(tool, {})
            for key, value in (call.get("arguments") or {}).items():
                spec = props.get(key) or {}
                declared = spec.get("type")
                fmt = spec.get("format")
                if declared in ("integer", "number"):
                    checked_numeric += 1
                    repaired = normalize_number_value(value, declared)
                    if repaired is not None and repaired != value:
                        numeric_changed.append(
                            {"id": sample["id"], "tool": tool, "key": key,
                             "gold": value, "repaired": repaired})
                elif declared == "string" and fmt in ("date", "date-time", "time"):
                    checked_dates += 1
                    repaired = normalize_date_value(value, fmt, now, tz_offset)
                    if repaired is not None and repaired != value:
                        date_changed.append(
                            {"id": sample["id"], "tool": tool, "key": key,
                             "gold": value, "repaired": repaired})
                    # Utility: the spoken phrase must resolve to the gold value.
                    dates = {c for kind, c in _spoken_candidates(transcript, now)
                             if kind == "date"}
                    clocks = {c for kind, c in _spoken_candidates(transcript, now)
                              if kind == "time"}
                    if fmt in ("date", "date-time"):
                        gold_date = value[:10] if fmt == "date-time" else value
                        utility_total += 1
                        if gold_date in dates:
                            utility_hit += 1
                        else:
                            anaphoric = bool(_ANAPHOR.search(transcript))
                            utility_misses.append(
                                {"id": sample["id"], "tool": tool, "gold": value,
                                 "anaphoric": anaphoric,
                                 "transcript": transcript[:160]})
                    if fmt in ("time", "date-time"):
                        gold_time = value[11:16] if fmt == "date-time" else value[:5]
                        time_total += 1
                        if gold_time in clocks:
                            time_hit += 1
                        else:
                            time_misses.append(
                                {"id": sample["id"], "tool": tool, "gold": value,
                                 "transcript": transcript[:160]})

                    # Repair: the guard must restore the gold value from every
                    # plausible spelling defect the model could emit.
                    if declared in ("integer", "number"):
                        defects = [(d, value) for d in _numeric_defects(value)]
                    else:
                        defects = [(d, value) for d in _date_defects(value, fmt)]
                    guard = Guard(split, now=now, tz_offset=tz_offset)
                    for defect, expected in defects:
                        repair_total += 1
                        repaired, _ = guard.normalize_values(tool, {key: defect})
                        got = repaired.get(key)
                        if got == expected:
                            repair_hit += 1
                        else:
                            repair_misses.append(
                                {"id": sample["id"], "tool": tool, "key": key,
                                 "defect": defect, "gold": expected, "repaired": got})

    report = {
        "split": split.name,
        "samples": len(list(split.iter_samples())),
        "safety": {
            "numeric_arguments_checked": checked_numeric,
            "numeric_arguments_rewritten": len(numeric_changed),
            "date_arguments_checked": checked_dates,
            "date_arguments_rewritten": len(date_changed),
            "examples": (numeric_changed + date_changed)[:10],
        },
        "utility": {
            "date_values_resolved_from_the_transcript": f"{utility_hit}/{utility_total}",
            "time_values_resolved_from_the_transcript": f"{time_hit}/{time_total}",
            "date_misses_needing_context":
                sum(1 for m in utility_misses if m["anaphoric"]),
            "date_misses_other": sum(1 for m in utility_misses if not m["anaphoric"]),
            "date_misses": utility_misses[:6],
            "time_misses": time_misses[:6],
        },
        "repair": {
            "defects_injected": repair_total,
            "defects_repaired_to_the_gold_value": f"{repair_hit}/{repair_total}",
            "misses": repair_misses[:10],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
