"""Deterministic Vietnamese text normalisation.

51.8% of transcripts contain Vietnamese number words ("hai triệu tám", "một nửa",
"30 phần trăm") while the gold numeric arguments are typed integers
(docs/VERIFIED_FACTS.md §12.5). A single perturbed typed argument costs ~79 F1 points on
average, so this conversion is not cosmetic — it is a gate concern.

Design rules (docs/DECISIONS.md, Decisions 3 and 15):

* Never fold diacritics on values that reach the model: the matcher is diacritic-sensitive.
* Numbers and dates are resolved by code, never by the model.
* The normaliser is pure and deterministic — same input, same output, no model involved.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta

# --- number words -------------------------------------------------------------

UNITS = {
    "không": 0, "khong": 0, "một": 1, "mot": 1, "mốt": 1, "hai": 2, "ba": 3,
    "bốn": 4, "bon": 4, "năm": 5, "nam": 5, "lăm": 5, "lam": 5,
    "sáu": 6, "sau": 6, "bảy": 7, "bay": 7, "tám": 8, "tam": 8, "chín": 9, "chin": 9,
}
TEN = 10
TENS = {"hai mươi": 20, "hai muoi": 20, "ba mươi": 30, "ba muoi": 30,
        "bốn mươi": 40, "bon muoi": 40, "năm mươi": 50, "nam muoi": 50,
        "sáu mươi": 60, "sau muoi": 60, "bảy mươi": 70, "bay muoi": 70,
        "tám mươi": 80, "tam muoi": 80, "chín mươi": 90, "chin muoi": 90}

SCALES = {"nghìn": 1000, "nghin": 1000, "ngàn": 1000, "ngan": 1000,
          "triệu": 1_000_000, "trieu": 1_000_000,
          "tỷ": 1_000_000_000, "ty": 1_000_000_000}

_HUNDRED = "trăm"

# One token of a numeral.
_TOKEN = (r"hai mươi|hai muoi|ba mươi|ba muoi|bốn mươi|bon muoi|năm mươi|nam muoi"
          r"|sáu mươi|sau muoi|bảy mươi|bay muoi|tám mươi|tam muoi|chín mươi|chin muoi"
          r"|không|khong|một|mot|mốt|hai|ba|bốn|bon|năm|nam|lăm|lam"
          r"|sáu|sau|bảy|bay|tám|tam|chín|chin"
          r"|mười|muoi|nghìn|nghin|ngàn|ngan|triệu|trieu|tỷ|ty|trăm")

# A single token; the multi-word forms come first so "hai mươi" stays one token.
_SINGLE_TOKEN = re.compile(rf"(?<![^\W_])({_TOKEN})(?![^\W_])")

# A numeral is a *run* of tokens: "một trăm hai mươi ba" is one number, not four.
# The word boundaries are load-bearing. Without them the pass eats the first two
# letters of a longer word: "bao nhiêu" becomes "3o nhiêu" and "vitamin D" becomes
# "vi8in D", because "bao" starts with the token "ba" and "vitamin" with "tam".
_TOKEN_PATTERN = re.compile(rf"(?<![^\W_])(?:{_TOKEN})(?:\s+(?:{_TOKEN}))*(?![^\W_])")
_YEAR = re.compile(r"^\s*(19|20)\d{2}\b")


def _evaluate_phrase(tokens: list[str]) -> int:
    """Evaluate a Vietnamese numeral phrase.

    "một trăm hai mươi ba" -> 123, "hai triệu tám" -> 2000008.
    """
    total = 0
    current = 0
    for token in tokens:
        lowered = token.lower()
        if lowered in TENS:
            current = TENS[lowered] + (current if current else 0)
        elif lowered in ("mười", "muoi"):
            current = TEN + (current if current else 0)
        elif lowered in UNITS:
            current += UNITS[lowered]
        elif lowered == _HUNDRED:
            current = (current if current else 1) * 100
        else:  # a thousand/million/billion scale
            total += (current if current else 1) * SCALES[lowered]
            current = 0
    return total + current


def _tokenize_phrase(phrase: str) -> list[str]:
    """Split a numeral phrase into its tokens ("hai triệu tám" -> ["hai","triệu","tám"])."""
    tokens = [m.group(1).lower() for m in _SINGLE_TOKEN.finditer(phrase)]
    # The phrase must be made up entirely of number tokens.
    if not tokens or " ".join(tokens) != re.sub(r"\s+", " ", phrase.strip().lower()):
        return []
    return tokens


def _parse_number_words(text: str) -> str:
    """Replace runs of Vietnamese number words with their digits."""
    pieces = []
    position = 0
    for match in _TOKEN_PATTERN.finditer(text):
        tokens = _tokenize_phrase(match.group(0))
        if not tokens:
            continue
        # "năm 2026" is a year, not the number five.
        if tokens == ["năm"] or tokens == ["nam"]:
            if _YEAR.match(text[match.end():]):
                continue
        pieces.append(text[position:match.start()])
        pieces.append(str(_evaluate_phrase(tokens)))
        position = match.end()
    pieces.append(text[position:])
    return "".join(pieces)


# Plain digits with Vietnamese thousand separators: 2.000.000, 30.000
_GROUPED_NUMBER = re.compile(r"(?<![\d.,])(\d{1,3}(?:\.\d{3})+)(?![\d.,])")
# Percentages written with a dot separator: 30.000 -> "30 phần trăm" style is 30%
_PERCENT_SIGN = re.compile(r"(\d+(?:[.,]\d+)?)\s*(%|phần trăm|phan tram|%)", re.IGNORECASE)
# "phần trăm" is protected from the number-word pass: "trăm" is the hundred multiplier,
# so "bốn mươi phần trăm" would otherwise be read as "40 phần 100".
_PERCENT_UNIT = re.compile(r"(?i)phần\s*trăm|phan\s*tram")
_PERCENT_SENTINEL = "\x00PCT\x00"
# Fractions: một nửa, một phần ba
FRACTIONS = {
    "một nửa": 0.5, "mot nua": 0.5, "nửa": 0.5, "nua": 0.5, "half": 0.5,
    "một phần ba": 1 / 3, "mot phan ba": 1 / 3,
    "một phần tư": 0.25, "mot phan tu": 0.25,
    "một phần hai": 0.5, "mot phan hai": 0.5,
}

_CURRENCY_UNITS = ("đồng", "dong", "vnd", "vnđ")


def normalize_numbers(text: str) -> str:
    """Convert spoken number forms into typed numbers.

    Handles: number words ("hai triệu tám" -> "2000008"), grouped digits
    ("2.000.000" -> "2000000"), percentages, fractions, and money units.
    """
    if not text:
        return text
    out = text

    # fractions first: they contain number words
    for phrase, value in FRACTIONS.items():
        if phrase in out.lower():
            out = re.sub(re.escape(phrase), _format_decimal(value), out, flags=re.IGNORECASE)

    # "phần trăm" is protected before the number-word pass: "trăm" is the hundred
    # multiplier, so "bốn mươi phần trăm" would be read as "40 phần 100".
    out = _PERCENT_UNIT.sub(_PERCENT_SENTINEL, out)

    # grouped digits: 2.000.000 -> 2000000 (Vietnamese uses '.' as the thousands
    # separator, so it is always a group separator)
    out = _GROUPED_NUMBER.sub(lambda m: m.group(1).replace(".", ""), out)

    # number words
    out = _parse_number_words(out)

    # money units are dropped: "2 triệu đồng" -> "2000000"
    for unit in _CURRENCY_UNITS:
        out = re.sub(rf"(?<=\d)\s*{re.escape(unit)}\b", "", out, flags=re.IGNORECASE)

    # percentages: "30 phần trăm" -> "30%", "30%" -> "30%"
    out = _PERCENT_SIGN.sub(lambda m: f"{m.group(1)}%", out)
    out = out.replace(_PERCENT_SENTINEL, "%")
    out = re.sub(r"\s+%", "%", out)

    return out


def _format_decimal(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


# --- dates --------------------------------------------------------------------

_WEEKDAYS = {
    "thứ hai": 0, "thu hai": 0, "monday": 0,
    "thứ ba": 1, "thu ba": 1, "tuesday": 1,
    "thứ tư": 2, "thu tu": 2, "wednesday": 2,
    "thứ năm": 3, "thu nam": 3, "thursday": 3,
    "thứ sáu": 4, "thu sau": 4, "friday": 4,
    "thứ bảy": 5, "thu bay": 5, "saturday": 5,
    "chủ nhật": 6, "chu nhat": 6, "sunday": 6,
}
_WEEKDAY_PATTERN = "|".join(
    sorted((re.escape(k) for k in _WEEKDAYS), key=len, reverse=True)
)

RELATIVE_DAYS = {
    "hôm nay": 0, "hom nay": 0, "today": 0,
    "ngày mai": 1, "ngay mai": 1, "mai": 1, "tomorrow": 1,
    "ngày kia": 2, "ngay kia": 2, "kia": 2,
    "hôm qua": -1, "hom qua": -1, "yesterday": -1,
    "hôm kia": -2, "hom kia": -2,
    "tuần sau": 7, "tuan sau": 7,
    "tuần này": 0, "tuan nay": 0,
    "tháng sau": 30, "thang sau": 30,
}

# Vietnamese weekday names, Monday first (matches ``datetime.weekday()``).
_VN_WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu",
                 "Thứ Bảy", "Chủ Nhật"]


def _monday_of(day) -> "date":
    """The Monday of the calendar week that contains ``day``."""
    return day - timedelta(days=day.weekday())


def relative_day_table(now: datetime, days: int = 14) -> str:
    """The relative-day table shown to the model so it never does date arithmetic.

    "hôm nay (thứ Sáu) = 2026-11-27" ...

    The table covers two calendar weeks because "thứ ba tuần sau" resolves inside the
    *next* calendar week, not seven days from today (measured on the train split:
    a Thursday with "ba giờ chiều thứ ba tuần sau" resolves to the Tuesday of next
    week, docs/VERIFIED_FACTS.md §12.10). The week a day belongs to is marked so the
    model does not have to derive it.
    """
    today = now.date()
    this_monday = _monday_of(today)
    lines = []
    for offset in range(days + 1):
        moment = today + timedelta(days=offset)
        weekday = _VN_WEEKDAYS[moment.weekday()]
        if offset == 0:
            marker = "hôm nay"
        elif moment >= this_monday + timedelta(days=7):
            marker = "tuần sau"
        else:
            marker = f"+{offset} ngày"
        lines.append(f"- {moment.strftime('%Y-%m-%d')} ({weekday}, {marker})")
    return "\n".join(lines)


def resolve_relative_dates(text: str, now: datetime) -> str:
    """Replace relative day phrases with absolute dates.

    Weekday-anchored phrases ("thứ Sáu tuần này") resolve to the next matching weekday.
    """
    if not text:
        return text
    out = text

    def weekday_match(match: re.Match) -> str:
        phrase = match.group(0).lower()
        target = _WEEKDAYS.get(phrase)
        if target is None:
            return match.group(0)
        days_ahead = (target - now.weekday()) % 7
        return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    out = re.sub(_WEEKDAY_PATTERN, weekday_match, out, flags=re.IGNORECASE)

    for phrase, offset in RELATIVE_DAYS.items():
        pattern = re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", re.IGNORECASE)
        if pattern.search(out):
            moment = now + timedelta(days=offset)
            out = pattern.sub(moment.strftime("%Y-%m-%d"), out)

    return out


# --- spoken clock times ---------------------------------------------------------

# A day part shifts the hour. The conventions are the ones the gold transcripts
# attest, and they differ per day part: "7 giờ tối" -> 19:00, "10 giờ tối" -> 22:00,
# "5 giờ chiều" -> 17:00, "3 giờ chiều" -> 15:00, "6 giờ rưỡi sáng" -> 06:30,
# "mười hai giờ trưa" -> 12:00, "mười một giờ đêm" -> 23:00.
_DAY_PART_PATTERN = re.compile(r"\b(sáng|trưa|chiều|tối|đêm|khuya)\b", re.IGNORECASE)
_HOUR_PATTERN = re.compile(r"(\d{1,2})\s*(?:giờ|tiếng|h)\b", re.IGNORECASE)
_MINUTE_PATTERN = re.compile(r"(\d{1,2})\s*phút\b", re.IGNORECASE)
_HALF_PATTERN = re.compile(r"\brưỡi\b|\bnửa\b(?=\s*giờ)", re.IGNORECASE)
_CLOCK_PATTERN = re.compile(r"(\d{1,2}):(\d{2})")


def _apply_day_part(part: str, hour: int) -> int:
    """The hour the day part names.

    The gold transcripts fix the conventions, and they are not interchangeable:

    "mười một giờ trưa" is 11:00 (train_1058) and "mười hai giờ trưa" is 12:00
    (train_1343), so noon keeps the hour; "mười một giờ đêm" is 23:00 (train_1717),
    so night does not; "ba giờ chiều" is 15:00 (train_0612) and "tám giờ tối" is
    20:00 (train_0620), so afternoon and evening take twelve off a small hour.
    """
    if part == "trưa":
        # Noon. Eleven and twelve keep their hour; an earlier hour named with "trưa"
        # is the afternoon — "hai giờ trưa" is 14:00 — so it takes the twelve it is
        # missing.
        return hour if hour >= 11 else hour + 12
    if part in ("chiều", "tối"):
        return hour + 12 if hour < 12 else hour
    if part == "đêm":
        # Night. "mười một giờ đêm" is 23:00 (train_1717). The small hours keep
        # theirs — "ba giờ đêm" is 03:00 — and twelve is midnight.
        if hour == 12:
            return 0
        return hour + 12 if 7 <= hour <= 11 else hour
    if part == "khuya":
        # The small hours after midnight: the hour as spoken, twelve is midnight.
        return 0 if hour == 12 else hour
    # Morning: the hour as spoken — "chín giờ sáng" is 09:00 (train_0467).
    return hour


def spoken_time_to_clock(text: str) -> str | None:
    """'chín giờ sáng' -> '09:00'. None when the text holds no hour.

    The day part decides the 12-hour offset; :func:`_apply_day_part` holds the
    conventions the gold transcripts attest.

    A day part belongs to the twelve-hour clock, so an hour of thirteen or more
    beside one contradicts itself — "hai mươi giờ tối" is neither reading, because
    twenty is already the twenty-four-hour clock and "tối" only applies to a small
    hour. Such a reading is not resolved: the hour is left to the model rather than
    a wrong time written into a date-time argument that is compared exactly. A
    twenty-four-hour clock on its own is read as it is — "hai mươi giờ" is 20:00.
    """
    if not text:
        return None
    normalized = normalize_numbers(text)
    clock = _CLOCK_PATTERN.search(normalized)
    if clock:
        hour, minute = int(clock.group(1)), int(clock.group(2))
        if hour <= 23 and minute <= 59:
            return f"{hour:02d}:{minute:02d}"
        return None

    hour_match = _HOUR_PATTERN.search(normalized)
    if not hour_match:
        return None
    hour = int(hour_match.group(1))
    if hour > 23:
        return None

    minute = 0
    minute_match = _MINUTE_PATTERN.search(normalized)
    if minute_match:
        minute = int(minute_match.group(1))
    elif _HALF_PATTERN.search(normalized):
        minute = 30

    part_match = _DAY_PART_PATTERN.search(normalized)
    if part_match:
        part = part_match.group(1).lower()
        if hour >= 13:
            # A day part beside a twenty-four-hour hour contradicts itself: "hai mươi
            # giờ tối" is neither 20:00 nor 08:00. Nothing is resolved.
            return None
        hour = _apply_day_part(part, hour)

    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


# --- calendar-aware date resolution ----------------------------------------------

_NUMBER_WORD = (r"mười|một|mốt|hai|ba|bốn|năm|sáu|bảy|tám|chín|không|mươi|\d+")
_DAY_PHRASE = rf"(?:{_NUMBER_WORD})(?:\s+(?:{_NUMBER_WORD})){{0,2}}"
_DAY_OF_MONTH_PATTERN = re.compile(
    rf"(?:hôm|ngày|ngay)\s+({_DAY_PHRASE})(?:\s|$)", re.IGNORECASE)
_WEEKDAY_EXPRESSION = re.compile(
    rf"(?P<weekday>{_WEEKDAY_PATTERN})(?:\s+(?P<week>tuần này|tuần sau|tuan nay|tuan sau))?",
    re.IGNORECASE,
)
_DAY_TODAY_PATTERN = re.compile(r"(?:hôm|ngày|ngay)\s*$", re.IGNORECASE)
_DMY_PATTERN = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{2,4}))?")
_D_MONTH_Y_PATTERN = re.compile(
    r"(\d{1,2})\s*tháng\s*(\d{1,2})(?:\s*năm\s*(\d{2,4}))?", re.IGNORECASE)
# "mười một tháng mười" — the date written in number words.
_D_MONTH_WORDS_PATTERN = re.compile(
    rf"({_DAY_PHRASE})\s*tháng\s+({_NUMBER_WORD})(?:\s+năm\s+({_NUMBER_WORD}))?",
    re.IGNORECASE)

# Self-contained relative day words. Anaphoric ones ("hôm đó", "hôm trước", "ngày đó")
# are deliberately absent: they need the conversation, so they belong to the model.
_RELATIVE_DAY_WORDS = [
    ("ngày hôm nay", 0), ("ngày nay", 0), ("hôm nay", 0), ("nay", 0),
    ("ngày mai", 1), ("mai", 1),
    ("ngày hôm qua", -1), ("hôm qua", -1),
    ("ngày kia", 2), ("hôm kia", -2),
]


def _eval_day_phrase(phrase: str) -> int | None:
    """Evaluate a spoken day-of-month: 'hai tám' -> 28, 'hai mươi tám' -> 28.

    Vietnamese reads a two-digit day digit by digit ("hai tám" is the 28th), while a
    number like "hai mươi tám" is read normally. Both appear in the train split.
    """
    phrase = phrase.strip().lower()
    if re.fullmatch(r"\d{1,2}", phrase):
        return int(phrase)
    tokens = _tokenize_phrase(phrase)
    if not tokens:
        return None
    if len(tokens) == 2 and tokens[0] in UNITS and tokens[1] in UNITS:
        return UNITS[tokens[0]] * 10 + UNITS[tokens[1]]
    try:
        value = _evaluate_phrase(tokens)
    except KeyError:
        return None
    return value


def _year_from(now: datetime, raw: str | None) -> int:
    if not raw:
        return now.year
    value = int(raw)
    if value < 100:
        return 2000 + value
    return value


def _safe_date(year: int, month: int, day: int) -> "date | None":
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _day_in_month(year: int, month: int, day: int) -> "date | None":
    """The given day of the given month, clamped to the month's last day."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return _safe_date(year, month, min(day, last_day))


def resolve_date_expression(text: str, now: datetime) -> str | None:
    """Resolve a self-contained day reference to 'YYYY-MM-DD'. None when ambiguous.

    Handles: 'mai', 'hôm qua', 'hôm nay', 'thứ ba tuần sau', 'hôm hai tám',
    '27/9/2026', '27 tháng 9'. Returns None for references that need the
    conversation ('hôm đó', 'hôm trước') — those are the model's to resolve.
    """
    if not text:
        return None
    lowered = text.lower()

    match = _DMY_PATTERN.search(lowered)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        resolved = _day_in_month(_year_from(now, match.group(3)), month, day)
        return resolved.isoformat() if resolved else None

    match = _D_MONTH_Y_PATTERN.search(lowered)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        resolved = _day_in_month(_year_from(now, match.group(3)), month, day)
        return resolved.isoformat() if resolved else None

    match = _D_MONTH_WORDS_PATTERN.search(lowered)
    if match:
        day = _eval_day_phrase(match.group(1))
        month = _eval_day_phrase(match.group(2))
        year = _eval_day_phrase(match.group(3)) if match.group(3) else None
        if day and month and 1 <= month <= 12 and 1 <= day <= 31:
            resolved = _day_in_month(_year_from(now, str(year) if year else None),
                                     month, day)
            return resolved.isoformat() if resolved else None

    match = _DAY_OF_MONTH_PATTERN.search(lowered)
    if match:
        day = _eval_day_phrase(match.group(1))
        if day and 1 <= day <= 31:
            today = now.date()
            resolved = _day_in_month(today.year, today.month, day)
            if resolved and resolved < today:
                # "hôm 5" said on the 20th means the 5th of next month.
                if today.month == 12:
                    resolved = _day_in_month(today.year + 1, 1, day)
                else:
                    resolved = _day_in_month(today.year, today.month + 1, day)
            return resolved.isoformat() if resolved else None

    match = _WEEKDAY_EXPRESSION.search(lowered)
    if match:
        target = _WEEKDAYS.get(match.group("weekday").lower())
        if target is not None:
            today = now.date()
            week = (match.group("week") or "").lower()
            if week in ("tuần sau", "tuan sau"):
                anchor = _monday_of(today) + timedelta(days=7)
            elif week in ("tuần này", "tuan nay"):
                anchor = _monday_of(today)
            else:
                days_ahead = (target - today.weekday()) % 7
                if days_ahead == 0:
                    # "thứ Sáu" said on a Friday is the next one, unless the day is
                    # named directly ("hôm thứ Sáu" is today).
                    days_ahead = 0 if _DAY_TODAY_PATTERN.search(
                        lowered[: match.start()]) else 7
                return (today + timedelta(days=days_ahead)).isoformat()
            return (anchor + timedelta(days=target)).isoformat()

    for phrase, offset in _RELATIVE_DAY_WORDS:
        pattern = re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", re.IGNORECASE)
        if pattern.search(text):
            return (now.date() + timedelta(days=offset)).isoformat()

    return None


# --- typed value normalisation ----------------------------------------------------

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{2})(?::(\d{2}))?"
    r"(Z|[+-]\d{2}:?\d{2})?$")
_ISO_TIME = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def normalize_number_value(value, declared_type: str):
    """Convert a spoken numeric argument into a typed number. None when unchanged.

    'bốn mươi phần trăm' -> 40, 'hai triệu đồng' -> 2000000, '2.000.000' -> 2000000.
    Only numeric-typed arguments are normalised: free text keeps the user's wording.
    """
    if isinstance(value, bool):
        return None
    if declared_type == "integer":
        if isinstance(value, int):
            return None
        if isinstance(value, float):
            return int(value) if value.is_integer() else None
        if not isinstance(value, str):
            return None
        text = normalize_numbers(value)
        text = re.sub(r"(?i)phần\s*trăm|phan\s*tram|%", " ", text)
        text = text.replace(" ", "")
        if not re.fullmatch(r"-?\d+", text):
            # A decimal stays a decimal: "một nửa" is 0.5, never 5.
            return None
        return int(text)
    if declared_type == "number":
        if isinstance(value, (int, float)):
            return None
        if not isinstance(value, str):
            return None
        text = normalize_numbers(value)
        text = re.sub(r"(?i)phần\s*trăm|phan\s*tram|%", " ", text)
        text = text.replace(",", ".")
        text = text.replace(".", "", 1) if text.count(".") > 1 else text
        text = text.replace(" ", "")
        if not re.fullmatch(r"-?\d+(\.\d+)?", text):
            return None
        number = float(text)
        return int(number) if number.is_integer() and declared_type == "integer" else number
    return None


def normalize_date_value(value, fmt: str, now: datetime, tz_offset: str = "+07:00"):
    """Repair a date/time argument into its declared format. None when unchanged.

    * already valid -> untouched
    * ISO without the timezone offset -> the offset is added (it changes the state hash)
    * spoken forms ('chín giờ sáng mai') -> resolved against the sample's clock
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()

    if fmt == "date":
        if _ISO_DATE.match(text):
            return None
        return resolve_date_expression(text, now)

    if fmt == "time":
        if _ISO_TIME.match(text):
            return None
        return spoken_time_to_clock(text)

    if fmt == "date-time":
        match = _ISO_DATETIME.match(text)
        if match:
            date_part, hour, minute, second, offset = match.groups()
            if offset and offset not in ("Z",):
                return None  # already carries an offset
            offset_text = tz_offset
            return (f"{date_part}T{int(hour):02d}:{minute}:{second or '00'}"
                    f"{offset_text}")
        resolved_date = resolve_date_expression(text, now)
        clock = spoken_time_to_clock(text)
        if resolved_date and clock:
            return f"{resolved_date}T{clock}:00{tz_offset}"
        return None

    return None


def recase(text: str) -> str:
    """Recase an uppercase ASR transcript to sentence case (Decision 6).

    gipformer emits uppercase text with no punctuation. A non-fuzzy string argument is
    compared with exact equality, so "BẬT ĐIỀU HÒA" would not match a gold value of
    "Bật điều hòa" — one wrong argument costs the whole sample.

    A full stop between digits is a thousands separator, not a sentence end, so
    "200.000 ĐỒNG" must not become "200.000 Đồng".
    """
    if not text:
        return text
    lowered = text.lower()

    # Mark the sentence starts: a full stop that is NOT between digits.
    out = []
    capitalise_next = True
    index = 0
    length = len(lowered)
    while index < length:
        char = lowered[index]
        if char == ".":
            before = lowered[index - 1] if index > 0 else ""
            after = lowered[index + 1] if index + 1 < length else ""
            thousands = before.isdigit() and after.isdigit()
            out.append(char)
            if not thousands:
                capitalise_next = True
            index += 1
            continue
        if char in ".!?\n":
            out.append(char)
            capitalise_next = True
            index += 1
            continue
        if char.isspace():
            out.append(char)
            index += 1
            continue
        if capitalise_next:
            out.append(char.upper())
            capitalise_next = False
        else:
            out.append(char)
        index += 1
    return "".join(out)


def nfc(text: str) -> str:
    """NFC normalisation. Diacritics are kept — the matcher is diacritic-sensitive."""
    return unicodedata.normalize("NFC", text or "")


def normalize_text(text: str, now: datetime | None = None) -> str:
    """The full deterministic pass: NFC -> numbers -> relative dates."""
    out = nfc(text)
    if now is not None:
        out = resolve_relative_dates(out, now)
    out = normalize_numbers(out)
    return nfc(out)
