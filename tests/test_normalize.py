"""The deterministic resolvers: numbers, clock times, dates.

Every expectation here was read off the gold trajectories of the train split, so the
tests encode the organisers' conventions rather than ours.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.text import (  # noqa: E402
    normalize_date_value,
    normalize_number_value,
    resolve_date_expression,
    spoken_time_to_clock,
)

# A Friday, 09:00, +07:00 — the anchor of train_0471.
NOW = datetime.fromisoformat("2026-09-25T09:00:00+07:00")
# A Thursday, 09:00, +07:00 — the anchor of train_0633.
THURSDAY = datetime.fromisoformat("2026-10-01T09:00:00+07:00")


# --- clock times -----------------------------------------------------------------

@pytest.mark.parametrize("spoken,expected", [
    ("chín giờ sáng", "09:00"),      # train_0467: due 2026-09-26T09:00
    ("năm giờ chiều", "17:00"),     # train_0471: due 2026-09-27T17:00
    ("bảy giờ tối", "19:00"),       # train_0476: rule 19:00
    ("mười giờ tối", "22:00"),      # train_0484: rule 22:00
    ("chín giờ tối", "21:00"),      # train_0510: rule 21:00
    ("tám giờ tối", "20:00"),       # train_0620: rule 20:00
    ("ba giờ chiều", "15:00"),      # train_0612: 2026-10-03T15:00
    ("sáu giờ rưỡi sáng", "06:30"),  # train_0578: 2026-09-25T06:30
    ("mười giờ sáng", "10:00"),     # train_0536: 2026-09-26T10:00
    ("9 giờ", "09:00"),
    ("19:00", "19:00"),
])
def test_spoken_clock_times(spoken, expected):
    assert spoken_time_to_clock(spoken) == expected


def test_an_impossible_hour_is_not_resolved():
    assert spoken_time_to_clock("hai mươi giờ tối") is None
    assert spoken_time_to_clock("không có giờ nào") is None


def test_a_24_hour_clock_is_read_as_it_is():
    """'hai mươi giờ' is 20:00, not an error."""
    assert spoken_time_to_clock("hai mươi giờ") == "20:00"


# --- dates -----------------------------------------------------------------------

def test_relative_days():
    assert resolve_date_expression("mai", NOW) == "2026-09-26"
    assert resolve_date_expression("ngày mai", NOW) == "2026-09-26"
    assert resolve_date_expression("hôm nay", NOW) == "2026-09-25"
    assert resolve_date_expression("hôm qua", NOW) == "2026-09-24"
    assert resolve_date_expression("ngày kia", NOW) == "2026-09-27"


def test_weekdays_resolve_to_the_next_occurrence():
    """'thứ ba' said on a Friday is the coming Tuesday (train_0472)."""
    assert resolve_date_expression("thứ ba", NOW) == "2026-09-29"
    assert resolve_date_expression("thứ bảy", THURSDAY) == "2026-10-03"
    assert resolve_date_expression("thứ tư", THURSDAY) == "2026-10-07"


def test_a_weekday_spoken_on_that_same_weekday_is_the_next_one():
    """train_1435: 'Sáu' said on a Friday is the Friday of next week."""
    friday = datetime.fromisoformat("2026-12-04T09:00:00+07:00")
    assert resolve_date_expression("thứ sáu", friday) == "2026-12-11"
    assert resolve_date_expression("hôm thứ sáu", friday) == "2026-12-04"


def test_next_week_is_the_next_calendar_week():
    """train_0633: a Thursday, 'thứ ba tuần sau' is the Tuesday of next week."""
    assert resolve_date_expression("thứ ba tuần sau", THURSDAY) == "2026-10-06"
    assert resolve_date_expression("thứ hai tuần sau", NOW) == "2026-09-28"


def test_this_week_is_the_current_calendar_week():
    """train_0921: 'tuần này' on a Friday covers Monday..Friday of that week."""
    friday = datetime.fromisoformat("2026-10-23T09:00:00+07:00")
    assert resolve_date_expression("thứ hai tuần này", friday) == "2026-10-19"


def test_day_of_month_spoken_in_words():
    """train_0471: 'hôm hai tám' is the 28th, read digit by digit."""
    assert resolve_date_expression("hôm hai tám", NOW) == "2026-09-28"
    assert resolve_date_expression("hôm hai mươi tám", NOW) == "2026-09-28"
    assert resolve_date_expression("ngày mười bốn", THURSDAY) == "2026-10-14"


def test_a_day_that_has_passed_rolls_to_the_next_month():
    """'ngày 5' said on the 25th is the 5th of next month."""
    assert resolve_date_expression("ngày năm", NOW) == "2026-10-05"


def test_dates_spelled_in_words():
    """train_0701: 'mười một tháng mười' is the 11th of October."""
    assert resolve_date_expression("mười một tháng mười", THURSDAY) == "2026-10-11"
    assert resolve_date_expression("27/9", NOW) == "2026-09-27"
    assert resolve_date_expression("27 tháng 9", NOW) == "2026-09-27"


def test_anaphoric_references_are_left_to_the_model():
    """'hôm đó' and 'hôm trước' need the conversation, so nothing is resolved."""
    assert resolve_date_expression("hôm đó", NOW) is None
    assert resolve_date_expression("hôm trước", NOW) is None
    assert resolve_date_expression("ngày đó", NOW) is None


# --- typed values -----------------------------------------------------------------

@pytest.mark.parametrize("spoken,expected", [
    ("bốn mươi phần trăm", 40),     # train_0620: brightness 40
    ("30 phần trăm", 30),           # train_0083: brightness 30
    ("50%", 50),                    # train_0075: brightness 50
    ("hai mươi sáu", 26),           # train_0075: temperature 26
    ("2.000.000", 2000000),
    ("hai triệu", 2000000),
    ("2000000 đồng", 2000000),
])
def test_spoken_numbers_become_typed_integers(spoken, expected):
    assert normalize_number_value(spoken, "integer") == expected


def test_a_non_integer_stays_a_non_integer():
    """'một nửa' is 0.5 — it must not become the integer 5."""
    assert normalize_number_value("một nửa", "integer") is None
    assert normalize_number_value("một nửa", "number") == 0.5


def test_a_correct_value_is_never_rewritten():
    """A typed number is already correct; only a string is converted."""
    assert normalize_number_value(26, "integer") is None
    assert normalize_number_value("26", "integer") == 26


def test_dates_keep_their_timezone_offset():
    """Dropping the offset changes the state hash, so the sample scores 0."""
    assert normalize_date_value("2026-09-26T09:00:00+07:00", "date-time",
                                NOW, "+07:00") is None
    assert normalize_date_value("2026-09-26T09:00:00", "date-time",
                                NOW, "+07:00") == "2026-09-26T09:00:00+07:00"
    assert normalize_date_value("2026-09-26 09:00", "date-time",
                                NOW, "+07:00") == "2026-09-26T09:00:00+07:00"


def test_a_spoken_date_time_is_resolved_against_the_sample_clock():
    assert normalize_date_value("chín giờ sáng mai", "date-time",
                                NOW, "+07:00") == "2026-09-26T09:00:00+07:00"
    assert normalize_date_value("năm giờ chiều hôm trước", "date-time",
                                NOW, "+07:00") is None  # anaphoric: not resolvable


def test_a_correct_date_is_never_rewritten():
    assert normalize_date_value("2026-09-27", "date", NOW, "+07:00") is None
    assert normalize_date_value("19:00", "time", NOW, "+07:00") is None
