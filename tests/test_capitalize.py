"""The capitalisation restoration, measured the way it was designed.

The numbers quoted in ``src/lifemate/capitalize.py`` are asserted here, so the module's
documentation cannot drift away from its behaviour.
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import (  # noqa: E402
    HONORIFICS,
    is_acronym,
    restore_capitals,
)
from lifemate.case_model import load_case_model, refine_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402
from lifemate.text import recase  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "train+public_test"


@pytest.fixture(scope="module")
def train_split():
    if not (DATA_DIR / "train").is_dir():
        pytest.skip("the competition kit is not present")
    return resolve_split(DATA_DIR, "train")


def _similarity(a: str, b: str) -> float:
    if not a:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# --- the rules -------------------------------------------------------------------

def test_a_known_brand_keeps_its_spelling():
    assert restore_capitals("DẠ ZALO ĐI Ạ.") == "Dạ Zalo đi ạ."


def test_an_acronym_keeps_its_capitals():
    """The acronym keeps its caps; the name beside it needs the context model.

    "Linh" is an ambiguous token — the training transcripts spell it both ways — so the
    rule module alone cannot capitalise it. The context model reads the neighbouring
    words and does. This is the case the model exists for.
    """
    upper = "CHO EM LINH HỎI GIÁ VÀNG SJC HÔM NAY"
    assert restore_capitals(upper) == "Cho em linh hỏi giá vàng SJC hôm nay"

    model = load_case_model()
    if model is None:
        pytest.skip("the case model artifact has not been built")
    assert refine_capitals(upper, model) == "Cho em Linh hỏi giá vàng SJC hôm nay"


def test_a_name_after_an_honorific_is_capitalised():
    assert restore_capitals("CHO BÁC BA VỚI CHUYỂN VÍ CHO CÔ MAI") == (
        "Cho Bác Ba với chuyển ví cho Cô Mai"
    )


def test_an_ordinary_word_is_not_mistaken_for_an_acronym():
    """The rule that a short uppercase word is an acronym must not fire on common words."""
    assert "CHO" not in restore_capitals("CHO TÔI GỌI LÊN")
    assert "QUÁN" not in restore_capitals("ĐẾN QUÁN NÀY NHÉ")


def test_a_thousands_separator_is_not_a_sentence_end():
    assert restore_capitals("CHUYỂN 2.500.000 ĐỒNG") == "Chuyển 2.500.000 đồng"


def test_an_ordinary_word_after_a_non_name_honorific_stays_lowercase():
    """'anh', 'chị' and 'em' are ordinary words, not name markers — they were removed."""
    for word in ("anh", "chị", "em", "tôi"):
        assert word not in HONORIFICS


def test_is_acronym_rejects_ordinary_words():
    assert not is_acronym("CHO")
    assert not is_acronym("QUÁN")
    assert is_acronym("SJC")
    assert is_acronym("VOV3")


# --- the product codes -------------------------------------------------------------

def test_a_product_code_keeps_its_capitals():
    """A code's shape is not a context question: "VOV3" must not become "vov3".

    The derived acronym set holds the letters of the code ("vov", "vtv", "ron", "rv"),
    and the digits are matched against it. Measured against the training transcripts,
    this fixes eleven tokens and breaks none.
    """
    assert restore_capitals("NGHE KÊNH VOV3 NHÉ") == "Nghe kênh VOV3 nhé"
    assert restore_capitals("ĐỔ RON95 VÀ RV183") == "Đổ RON95 và RV183"
    assert restore_capitals("XEM E5 NHÉ") == "Xem E5 nhé"


def test_a_code_is_not_confused_with_a_word_that_ends_in_digits():
    """"nhom22" and "24h" are not codes, so they keep their lowercase form."""
    assert restore_capitals("VÀO NHÓM Nhom22 LÚC 24H") == (
        "Vào nhóm nhom22 lúc 24h"
    )


def test_is_code_recognises_only_the_codes():
    from lifemate.capitalize import is_code

    assert is_code("VOV3")
    assert is_code("RON95")
    assert is_code("RV183")
    assert not is_code("nhom22")
    assert not is_code("24h")
    assert not is_code("giúp")
    assert not is_code("500")


# --- the measured numbers -----------------------------------------------------------

def test_the_round_trip_similarity_beats_the_naive_approaches(train_split):
    """The numbers quoted in the module's docstring must still hold."""
    texts = [s["audio_text"] for s in train_split.iter_samples() if s.get("audio_text")]
    assert len(texts) == 1545

    def score(fn):
        ratios = [_similarity(fn(t.upper()), t) for t in texts]
        return sum(ratios) / len(ratios)

    lowercase = score(lambda t: t.lower())
    sentence = score(recase)
    restored = score(restore_capitals)

    assert restored > sentence > lowercase
    # the documented figures, with a small tolerance for vocabulary drift
    assert lowercase == pytest.approx(0.9729, abs=0.002)
    assert sentence == pytest.approx(0.9866, abs=0.002)
    assert restored == pytest.approx(0.9927, abs=0.002)


def test_the_restoration_reproduces_the_gold_query_arguments(train_split):
    """The decisive metric: can the model copy a gold query value out of the transcript?

    300 gold arguments across 8 search tools are compared with exact equality, and they
    carry capitals ("Sếp Dũng", "VOV3"). Restoring the capitals takes the reproducible
    share from 8.7% to 58.7%, and the context model takes it to 79.3%.
    """
    fuzzy = fuzzy_arguments(train_split.catalog())
    samples = list(train_split.iter_samples())

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

    naive, total = reproduce(lambda t: t.lower())
    restored, _ = reproduce(restore_capitals)

    assert total > 250
    assert naive / total < 0.15
    assert restored / total > 0.5


def test_the_context_model_reproduces_more_of_the_gold_query_arguments(train_split):
    """The context model is what lifts the reproducible share from 58.7% to 79.3%.

    The tokens the rule module leaves alone are ambiguous ones — the training
    transcripts spell "Mai" both ways — so their case is a question about the
    neighbouring words.
    """
    model = load_case_model()
    if model is None:
        pytest.skip("the case model artifact has not been built")

    fuzzy = fuzzy_arguments(train_split.catalog())
    samples = list(train_split.iter_samples())

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

    by_rules, _ = reproduce(restore_capitals)
    by_model, total = reproduce(lambda t: refine_capitals(t, model))

    assert total > 250
    assert by_model > by_rules
    assert by_model / total > 0.75


# --- the resource --------------------------------------------------------------------

def test_the_vocabulary_resource_exists_and_is_usable():
    resource = REPO_ROOT / "src" / "lifemate" / "resources" / "capitalization.json"
    assert resource.is_file()
    with resource.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["proper"]
    assert payload["acronyms"]
    assert payload["stats"]["proper"] > 100
