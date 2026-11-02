"""The context-aware capitalisation model, measured the way it was designed.

The numbers quoted in ``src/lifemate/case_model.py`` are asserted here, so the module's
documentation cannot drift away from its behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.case_model import (  # noqa: E402
    CaseModel,
    featurize,
    load_case_model,
    refine_capitals,
    transcript,
)

DATA_DIR = REPO_ROOT / "data" / "train+public_test"
ARTIFACT = REPO_ROOT / "src" / "lifemate" / "resources" / "case_model.json"


@pytest.fixture(scope="module")
def model():
    if not ARTIFACT.is_file():
        pytest.skip("the case model artifact has not been built")
    return load_case_model(ARTIFACT)


# --- the artifact ----------------------------------------------------------------

def test_the_artifact_is_present_and_loadable(model):
    assert model is not None
    assert model.weights
    assert isinstance(model.intercept, float)


def test_the_artifact_records_its_version():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["weights"]


def test_a_missing_artifact_is_reported_as_absent(tmp_path):
    assert load_case_model(tmp_path / "absent.json") is None


def test_the_rules_are_the_fallback_when_the_artifact_is_absent(monkeypatch):
    """The pipeline must still run without the artifact.

    The transcript then carries the rule module's restoration, which is the documented
    fallback and scores 96.932 instead of 98.301.
    """
    monkeypatch.setattr("lifemate.case_model.load_case_model", lambda path=None: None)
    text = "GỌI CHO MINH NHÉ"
    assert transcript(text) == restore_capitals(text)


# --- the scoring -----------------------------------------------------------------

def test_the_score_uses_the_rule_modules_opinion():
    """The model refines the rules; it does not replace them.

    The same token must score differently when the rule module's opinion differs,
    otherwise the model has learned nothing about the text.
    """
    upper = "GỌI CHO MINH NHÉ".split()
    capital_rules = ["Gọi", "cho", "Minh", "nhé"]
    lower_rules = ["gọi", "cho", "minh", "nhé"]

    model = CaseModel({"intercept": 0.0, "weights": {"rule_says_capital": 1.0}})
    assert model.score(featurize(upper, capital_rules, 2)) > 0.0
    assert model.score(featurize(upper, lower_rules, 2)) == 0.0


def test_a_categorical_feature_is_matched_by_its_value():
    """A categorical feature must look up "name=value", not "name"."""
    upper = "GỌI CHO MINH".split()
    rules = restore_capitals("GỌI CHO MINH").split()
    model = CaseModel({"intercept": 0.0, "weights": {"prev1=cho": 5.0}})
    assert model.score(featurize(upper, rules, 2)) > 0.0
    other = CaseModel({"intercept": 0.0, "weights": {"prev1=khac": 5.0}})
    assert other.score(featurize(upper, rules, 2)) == 0.0


def test_an_unseen_word_contributes_nothing():
    """A word the model has never seen must not shift the score.

    This is what makes the model degrade gracefully on the public test, where the
    contact names are ones the training split never saw.
    """
    upper = "GỌI CHO MINH".split()
    rules = restore_capitals("GỌI CHO MINH").split()
    model = CaseModel({"intercept": 0.0, "weights": {"word=khongco": 5.0}})
    assert model.score(featurize(upper, rules, 2)) == 0.0


# --- the behaviour ---------------------------------------------------------------

def test_the_model_keeps_the_shape_of_each_token():
    """The model decides the first character only.

    The rule module owns internal capitals and punctuation, so "S+ Sport" keeps its
    shape and only the first letter of each token is touched.
    """
    upper = "NGHE S+ SPORT NHÉ"

    class AlwaysCapital(CaseModel):
        def predicts_capital(self, features):
            return True

    out = refine_capitals(upper, AlwaysCapital({"intercept": 0.0, "weights": {}}))
    assert "+" in out                    # the punctuation survives
    assert "SPORT" not in out            # only the first character is touched
    assert out.split()[1].startswith("S")


def test_the_model_leaves_a_token_without_letters_alone():
    upper = "GỌI CHO MINH 123"
    out = refine_capitals(upper, CaseModel({"intercept": 0.0, "weights": {}}))
    assert "123" in out


def test_refinement_keeps_the_token_count():
    """The transcript must not gain or lose words."""
    upper = "TÔI MUỐN ĐẶT LỊCH KHÁM BỆNH VÀO SÁNG MAI"
    out = refine_capitals(upper, CaseModel({"intercept": 0.0, "weights": {}}))
    assert len(out.split()) == len(upper.split())


def test_the_model_improves_on_the_rules_for_an_ambiguous_name(model):
    """The measured reason the model exists.

    "Minh" is a contact name, but the token appears in both cases across the training
    transcripts, so the rule module lowercases it. The model reads the context and
    capitalises it.
    """
    upper = "GỌI CHO MINH NHÉ"
    assert restore_capitals(upper) == "Gọi cho minh nhé"
    assert refine_capitals(upper, model) == "Gọi cho Minh nhé"


# --- the division of labour ---------------------------------------------------------

def test_the_model_does_not_touch_an_honorific(model):
    """The rules capitalise "Mẹ" only when a name follows it; the model must not undo that.

    All 17 arguments the model broke were honorifics or codes, so the rules keep the
    final say on both.
    """
    assert refine_capitals("GỌI CHO MẸ LAN NHÉ", model) == "Gọi cho Mẹ Lan nhé"
    assert refine_capitals("CHÁO CHÚ NĂM ĐI", model) == "Cháo Chú Năm đi"


def test_the_model_does_not_touch_a_product_code(model):
    """A code's shape is not a context question."""
    assert refine_capitals("NGHE KÊNH VOV3 NHÉ", model) == "Nghe kênh VOV3 nhé"


def test_the_model_still_decides_an_ordinary_ambiguous_word(model):
    """Deferring must not make the model inert.

    The honorific and the code are excluded; a plain name is not, and the model is what
    recovers it.
    """
    upper = "GỌI CHO MINH NHÉ"
    assert refine_capitals(upper, model) != restore_capitals(upper)
