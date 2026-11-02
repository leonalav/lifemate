"""The casing policy, the truecaser hint and the two safety nets.

The pipeline no longer restores the capitalisation the recogniser threw away: the
transcript reaches the model in lowercase and the model casts the arguments itself
(docs/DECISIONS.md, Decision 24). These tests pin the behaviour that decision buys:
one policy switch, one place where it is applied, and two safety nets that recover a
lost capital without touching the model's wording.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.agent.guard import Guard, collect_strings  # noqa: E402
from lifemate.agent.prompt import build_prompt  # noqa: E402
from lifemate.casing import apply_case, get_mode, is_recasing, set_mode  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.text import normalize_number_value  # noqa: E402
from lifemate.truecaser import _forms  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "train+public_test"


@pytest.fixture(scope="module")
def train_split():
    if not (DATA_DIR / "train").is_dir():
        pytest.skip("the competition kit is not present")
    return resolve_split(DATA_DIR, "train")


@pytest.fixture(autouse=True)
def _default_mode():
    """Every test starts from the shipped policy."""
    set_mode(None)
    yield
    set_mode(None)


# --- the policy -----------------------------------------------------------------

def test_the_default_policy_lowercases():
    assert get_mode() == "none"
    assert apply_case("Nhắn Sếp Dũng") == "nhắn sếp dũng"


def test_the_restoring_modes_are_available_for_the_ablation():
    assert "Sếp" in apply_case("NHẮN SẾP DŨNG", "rules")
    assert "Dũng" in apply_case("NHẮN SẾP DŨNG", "model")


def test_every_mode_is_idempotent():
    """The transcript can reach the policy twice (the request and a history turn)."""
    text = "nhắn sếp dũng giờ mới"
    for mode in ("none", "rules", "model"):
        once = apply_case(text, mode)
        assert apply_case(once, mode) == once


def test_the_mode_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("LIFEMATE_CASE_MODE", "rules")
    set_mode(None)
    assert get_mode() == "rules"
    assert is_recasing()
    set_mode("none")
    assert not is_recasing()


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        set_mode("magic")


# --- the prompt -----------------------------------------------------------------

def test_the_prompt_shows_the_lowercase_transcript(train_split):
    sample = next(train_split.iter_samples())
    prompt = build_prompt(train_split, sample, sample["audio_text"])
    assert sample["audio_text"].lower() in prompt
    # the cased original must not leak in
    assert sample["audio_text"] not in prompt


def test_the_prompt_carries_the_truecaser_hint_when_there_is_one(train_split):
    sample = next(train_split.iter_samples())

    def truecaser(text: str) -> str:
        return "Nhắn Sếp Dũng"

    prompt = build_prompt(train_split, sample, sample["audio_text"], case_fn=truecaser)
    assert "## Gợi ý viết hoa" in prompt
    assert "Nhắn Sếp Dũng" in prompt


def test_a_broken_truecaser_does_not_break_the_prompt(train_split):
    sample = next(train_split.iter_samples())

    def broken(text: str) -> str:
        raise RuntimeError("no weights")

    prompt = build_prompt(train_split, sample, sample["audio_text"], case_fn=broken)
    assert "## Gợi ý viết hoa" not in prompt


# --- the truecaser --------------------------------------------------------------

def test_the_candidates_of_a_word_cover_the_usual_spellings():
    assert _forms("mai anh") == ["mai anh", "Mai anh", "MAI ANH"]
    # a word that already carries capitals keeps its own spelling
    assert "VOV3" in _forms("VOV3")
    # a word without letters is left alone
    assert _forms("2026") == ["2026"]


# --- the safety nets ------------------------------------------------------------

def test_a_lowercase_argument_is_snapped_to_the_observed_spelling(train_split):
    guard = Guard(train_split)
    guard.observed_strings = collect_strings(
        {"name": "Mai Anh", "id": "c_08"}, {}
    )
    arguments, changed = guard.snap_to_observed("contacts.search", {"query": "mai anh"})
    assert arguments["query"] == "Mai Anh"
    assert changed


def test_an_argument_that_already_has_capitals_is_left_alone(train_split):
    guard = Guard(train_split)
    guard.observed_strings = collect_strings({"name": "Mai Anh"}, {})
    arguments, changed = guard.snap_to_observed("contacts.search", {"query": "Mai Anh"})
    assert arguments["query"] == "Mai Anh"
    assert not changed


def test_a_value_that_is_not_an_observed_entity_is_left_alone(train_split):
    guard = Guard(train_split)
    guard.observed_strings = collect_strings({"name": "Mai Anh"}, {})
    arguments, changed = guard.snap_to_observed(
        "sms.send", {"content": "dời sang 9 giờ mai"}
    )
    assert arguments["content"] == "dời sang 9 giờ mai"
    assert not changed


def test_the_observed_spelling_with_capitals_wins_over_the_plain_one():
    seen = collect_strings({"first": "mai anh", "second": "Mai Anh"}, {})
    assert seen["mai anh"] == "Mai Anh"


def test_the_number_pass_survives_a_capitalised_number_word():
    """A model that writes "Ba mươi" means thirty; the guard must still repair it."""
    assert normalize_number_value("Ba mươi", "integer") == 30
    assert normalize_number_value("bốn mươi phần trăm", "integer") == 40


# --- the training set ------------------------------------------------------------

def test_the_dataset_shows_a_lowercase_transcript_and_a_cased_target(train_split):
    """The whole point of Decision 24, in one test.

    The prompt carries the recogniser's text in lowercase; the target is the gold
    action, whose arguments keep the user's own capitalisation. The model therefore
    has to learn that "sếp dũng" is "Sếp Dũng".
    """
    from lifemate.training_data import build_examples

    samples = list(train_split.iter_samples())[:2]
    # a stand-in for the recogniser's output: the uppercase transcript
    cache = {s["audio"]: (s.get("audio_text") or "").upper() for s in samples}

    set_mode("none")
    examples = build_examples(train_split, samples, transcripts=cache, case_mode="none")
    assert examples

    for example in examples:
        request = example.prompt.split("## Yêu cầu hiện tại của người dùng")[1]
        request = request.split("##")[0]
        assert request == request.lower()

    # the target keeps the gold capitalisation
    cased = [e for e in examples if "Sếp" in e.completion]
    if cased:
        assert "sếp dũng" not in cased[0].completion


def test_the_recasing_baseline_still_recases_the_prompt(train_split):
    """The ablation baseline: the context model recases the transcript again."""
    from lifemate.training_data import build_examples

    samples = list(train_split.iter_samples())[:2]
    cache = {s["audio"]: (s.get("audio_text") or "").upper() for s in samples}

    set_mode("model")
    examples = build_examples(train_split, samples, transcripts=cache, case_mode="model")
    request = examples[0].prompt.split("## Yêu cầu hiện tại của người dùng")[1]
    request = request.split("##")[0]
    assert request != request.lower()
