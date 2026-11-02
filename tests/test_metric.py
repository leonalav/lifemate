"""Verify the packaged metric against the research reference implementation.

The research script ``research/scorer.py`` is the instrument every PRD number is quoted
from. This test proves the packaged version (which the CLI uses) agrees with it.

Run:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import discover_splits, resolve_split  # noqa: E402
from lifemate.metric import (  # noqa: E402
    calls_match,
    fuzzy_arguments,
    gold_hash,
    max_bipartite_match,
    replay_hash,
    sample_score,
    score_split,
    token_f1,
)

DATA_DIR = REPO_ROOT / "data" / "train+public_test"


@pytest.fixture(scope="module")
def train_split():
    if not (DATA_DIR / "train").is_dir():
        pytest.skip("the competition kit is not present")
    return resolve_split(DATA_DIR, "train")


def test_token_f1_matches_the_documented_boundaries():
    assert token_f1("Mai Anh", "mai anh") == pytest.approx(1.0)
    assert token_f1("Bật điều hòa", "bật điều hoà") > 0.6
    assert token_f1("Hương đi học", "huong di hoc") == 0.0
    assert token_f1("abc def", "abc xyz") == 0.5


def test_fuzzy_set_is_per_tool_not_pooled(train_split):
    fuzzy = fuzzy_arguments(train_split.catalog())
    assert fuzzy, "the train catalog declares fuzzy arguments"
    for tool, keys in fuzzy.items():
        assert tool.count(".") == 1
        assert keys


def test_exact_key_set_is_required(train_split):
    fuzzy = fuzzy_arguments(train_split.catalog())
    gold = {"name": "smart_home.light_set", "arguments": {"device_id": "d1"}}
    assert calls_match(gold, gold, fuzzy)
    # an extra optional argument breaks the match
    assert not calls_match(
        {"name": "smart_home.light_set", "arguments": {"device_id": "d1", "power": "on"}},
        gold, fuzzy)
    # a missing argument breaks the match
    assert not calls_match({"name": "smart_home.light_set", "arguments": {}}, gold, fuzzy)
    # a different tool breaks the match
    assert not calls_match({"name": "smart_home.get_status", "arguments": {"device_id": "d1"}},
                           gold, fuzzy)


def test_type_sensitive_equality(train_split):
    fuzzy = fuzzy_arguments(train_split.catalog())
    gold = {"name": "smart_home.light_set", "arguments": {"device_id": "d1"}}
    assert not calls_match(
        {"name": "smart_home.light_set", "arguments": {"device_id": 1}}, gold, fuzzy)


def test_one_call_matches_at_most_one_gold_call(train_split):
    fuzzy = fuzzy_arguments(train_split.catalog())
    gold = {"name": "smart_home.light_set", "arguments": {"device_id": "d1"}}
    preds = [gold, gold]
    assert max_bipartite_match(preds, [gold], fuzzy) == 1


def test_empty_gold_scores_one_only_for_an_empty_submission():
    fuzzy = {}
    assert sample_score([], [], "a", "b", fuzzy) == 1.0
    assert sample_score([{"name": "x.y", "arguments": {}}], [], "a", "b", fuzzy) == 0.0


def test_state_gate_kills_a_mismatching_hash():
    fuzzy = {}
    gold = [{"name": "smart_home.light_set", "arguments": {"device_id": "d1"}}]
    pred = list(gold)
    assert sample_score(pred, gold, "same", "same", fuzzy) == 1.0
    assert sample_score(pred, gold, "different", "same", fuzzy) == 0.0


@pytest.mark.slow
def test_gold_replay_reproduces_the_gold_state_hash(train_split):
    """The gold hash must equal the hash after replaying the gold trajectory."""
    samples = list(train_split.iter_samples())[:25]
    for sample in samples:
        assert gold_hash(train_split, sample) == replay_hash(
            train_split, sample["id"], sample["label"]["tool_calls"])


@pytest.mark.slow
def test_gold_replay_scores_one_and_an_empty_submission_scores_the_out_of_scope_rate(train_split):
    """The two numbers the PRD quotes: gold replay = 1.000000, empty = 7.18/100."""
    samples = list(train_split.iter_samples())
    gold_submissions = {
        s["id"]: s["label"]["tool_calls"] for s in samples
    }
    result = score_split(train_split, samples, gold_submissions)
    assert result["mean"] == pytest.approx(1.0, abs=1e-9)
    assert result["perfect"] == len(samples)

    empty = score_split(train_split, samples, {})
    assert empty["score"] == pytest.approx(7.18, abs=0.05)


def test_discovers_both_kit_layouts(tmp_path):
    """The kit arrives as a bundle folder or with split folders at the top level."""
    for layout in ("bundle", "flat"):
        root = tmp_path / layout
        split_dir = root / "train" if layout == "flat" else root / "train+public_test" / "train"
        split_dir.mkdir(parents=True)
        (split_dir / "train.jsonl").write_text("", encoding="utf-8")
        (split_dir / "train_tools_catalog.json").write_text("{}", encoding="utf-8")
        found = discover_splits(root)
        assert "train" in found
        assert found["train"].path == split_dir
