"""End-to-end pipeline test on CPU, without any model.

The pipeline (prompt -> policy -> guard -> environment -> submission) is exercised with a
scripted policy, so the whole thing is verifiable locally. Nothing here trains or infers a
model — that only happens on Colab.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.agent.guard import Guard, collect_ids  # noqa: E402
from lifemate.agent.loop import Agent  # noqa: E402
from lifemate.agent.policy import Action, ScriptedPolicy, parse_action  # noqa: E402
from lifemate.agent.prompt import build_action_schema, build_prompt  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.kit import tool_env  # noqa: E402
from lifemate.metric import score_split  # noqa: E402
from lifemate.text import normalize_numbers, relative_day_table, resolve_relative_dates  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "train+public_test"


@pytest.fixture(scope="module")
def train_split():
    if not (DATA_DIR / "train").is_dir():
        pytest.skip("the competition kit is not present")
    return resolve_split(DATA_DIR, "train")


# --- the normaliser -------------------------------------------------------------

def test_vietnamese_number_words_become_typed_numbers():
    assert normalize_numbers("hai triệu tám") == "2000008"
    assert normalize_numbers("một trăm hai mươi ba") == "123"
    assert normalize_numbers("2.000.000 đồng") == "2000000"
    assert normalize_numbers("một nửa") == "0.5"


def test_percentage_is_normalised():
    assert normalize_numbers("30 phần trăm") == "30%"
    assert normalize_numbers("30%") == "30%"


def test_normalisation_keeps_diacritics():
    """The matcher is diacritic-sensitive; folding would silently break matches."""
    from lifemate.text import nfc

    assert nfc("Bật điều hòa") == "Bật điều hòa"
    assert "điều" in nfc("Bật điều hòa")


def test_relative_dates_resolve_to_absolute_dates():
    from datetime import datetime

    now = datetime.fromisoformat("2026-11-28T07:10:00+07:00")
    assert "2026-11-29" in resolve_relative_dates("mai", now)


def test_relative_day_table_lists_the_expected_days():
    from datetime import datetime

    now = datetime.fromisoformat("2026-11-28T07:10:00+07:00")
    table = relative_day_table(now, days=2)
    assert "2026-11-28" in table
    assert "2026-11-29" in table
    assert "2026-11-30" in table


def test_the_asr_transcript_is_recased_to_sentence_case():
    """gipformer emits uppercase; a non-fuzzy string argument is compared exactly."""
    from lifemate.text import recase

    assert recase("BẬT ĐIỀU HÒA PHÒNG KHÁCH") == "Bật điều hòa phòng khách"
    # Proper nouns lose their capital: "Mai Anh" -> "Mai anh". This is the documented
    # cost of the recasing (docs/DECISIONS.md, Decision 6), and it is safe because
    # free-text arguments are fuzzy, where both sides are lowercased before matching.
    assert recase("MAI ANH DẬY SÁNG") == "Mai anh dậy sáng"
    # a full stop between digits is a thousands separator, not a sentence end
    assert recase("200.000 ĐỒNG") == "200.000 Đồng"
    assert recase("CHUYỂN 2.500.000 ĐỒNG") == "Chuyển 2.500.000 Đồng"


def test_recasing_keeps_the_text_readable():
    from lifemate.text import recase

    assert recase("") == ""
    assert recase("XIN CHÀO") == "Xin chào"


# --- the guard ------------------------------------------------------------------

def test_guard_rejects_a_tool_that_is_not_installed(train_split):
    sample = next(train_split.iter_samples())
    installed = []
    for app in sample["context"]["installed_apps"].values():
        installed.extend(app["tools"])
    guard = Guard(train_split, allowed_tools=installed)
    result = guard.check("totally.unknown_tool", {})
    assert not result.ok
    assert "catalog" in result.reason or "cài" in result.reason


def test_guard_drops_unknown_and_empty_arguments(train_split):
    guard = Guard(train_split, observed_ids={"d1"})
    name = "smart_home.light_set"
    result = guard.check(name, {"device_id": "d1", "bogus": 5, "power": ""})
    assert result.ok
    assert set(result.arguments) == {"device_id"}


def test_guard_coerces_types_toward_the_declared_type(train_split):
    guard = Guard(train_split, observed_ids={"d1"})
    result = guard.check("smart_home.light_set", {"device_id": "d1", "brightness": "80"})
    assert result.ok
    assert result.arguments["brightness"] == 80
    assert result.repaired


def test_guard_blocks_an_ungrounded_id(train_split):
    guard = Guard(train_split, observed_ids=set())
    result = guard.check("smart_home.light_set", {"device_id": "never_seen"})
    assert not result.ok
    assert "x-ref" in result.reason or "quan sát" in result.reason


def test_guard_allows_a_grounded_id(train_split):
    guard = Guard(train_split, observed_ids={"device_01"})
    result = guard.check("smart_home.light_set", {"device_id": "device_01"})
    assert result.ok


def test_guard_reports_a_missing_required_argument(train_split):
    guard = Guard(train_split, observed_ids={"d1"})
    result = guard.check("smart_home.light_set", {})
    assert not result.ok
    assert "bắt buộc" in result.reason


# --- the prompt -----------------------------------------------------------------

def test_prompt_contains_the_installed_schemas_and_the_request(train_split):
    sample = next(train_split.iter_samples())
    prompt = build_prompt(train_split, sample, sample["audio_text"])
    assert "smart_home.light_set" in prompt
    assert sample["audio_text"] in prompt
    assert "Bảng ngày tương đối" in prompt
    # the request must not be drowned out by the schemas
    assert len(prompt) < 40000


def test_action_schema_covers_every_installed_tool_plus_finish(train_split):
    sample = next(train_split.iter_samples())
    schema = build_action_schema(train_split, sample)
    variants = schema["anyOf"]
    names = {v["properties"]["name"]["const"] for v in variants}
    assert "finish" in names
    for app in sample["context"]["installed_apps"].values():
        for tool in app["tools"]:
            assert tool in names


# --- action parsing --------------------------------------------------------------

def test_action_parsing_handles_plain_and_fenced_json():
    assert parse_action('{"name": "a.b", "arguments": {"x": 1}}').name == "a.b"
    assert parse_action('```json\n{"name": "a.b", "arguments": {}}\n```').name == "a.b"
    assert parse_action('Here you go: {"name": "a.b", "arguments": {}}').name == "a.b"
    assert parse_action("nonsense").is_finish


# --- the loop --------------------------------------------------------------------

def test_agent_loop_reproduces_the_gold_trajectory_and_scores_one(train_split):
    """Replaying the gold actions through the real loop must score exactly 1.0.

    This isolates the pipeline from the model: if it ever drops, the pipeline is broken.
    """
    samples = list(train_split.iter_samples())[:20]
    for sample in samples:
        gold = sample["label"]["tool_calls"]
        actions = [Action(name=c["name"], arguments=c["arguments"]) for c in gold]
        policy = ScriptedPolicy(actions)
        agent = Agent(train_split, policy, max_steps=8, time_budget_s=25.0)
        trajectory = agent.run(sample, sample["audio_text"])
        assert trajectory.tool_calls == gold
        assert trajectory.state_hash
    result = score_split(train_split, samples, {
        s["id"]: s["label"]["tool_calls"] for s in samples
    })
    assert result["mean"] == pytest.approx(1.0, abs=1e-9)


def test_agent_loop_stops_after_ask_missing_info(train_split):
    sample = next(
        s for s in train_split.iter_samples()
        if any(c["name"] == "sys.ask_missing_info" for c in s["label"]["tool_calls"])
    )
    gold = sample["label"]["tool_calls"]
    actions = [Action(name=c["name"], arguments=c["arguments"]) for c in gold]
    trajectory = Agent(train_split, ScriptedPolicy(actions)).run(sample, sample["audio_text"])
    assert trajectory.stop_reason == "ask_missing_info"
    assert trajectory.tool_calls[-1]["name"] == "sys.ask_missing_info"


def test_agent_loop_emits_nothing_for_an_out_of_scope_request(train_split):
    sample = next(
        s for s in train_split.iter_samples() if not s["label"]["tool_calls"]
    )
    trajectory = Agent(train_split, ScriptedPolicy([])).run(sample, sample["audio_text"])
    assert trajectory.tool_calls == []
    assert trajectory.stop_reason == "finish"


def test_the_guard_grounds_ids_from_the_conversation_history(train_split):
    """The ids a gold call uses are usually observed in an *earlier turn*.

    Measured on the train split: 1,277 of 1,278 gold x-ref values appear in the
    conversation history, but only 964 appear in the current turn. A guard that
    ignored the history would wrongly reject a fifth of the split.
    """
    total = 0
    grounded = 0
    for sample in train_split.iter_samples():
        observed: set[str] = set()
        for turn in sample.get("history") or []:
            for result in turn.get("tool_results") or []:
                collect_ids(result, observed)
        env = tool_env(train_split, sample["id"])
        for call in sample["label"]["tool_calls"]:
            for key in _xref_keys(train_split, call["name"]):
                value = (call.get("arguments") or {}).get(key)
                if value is None:
                    continue
                total += 1
                if not isinstance(value, str) or value in observed:
                    grounded += 1
            result = env.call(call["name"], call.get("arguments") or {})
            collect_ids(result, observed)
    assert total > 1000    # at most a handful of gold x-ref values are ungrounded
    assert grounded >= total - 5, f"only {grounded}/{total} x-ref values are grounded"


def test_the_agent_loop_accepts_a_gold_call_whose_id_came_from_the_history(train_split):
    """A sample whose only call uses a history-grounded id must not be rejected."""
    for sample in train_split.iter_samples():
        calls = sample["label"]["tool_calls"]
        if not calls or not sample.get("history"):
            continue
        keys = _xref_keys(train_split, calls[0]["name"])
        if not keys:
            continue
        value = (calls[0].get("arguments") or {}).get(keys[0])
        if not isinstance(value, str):
            continue
        observed: set[str] = set()
        for turn in sample["history"]:
            for result in turn.get("tool_results") or []:
                collect_ids(result, observed)
        if value not in observed:
            continue
        actions = [Action(name=c["name"], arguments=c["arguments"]) for c in calls]
        trajectory = Agent(train_split, ScriptedPolicy(actions)).run(
            sample, sample["audio_text"])
        assert trajectory.tool_calls == calls, (
            f"{sample['id']}: the guard rejected a valid history-grounded call")
        return
    pytest.skip("no sample in the kit matches this shape")


def _xref_keys(split, tool_name):
    """The argument names of a tool that the catalog marks as x-ref."""
    for app in split.catalog().get("apps", []) or []:
        for tool in app.get("tools", []) or []:
            if tool["name"] != tool_name:
                continue
            properties = (tool.get("parameters") or {}).get("properties") or {}
            return [k for k, v in properties.items() if v.get("x-ref")]
    return []


def test_the_guard_refuses_an_identical_repeated_call(train_split):
    """A repeated write corrupts the state (two messages instead of one)."""
    guard = Guard(train_split, observed_ids={"d1"})
    first = guard.check("smart_home.light_set", {"device_id": "d1"})
    assert first.ok
    dispatched = {("smart_home.light_set", '{"device_id": "d1"}')}
    second = guard.check("smart_home.light_set", {"device_id": "d1"}, dispatched)
    assert not second.ok
    assert "đã được thực hiện" in second.reason


def test_the_guard_allows_the_same_tool_with_different_arguments(train_split):
    """163 gold samples call the same tool twice; only exact repeats are refused."""
    guard = Guard(train_split, observed_ids={"d1", "d2"})
    first = guard.check("smart_home.light_set", {"device_id": "d1"})
    assert first.ok
    dispatched = {("smart_home.light_set", '{"device_id": "d1"}')}
    second = guard.check("smart_home.light_set", {"device_id": "d2"}, dispatched)
    assert second.ok


def test_the_agent_loop_does_not_dispatch_an_identical_call_twice(train_split):
    """The loop must refuse the repeat, so the state is not corrupted."""
    for sample in train_split.iter_samples():
        calls = sample["label"]["tool_calls"]
        if len(calls) < 2:
            continue
        key = lambda c: (c["name"], json.dumps(c.get("arguments") or {}, sort_keys=True))
        if key(calls[0]) != key(calls[1]):
            continue
        actions = [Action(name=c["name"], arguments=c["arguments"]) for c in calls]
        trajectory = Agent(train_split, ScriptedPolicy(actions)).run(
            sample, sample["audio_text"])
        assert len(trajectory.tool_calls) == len(calls) - 1, (
            f"{sample['id']}: the identical call was dispatched twice")
        return
    pytest.skip("no sample in the kit repeats an identical call")


def test_submission_row_has_the_required_shape(train_split):
    sample = next(train_split.iter_samples())
    trajectory = Agent(train_split, ScriptedPolicy([])).run(sample, sample["audio_text"])
    row = trajectory.as_submission_row()
    assert set(row) == {"id", "tool_calls", "state_hash"}
    assert row["id"] == sample["id"]
    json.dumps(row, ensure_ascii=False)
