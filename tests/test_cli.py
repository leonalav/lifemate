"""The CLI surface, exercised the way the Colab launcher exercises it.

These tests are the local proof that the commands in ``colab_launcher.ipynb`` work:
the same flags, the same paths, the same outputs. No model is involved — the infer
command is driven through the pipeline directly, and the train command is checked for
its argument handling and its checkpoint bookkeeping.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.agent.policy import ScriptedPolicy  # noqa: E402
from lifemate.cli import build_parser  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.infer import InferConfig, run_inference, validate_submission  # noqa: E402
from lifemate.train import (  # noqa: E402
    TrainConfig,
    choose_holdout_apps,
    last_checkpoint,
    load_state,
    save_state,
    split_samples,
    TrainState,
)

DATA_DIR = REPO_ROOT / "data" / "train+public_test"


@pytest.fixture(scope="module")
def train_split():
    if not (DATA_DIR / "train").is_dir():
        pytest.skip("the competition kit is not present")
    return resolve_split(DATA_DIR, "train")


# --- the frozen CLI contract -----------------------------------------------------

def test_every_verb_in_the_prd_exists():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions
    verbs = set(actions[0].choices)
    assert {"train", "infer", "score", "transcribe"} <= verbs


def test_infer_accepts_the_flags_the_notebook_passes():
    parser = build_parser()
    args = parser.parse_args([
        "infer",
        "--data-dir", "/content/data_local",
        "--split", "public_test",
        "--checkpoint", "/drive/checkpoints",
        "--output-file", "/drive/submission.jsonl",
        "--max-steps", "8",
        "--time-budget-s", "25",
    ])
    assert args.command == "infer"
    assert args.data_dir == "/content/data_local"
    assert args.split == "public_test"
    assert args.max_steps == 8
    assert args.time_budget_s == 25


def test_train_accepts_the_flags_the_notebook_passes():
    parser = build_parser()
    args = parser.parse_args([
        "train",
        "--data-dir", "/content/data_local",
        "--output-dir", "/drive/runs",
        "--config", "/content/lifemate/configs/sft_qlora.yaml",
        "--epochs", "3",
        "--batch-size", "1",
        "--resume",
    ])
    assert args.command == "train"
    assert args.resume == "auto"
    assert args.epochs == 3


def test_score_accepts_the_flags_the_notebook_passes():
    parser = build_parser()
    args = parser.parse_args([
        "score",
        "--split", "train",
        "--submission", "/drive/submission.jsonl",
        "--report", "/drive/scores.jsonl",
    ])
    assert args.command == "score"
    assert args.split == "train"


def test_transcribe_accepts_the_flags_the_prd_defines():
    parser = build_parser()
    args = parser.parse_args([
        "transcribe",
        "--split", "public_test",
        "--data-dir", "/content/data_local",
        "--out", "/drive/transcripts.jsonl",
    ])
    assert args.command == "transcribe"
    assert args.out == "/drive/transcripts.jsonl"


# --- the config ------------------------------------------------------------------

def test_the_shipped_training_config_loads():
    config = TrainConfig.from_yaml(REPO_ROOT / "configs" / "sft_qlora.yaml")
    assert config.base_model == "Qwen/Qwen3.5-4B"
    assert config.max_seq_len <= 16384
    assert config.lora_r > 0


def test_an_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("base_model: x\nnot_a_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_key"):
        TrainConfig.from_yaml(path)


# --- holdout splitting -------------------------------------------------------------

def test_the_holdout_contains_apps_the_training_set_never_sees(train_split):
    samples = list(train_split.iter_samples())
    holdout_apps = choose_holdout_apps(samples, 10, 20261028)
    assert len(holdout_apps) == 10
    train_samples, holdout_samples = split_samples(samples, holdout_apps)
    assert train_samples and holdout_samples
    # no sample may appear on both sides
    train_ids = {s["id"] for s in train_samples}
    holdout_ids = {s["id"] for s in holdout_samples}
    assert not (train_ids & holdout_ids)
    # every holdout sample uses at least one held-out app
    for sample in holdout_samples:
        apps = set(sample["context"]["installed_apps"])
        assert apps & holdout_apps


# --- checkpointing ------------------------------------------------------------------

def test_a_resume_state_round_trips(tmp_path):
    state = TrainState(epoch=2, global_step=40, best_metric=0.5)
    path = save_state(state, tmp_path, "last")
    assert (path / "trainer_state.json").is_file()
    loaded = load_state(path)
    assert loaded.epoch == 2
    assert loaded.global_step == 40
    assert loaded.best_metric == 0.5
    assert last_checkpoint(tmp_path) == path


def test_no_checkpoint_is_reported_when_there_is_none(tmp_path):
    assert last_checkpoint(tmp_path) is None


# --- inference and scoring -----------------------------------------------------------

def test_inference_writes_a_valid_submission(train_split, tmp_path):
    """The whole infer path: run the agent over the split, write the file, validate it."""
    output = tmp_path / "submission.jsonl"
    config = InferConfig(max_steps=8, time_budget_s=25.0, verbose=False)
    report = run_inference(DATA_DIR, "train", ScriptedPolicy([]), str(output), config)

    assert report.rows_written == len(train_split.ids())
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == report.rows_written
    for row in rows:
        assert set(row) == {"id", "tool_calls", "state_hash"}
        assert row["id"]
        assert isinstance(row["tool_calls"], list)
        assert isinstance(row["state_hash"], str)

    result = validate_submission(output, train_split)
    assert result["ok"], result


def test_inference_honours_the_limit(train_split, tmp_path):
    output = tmp_path / "limited.jsonl"
    config = InferConfig(max_steps=8, time_budget_s=25.0, verbose=False, limit=5)
    report = run_inference(DATA_DIR, "train", ScriptedPolicy([]), str(output), config)
    assert report.rows_written == 5
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 5


def test_inference_honours_the_ids_file(train_split, tmp_path):
    ids = train_split.ids()[:3]
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\n".join(ids), encoding="utf-8")
    output = tmp_path / "subset.jsonl"
    config = InferConfig(max_steps=8, time_budget_s=25.0, verbose=False,
                         ids_file=str(ids_file))
    report = run_inference(DATA_DIR, "train", ScriptedPolicy([]), str(output), config)
    assert report.rows_written == 3
    written = [json.loads(line)["id"]
               for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert written == ids


def test_a_gold_replay_submission_scores_one_hundred(train_split, tmp_path):
    """The two numbers the PRD quotes, through the CLI's own code paths."""
    from lifemate.metric import score_split

    samples = list(train_split.iter_samples())[:25]
    output = tmp_path / "gold.jsonl"
    with output.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps({
                "id": sample["id"],
                "tool_calls": sample["label"]["tool_calls"],
                "state_hash": "",
            }, ensure_ascii=False) + "\n")
    result = score_split(train_split, samples, {
        s["id"]: s["label"]["tool_calls"] for s in samples
    })
    assert result["score"] == pytest.approx(100.0, abs=1e-6)


def test_validation_detects_a_missing_sample(train_split, tmp_path):
    output = tmp_path / "partial.jsonl"
    sample = next(train_split.iter_samples())
    output.write_text(json.dumps({"id": sample["id"], "tool_calls": [],
                                   "state_hash": ""}, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    result = validate_submission(output, train_split)
    assert not result["ok"]
    assert result["missing"]


def test_validation_detects_an_invalid_call(train_split, tmp_path):
    sample = next(train_split.iter_samples())
    output = tmp_path / "invalid.jsonl"
    output.write_text(json.dumps({
        "id": sample["id"],
        "tool_calls": [{"name": "not.a.tool", "arguments": {}}],
        "state_hash": "",
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    result = validate_submission(output, train_split)
    assert not result["ok"]
    assert result["invalid_calls"]
