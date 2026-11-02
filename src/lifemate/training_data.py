"""Building the next-action training set.

Each train sample's gold trajectory is unrolled against a live ``ToolEnv``: at step *t* the
input is the prompt plus the history of gold actions and the *real* observations produced
so far, and the target is gold action *t+1* (or ``finish``). Multi-call samples therefore
yield several examples — roughly 3.5k action examples from 1,545 samples.

Transcripts used in training are the **stock-gipformer outputs** (the test-time
distribution), optionally mixed with the gold transcript for stability. ASR-fine-tuned
transcripts are deliberately not used for LLM training: a fine-tuned ASR memorises train
audio and hides the real error modes from the policy (docs/PRD.md §5.1).

The transcript reaches the model in the pipeline's casing policy — lowercase by
default — while the target (the gold action) keeps the user's own capitalisation. The
model therefore has to learn to cast the arguments itself (docs/DECISIONS.md,
Decision 24).

One ``ToolEnv`` session per sample, created once — the same rule the agent loop follows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .agent.prompt import build_prompt
from .data import Split
from .kit import tool_env


@dataclass
class Example:
    """One supervised example: a prompt and the action that follows it."""

    prompt: str
    completion: str
    sample_id: str
    step: int
    app: str


def _completion(action: dict | None) -> str:
    """The target: a single JSON action, or ``finish``."""
    if action is None:
        return json.dumps({"name": "finish"}, ensure_ascii=False)
    return json.dumps(
        {"name": action.get("name"), "arguments": action.get("arguments") or {}},
        ensure_ascii=False,
        sort_keys=True,
    )


def build_examples(split: Split, samples: list[dict], transcripts: dict[str, str] | None = None,
                   transcript_for_turn=None, limit: int | None = None,
                   case_mode: str | None = None, case_fn=None) -> list[Example]:
    """Unroll the gold trajectories into next-action examples.

    ``transcripts`` maps sample id -> transcript text (the ASR cache, keyed by the
    sample's ``audio`` field). Samples without an entry fall back to the split's own
    transcript field (train carries ``audio_text``).

    ``transcript_for_turn`` resolves the transcript of a *history* turn the same way,
    so a history turn reads like the recogniser's output rather than like the
    reference transcript.

    The completion is the gold action, whose string arguments are correctly cased;
    the prompt holds the transcript in the pipeline's casing policy (lowercase by
    default). That mismatch is the point: the model has to learn to cast the
    arguments itself (docs/DECISIONS.md, Decision 24).
    """
    examples: list[Example] = []
    transcripts = transcripts or {}

    def turn_transcript(turn: dict, sample_id: str = "") -> str:
        if transcript_for_turn is not None:
            found = transcript_for_turn(turn, sample_id)
            if found:
                return found
        return turn.get("audio_text") or ""

    for sample in samples[:limit] if limit else samples:
        sample_id = sample["id"]
        transcript = transcripts.get(sample_id) or sample.get("audio_text") or ""
        if not transcript:
            continue

        def history_text(turn: dict, _sample_id: str = sample_id) -> str:
            return turn_transcript(turn, _sample_id)

        gold_calls = (sample.get("label") or {}).get("tool_calls") or []
        session = tool_env(split, sample_id)

        # The prompt is rebuilt at every step so the observation history grows, exactly as
        # it does at inference time.
        working = _working_sample(sample)
        history: list[dict] = []

        for step, call in enumerate(gold_calls):
            prompt = build_prompt(
                split, working, transcript,
                transcript_for_history=history_text,
                case_mode=case_mode,
                case_fn=case_fn,
            )
            examples.append(Example(
                prompt=prompt,
                completion=_completion(call),
                sample_id=sample_id,
                step=step,
                app=(call.get("name") or "").split(".")[0],
            ))
            # Apply the gold call and feed the real observation back.
            result = session.call(call.get("name", ""), call.get("arguments") or {})
            history.append({
                "role": "assistant",
                "text": "",
                "tool_calls": [call],
                "tool_results": [result],
            })

        # The final target: stop. Only emitted when the trajectory did not end with
        # ask_missing_info, because gold always ends the turn there.
        if not gold_calls or gold_calls[-1].get("name") != "sys.ask_missing_info":
            working["history"] = (working.get("history") or []) + history
            prompt = build_prompt(
                split, working, transcript,
                transcript_for_history=history_text,
                case_mode=case_mode,
                case_fn=case_fn,
            )
            examples.append(Example(
                prompt=prompt,
                completion=_completion(None),
                sample_id=sample_id,
                step=len(gold_calls),
                app="",
            ))

    return examples


def _working_sample(sample: dict) -> dict:
    """A copy of the sample whose history grows with the gold trajectory."""
    working = dict(sample)
    working["history"] = list(sample.get("history") or [])
    return working


def write_jsonl(examples: list[Example], path: str | Path) -> None:
    """Write the examples as JSONL (prompt, completion) for the trainer."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(
                {"prompt": example.prompt, "completion": example.completion},
                ensure_ascii=False,
            ) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def current_time(sample: dict) -> datetime:
    value = (sample.get("context") or {}).get("current_time")
    return datetime.fromisoformat(value) if value else datetime.now().astimezone()
