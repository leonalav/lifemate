"""The agent loop.

One model step produces exactly one action; a deterministic guard sits between the model
and the environment; the environment's observation is fed back. Each dispatched call is
therefore independently attributable in the journal and independently fixable
(docs/DECISIONS.md, Decision 4).

Competition rules this loop enforces:

* **Exactly one ToolEnv session per sample.** The session is created once and every call
  made on it appears in the submission. Nothing under ``<split>_environment/`` is ever read
  directly.
* The run stops on ``finish``, after ``sys.ask_missing_info`` (gold always ends the turn
  there), at ``max_steps``, or when the time budget is spent.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime

from ..data import Split
from ..kit import tool_env
from .guard import Guard, collect_ids, collect_strings
from .policy import Action, Policy
from .prompt import build_action_schema, build_prompt

ASK_MISSING_INFO = "sys.ask_missing_info"


def _sample_clock(sample: dict) -> tuple[datetime | None, str]:
    """The sample's clock: 'now' and its UTC offset, e.g. ('2026-09-25T09:00+07:00', '+07:00').

    Dates and times are resolved against the sample, never against the wall clock of
    the machine running the agent.
    """
    raw = (sample.get("context") or {}).get("current_time")
    if not raw:
        return None, "+07:00"
    try:
        now = datetime.fromisoformat(raw)
    except ValueError:
        return None, "+07:00"
    offset = now.utcoffset()
    if offset is None:
        return now, "+07:00"
    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)
    return now, f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def _stable(arguments: dict) -> str:
    """A stable key for an argument dict, so two equal calls compare equal."""
    import json

    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


@dataclass
class Step:
    """One model step and what happened to its proposal."""

    index: int
    action: Action
    accepted: bool
    reason: str | None = None
    result: dict | None = None
    repaired: bool = False


@dataclass
class Trajectory:
    """The full record of one sample's run."""

    sample_id: str
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    state_hash: str = ""
    stop_reason: str = ""

    def as_submission_row(self) -> dict:
        """The submission line: {"id", "tool_calls", "state_hash"}."""
        return {
            "id": self.sample_id,
            "tool_calls": self.tool_calls,
            "state_hash": self.state_hash,
        }


class Agent:
    """Runs one sample through the policy, the guard and the environment."""

    def __init__(self, split: Split, policy: Policy, max_steps: int = 8,
                 time_budget_s: float = 25.0, transcript_fn=None, case_fn=None,
                 case_mode: str | None = None):
        self.split = split
        self.policy = policy
        self.max_steps = max_steps
        self.time_budget_s = time_budget_s
        self.transcript_fn = transcript_fn
        # An optional truecaser: it contributes a casing *hint* to the prompt, and
        # nothing else. None when the weights are not in the image.
        self.case_fn = case_fn
        self.case_mode = case_mode

    def run(self, sample: dict, transcript: str) -> Trajectory:
        """Run one sample and return its trajectory."""
        started = time.monotonic()
        sample_id = sample["id"]
        trajectory = Trajectory(sample_id=sample_id)

        # One session per sample, created once.
        session = tool_env(self.split, sample_id)

        # The sample's own clock, used to resolve dates and times.
        clock = _sample_clock(sample)

        # Ground the ids from the conversation history first. Measured on the train split:
        # 1,277 of 1,278 gold x-ref values appear in an earlier turn's tool results, but
        # only 964 appear in the *current* turn. A guard that ignored the history would
        # wrongly reject 254 samples — a fifth of the split.
        observed: set[str] = set()
        observed_strings: dict[str, str] = {}
        for turn in sample.get("history") or []:
            for result in turn.get("tool_results") or []:
                collect_ids(result, observed)
                collect_strings(result, observed_strings)

        guard = Guard(self.split, allowed_tools=_installed(sample),
                      now=clock[0], tz_offset=clock[1])
        guard.observed_ids = observed
        guard.observed_strings = observed_strings

        schema = build_action_schema(self.split, sample)

        # A repeated identical call is refused (see Guard.check).
        dispatched: set[tuple] = set()

        # A rejected proposal is fed back so the model can correct itself.
        working = dict(sample)
        working["history"] = list(sample.get("history") or [])

        for step_index in range(self.max_steps):
            if time.monotonic() - started > self.time_budget_s:
                trajectory.stop_reason = "time_budget"
                break

            prompt = build_prompt(
                self.split, working, transcript,
                transcript_for_history=self.transcript_fn,
                case_mode=self.case_mode,
                case_fn=self.case_fn,
            )
            action = self.policy.propose(prompt, schema, sample)

            if action.is_finish:
                trajectory.steps.append(Step(step_index, action, False, "finish"))
                trajectory.stop_reason = "finish"
                break

            checked = guard.check(action.name, action.arguments, dispatched)
            if not checked.ok:
                # A rejected proposal is not dispatched: it costs nothing against the
                # metric, and the model is told why so it can correct itself.
                trajectory.steps.append(
                    Step(step_index, action, False, checked.reason)
                )
                working["history"] = working["history"] + [{
                    "role": "assistant",
                    "text": "",
                    "tool_calls": [{"name": action.name,
                                    "arguments": action.arguments}],
                    "tool_results": [{"ok": False, "error": checked.reason}],
                }]
                if step_index + 1 >= self.max_steps:
                    trajectory.stop_reason = "max_steps"
                continue

            result = session.call(checked.name, checked.arguments)
            trajectory.tool_calls.append(
                {"name": checked.name, "arguments": checked.arguments}
            )
            dispatched.add((checked.name, _stable(checked.arguments)))
            trajectory.steps.append(
                Step(step_index, action, True, None, result, checked.repaired)
            )
            collect_ids(result, observed)
            collect_strings(result, observed_strings)
            guard.observed_ids = observed
            guard.observed_strings = observed_strings

            # The observation goes back into the prompt, so the next step sees it.
            working["history"] = working["history"] + [{
                "role": "assistant",
                "text": "",
                "tool_calls": [{"name": checked.name,
                                "arguments": checked.arguments}],
                "tool_results": [result],
            }]

            if checked.name == ASK_MISSING_INFO:
                # gold always ends the turn after ask_missing_info
                trajectory.stop_reason = "ask_missing_info"
                break
        else:
            trajectory.stop_reason = trajectory.stop_reason or "max_steps"

        trajectory.state_hash = session.get_state_hash()
        # Every call made on the session, including any the loop dispatched directly.
        trajectory.tool_calls = session.get_tool_calls()
        return trajectory


def _installed(sample: dict) -> list[str]:
    from ..data import installed_tools

    tools = installed_tools(sample)
    tools.append(ASK_MISSING_INFO)
    return tools
