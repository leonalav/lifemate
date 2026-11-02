"""The LLM policy: one action per step.

The model is consulted through a narrow interface so the agent loop can be tested on CPU
without a GPU and without weights — the pipeline (prompt, guard, environment, submission)
is exercised by a scripted policy instead.

At inference the policy is served by vLLM with greedy decoding and a fixed seed; the action
is constrained to the JSON schema in :mod:`lifemate.agent.prompt`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Action:
    """One proposed action from the model."""

    name: str
    arguments: dict = field(default_factory=dict)
    raw: str = ""

    @property
    def is_finish(self) -> bool:
        return self.name == "finish"


class Policy(Protocol):
    """Anything that can propose the next action."""

    def propose(self, prompt: str, schema: dict, sample: dict) -> Action:
        ...


def parse_action(raw: str) -> Action:
    """Parse one action out of the model's output.

    Tolerates a fenced code block and surrounding prose, because a constrained decoder
    emits clean JSON but a fallback decoder might not.
    """
    text = (raw or "").strip()
    if not text:
        return Action(name="finish", arguments={}, raw=raw)

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return Action(name="finish", arguments={}, raw=raw)

    if not isinstance(payload, dict):
        return Action(name="finish", arguments={}, raw=raw)
    name = payload.get("name") or payload.get("tool") or "finish"
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("parameters") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return Action(name=str(name), arguments=arguments, raw=raw)


class ScriptedPolicy:
    """A policy that replays a fixed list of actions.

    Used by the tests and by the local dry-run to exercise the pipeline end to end
    without any model. Not used at inference.
    """

    def __init__(self, actions: list[Action] | None = None):
        self.actions = list(actions or [])
        self.calls = 0

    def propose(self, prompt: str, schema: dict, sample: dict) -> Action:
        if self.calls < len(self.actions):
            action = self.actions[self.calls]
        else:
            action = Action(name="finish", arguments={})
        self.calls += 1
        return action


class VllmPolicy:
    """vLLM-backed policy.

    Greedy decoding (temperature 0) with a fixed seed: the competition requires
    deterministic decoding, and the organisers re-run the image five times and average.
    """

    def __init__(self, model_path: str, max_model_len: int = 16384,
                 gpu_memory_utilization: float = 0.85, seed: int = 20261028):
        from vllm import LLM  # imported lazily: vLLM is not installed on a CPU box

        self.llm = LLM(
            model=model_path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            enforce_eager=False,
            dtype="bfloat16",
        )
        self.max_model_len = max_model_len

    def propose(self, prompt: str, schema: dict, sample: dict) -> Action:
        from vllm.sampling_params import GuidedDecodingParams, SamplingParams

        params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            seed=20261028,
            max_tokens=512,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
        outputs = self.llm.generate([prompt], params)
        return parse_action(outputs[0].outputs[0].text)
