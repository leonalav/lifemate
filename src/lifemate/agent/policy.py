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

    Constrained decoding follows whatever vLLM version is installed: newer versions
    renamed ``GuidedDecodingParams`` to ``StructuredOutputsParams`` (removed in
    v0.12) and accept it as ``structured_outputs=``; older ones accept it as
    ``guided_decoding=`` or the raw schema as ``guided_json=``. The helper below
    probes the installed API so the same code works on Colab and in Docker.
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
        params = _make_sampling_params(schema)
        outputs = self.llm.generate([prompt], params)
        return parse_action(outputs[0].outputs[0].text)


def _make_sampling_params(schema: dict):
    """Build SamplingParams with JSON-constrained decoding on any vLLM API.

    v0.12 removed ``GuidedDecodingParams`` in favour of ``StructuredOutputsParams``
    (``structured_outputs=``); older versions take ``guided_decoding=`` or the raw
    ``guided_json=`` schema. The installed signature is probed so Colab (new) and
    the Docker image (pinned older) both work, instead of failing mid-run.
    """
    import inspect

    from vllm.sampling_params import SamplingParams

    base = {
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20261028,
        "max_tokens": 512,
    }
    try:
        accepted = set(inspect.signature(SamplingParams).parameters)
    except (TypeError, ValueError):
        accepted = set()

    def _try_kwargs(extra: dict):
        try:
            return SamplingParams(**base, **extra)
        except TypeError:
            return None

    # New API first: StructuredOutputsParams(json=schema) as structured_outputs=.
    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        StructuredOutputsParams = None  # type: ignore[assignment]
    if StructuredOutputsParams is not None:
        for key in ("structured_outputs", "guided_decoding"):
            if not accepted or key in accepted:
                try:
                    structured = StructuredOutputsParams(json=schema)
                except TypeError:
                    continue
                params = _try_kwargs({key: structured})
                if params is not None:
                    return params

    # Old API: GuidedDecodingParams(json=schema) as guided_decoding=.
    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        GuidedDecodingParams = None  # type: ignore[assignment]
    if GuidedDecodingParams is not None and (not accepted or "guided_decoding" in accepted):
        try:
            guided = GuidedDecodingParams(json=schema)
        except TypeError:
            guided = None
        if guided is not None:
            params = _try_kwargs({"guided_decoding": guided})
            if params is not None:
                return params

    # Oldest API: the raw schema as guided_json=.
    if not accepted or "guided_json" in accepted:
        params = _try_kwargs({"guided_json": schema})
        if params is not None:
            return params

    raise TypeError(
        "installed vLLM SamplingParams accepts none of "
        "'structured_outputs' / 'guided_decoding' / 'guided_json'; "
        f"signature parameters: {sorted(accepted) if accepted else 'unknown'}"
    )
