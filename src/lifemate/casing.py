"""The transcript casing policy: one switch, three settings.

The recogniser emits uppercase text with no punctuation, so every capital letter the
user actually said is lost before the policy model sees the utterance. Two ways to put
it back:

* **code** — the rule module (``capitalize.py``) and the context model over it
  (``case_model.py``). Both are kept, because they are the baseline the ablation
  measures against, but they are **not in the runtime path any more** (Decision 24).
* **the policy model itself** — the supervised model is trained on lowercase
  transcripts paired with the gold arguments, which are correctly cased, so it learns
  that "mai anh" is "Mai Anh" and brings the pretrained knowledge of Vietnamese names
  that a vocabulary cannot.

This module is the single place where that choice is made, so the training prompt and
the inference prompt can never disagree about it.

    none    lowercase only (default) — the policy model casts the arguments
    rules   capitalize.restore_capitals — the rule module alone
    model   case_model.transcript — the shipped context model (the ablation baseline)

The setting is read from ``LIFEMATE_CASE_MODE`` unless it is set explicitly, so an
ablation arm can flip it without touching the code.

Every mode is **idempotent**: applying it twice gives the same text. That matters
because the transcript is cased at the point it becomes model input, and the same text
may reach that point by two different routes (the current request and a history turn).
"""
from __future__ import annotations

import os

NONE = "none"
RULES = "rules"
MODEL = "model"

MODES = (NONE, RULES, MODEL)
DEFAULT_MODE = NONE

ENV_VAR = "LIFEMATE_CASE_MODE"

_MODE: str | None = None


def get_mode() -> str:
    """The active casing mode.

    Read from the environment on first use, so a run can be pointed at a mode without
    a code change; :func:`set_mode` overrides it for the rest of the process.
    """
    global _MODE
    if _MODE is None:
        raw = os.environ.get(ENV_VAR, DEFAULT_MODE).strip().lower()
        if raw not in MODES:
            raise ValueError(
                f"unknown casing mode {raw!r} in {ENV_VAR}; expected one of "
                f"{', '.join(MODES)}"
            )
        _MODE = raw
    return _MODE


def set_mode(mode: str | None) -> str:
    """Set the active casing mode. ``None`` restores the environment default."""
    global _MODE
    if mode is None:
        _MODE = None
        return get_mode()
    cleaned = mode.strip().lower()
    if cleaned not in MODES:
        raise ValueError(
            f"unknown casing mode {mode!r}; expected one of {', '.join(MODES)}"
        )
    _MODE = cleaned
    return _MODE


def apply_case(text: str, mode: str | None = None) -> str:
    """Return the text as the policy model should see it.

    ``mode`` defaults to the active mode. The two restoring modes uppercase their
    input first, so they are idempotent and safe to apply to text that has already
    been cased.
    """
    if not text:
        return text
    active = (mode or get_mode()).strip().lower()
    if active == NONE:
        return text.lower()
    if active == RULES:
        from .capitalize import restore_capitals

        return restore_capitals(text.upper())
    if active == MODEL:
        from .case_model import transcript

        return transcript(text.upper())
    raise ValueError(
        f"unknown casing mode {active!r}; expected one of {', '.join(MODES)}"
    )


def is_recasing(mode: str | None = None) -> bool:
    """True when the mode restores capitalisation instead of lowercasing."""
    return (mode or get_mode()).strip().lower() != NONE
