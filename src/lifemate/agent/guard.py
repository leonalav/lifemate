"""The deterministic guard: the code between the model and the environment.

No parameters, no learning. Every check here removes a failure mode the metric punishes
hard (docs/VERIFIED_FACTS.md §12.3):

  drop a required argument      -> mean 0.069, 355/400 samples killed by the state gate
  perturb a typed argument      -> mean 0.213, 268/400 killed
  drop the last call            -> mean 0.192, 292/400 killed
  drop an optional argument     -> mean 0.511, 164/400 killed
  wrong tool name               -> mean 0.596, 50/400 killed
  an int passed as a string     -> mean 0.773, 91/400 killed
  a fuzzy argument perturbed    -> mean 1.000, 0 killed  (free text never touches the hash)

So the priority is: required arguments and typed values first, argument key sets next,
fuzzy wording last. The guard implements exactly that order, and reports the *cause* of a
rejection rather than the consequence (docs/DECISIONS.md, Decision 7).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from ..data import Split
from ..kit import schema_error
from ..text import nfc, normalize_date_value, normalize_number_value

_EMPTY = (None, "", [], {})


@dataclass
class GuardResult:
    """The outcome of checking one proposed call."""

    ok: bool
    name: str
    arguments: dict = field(default_factory=dict)
    reason: str | None = None
    repaired: bool = False


class Guard:
    """Checks and repairs proposed tool calls before they touch the environment."""

    def __init__(self, split: Split, allowed_tools: list[str] | None = None,
                 observed_ids: set[str] | None = None,
                 now: datetime | None = None, tz_offset: str = "+07:00"):
        self.split = split
        self.catalog = split.catalog()
        self.tools = {
            tool["name"]: tool
            for app in self.catalog.get("apps", []) or []
            for tool in app.get("tools", []) or []
        }
        self.allowed = set(allowed_tools) if allowed_tools is not None else None
        self.observed_ids = set(observed_ids or ())
        # The sample's clock. Dates and times are resolved against it, never against
        # the wall clock of the machine running the agent.
        self.now = now
        self.tz_offset = tz_offset

    # -- 1. installed? ------------------------------------------------------------------
    def check_installed(self, name: str) -> str | None:
        """The tool must exist in the catalog and be installed on this sample."""
        if name not in self.tools:
            return f"tool {name!r} không có trong catalog"
        if self.allowed is not None and name not in self.allowed:
            return f"tool {name!r} không thuộc app đã cài"
        return None

    # -- 3. argument key set ------------------------------------------------------------
    def check_argument_keys(self, name: str, arguments: dict) -> tuple[dict, bool, str | None]:
        """Drop unknown keys and empty optional values. Never invent an optional argument."""
        spec = self.tools.get(name) or {}
        properties = (spec.get("parameters") or {}).get("properties") or {}
        kept = {}
        for key, value in (arguments or {}).items():
            if key not in properties:
                continue  # unknown key: the matcher compares the exact key set
            if value in _EMPTY:
                continue  # an empty optional value counts as not given
            kept[key] = value
        return kept, kept != (arguments or {}), None

    # -- 3b. typed value normalisation ----------------------------------------------------
    def normalize_values(self, name: str, arguments: dict) -> tuple[dict, bool]:
        """Repair numeric and date arguments into the shape the metric compares.

        Only arguments whose *declared* type is numeric or date-shaped are touched;
        free text keeps the user's own wording, because the metric compares it
        fuzzily (token-F1 > 0.6) and rewriting it would break the match.

        * a spoken number becomes a typed number: 'bốn mươi phần trăm' -> 40
        * a date/time is completed or resolved against the sample's clock:
          '2026-09-26T09:00' -> '2026-09-26T09:00:00+07:00', 'chín giờ sáng mai'
          -> '2026-09-26T09:00:00+07:00'

        The timezone offset is added because it lands in the environment state and
        therefore in the state hash: dropping it changes the hash and scores 0.
        """
        spec = self.tools.get(name) or {}
        properties = (spec.get("parameters") or {}).get("properties") or {}
        out = dict(arguments or {})
        changed = False
        for key, value in out.items():
            prop = properties.get(key) or {}
            declared = prop.get("type")
            fmt = prop.get("format")
            repaired = None
            if declared in ("integer", "number"):
                repaired = normalize_number_value(value, declared)
            elif declared == "string" and fmt in ("date", "date-time", "time"):
                if self.now is not None:
                    repaired = normalize_date_value(value, fmt, self.now, self.tz_offset)
            if repaired is not None and repaired != value:
                out[key] = repaired
                changed = True
        return out, changed

    # -- 4. type coercion ----------------------------------------------------------------
    def coerce_types(self, name: str, arguments: dict) -> tuple[dict, bool, str | None]:
        """Coerce values toward the declared type, bidirectionally.

        "26" -> 26 for an integer, 26 -> "26" for a string, "2.000.000" -> 2000000.
        """
        spec = self.tools.get(name) or {}
        properties = (spec.get("parameters") or {}).get("properties") or {}
        out = dict(arguments)
        changed = False
        for key, value in arguments.items():
            declared = (properties.get(key) or {}).get("type")
            coerced = self._coerce(value, declared)
            if coerced is not _EMPTY and coerced != value:
                out[key] = coerced
                changed = True
        return out, changed, None

    @staticmethod
    def _coerce(value, declared):
        if declared == "integer":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                digits = value.strip().replace(".", "").replace(" ", "")
                if re.fullmatch(r"-?\d+", digits):
                    return int(digits)
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
        elif declared == "number":
            if isinstance(value, str):
                digits = value.strip().replace(".", "").replace(" ", "").replace(",", ".")
                if re.fullmatch(r"-?\d+(\.\d+)?", digits):
                    return float(digits)
                return value
        elif declared == "string":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
        elif declared == "boolean":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "có", "co", "1", "yes"):
                    return True
                if lowered in ("false", "không", "khong", "0", "no"):
                    return False
        return value

    # -- 5. x-ref grounding ---------------------------------------------------------------
    def check_grounded(self, name: str, arguments: dict) -> str | None:
        """Every x-ref identifier must have appeared in an observed result."""
        spec = self.tools.get(name) or {}
        properties = (spec.get("parameters") or {}).get("properties") or {}
        for key, value in (arguments or {}).items():
            prop = properties.get(key) or {}
            if not prop.get("x-ref"):
                continue
            if not isinstance(value, str):
                continue
            if value not in self.observed_ids:
                return (
                    f"tham số {key!r}={value!r} chưa được quan sát trong kết quả tool "
                    f"trước đó (x-ref: {prop['x-ref']})"
                )
        return None

    # -- 7. x-one-of ------------------------------------------------------------------------
    def check_one_of(self, name: str, arguments: dict) -> str | None:
        """At least one group of an x-one-of must be fully present."""
        spec = self.tools.get(name) or {}
        groups = (spec.get("parameters") or {}).get("x-one-of") or []
        if not groups:
            return None
        for group in groups:
            if all(arguments.get(k) not in _EMPTY for k in group):
                return None
        readable = " / ".join(" + ".join(g) for g in groups)
        return f"cần một trong: {readable}"

    # -- the whole pipeline -----------------------------------------------------------------
    def check(self, name: str, arguments: dict, dispatched: set[tuple] | None = None) -> GuardResult:
        """Run every check in priority order and return the cleaned call.

        The order matters: the reason reported is the cause, not a downstream symptom.
        ``dispatched`` holds the (name, arguments) pairs already sent in this trajectory;
        an identical repeat is refused, because a repeated write corrupts the state
        (two messages instead of one) and gold almost never repeats a call — measured:
        2 exact duplicates across all 1,545 gold trajectories.
        """
        name = (name or "").strip()
        arguments = dict(arguments or {})

        error = self.check_installed(name)
        if error:
            return GuardResult(False, name, {}, error)

        cleaned, _, _ = self.check_argument_keys(name, arguments)
        normalized, _ = self.normalize_values(name, cleaned)
        coerced, _, _ = self.coerce_types(name, normalized)
        repaired = coerced != arguments

        if dispatched is not None:
            key = (name, _stable(coerced))
            if key in dispatched:
                return GuardResult(
                    False, name, coerced,
                    f"lệnh {name} với các tham số này đã được thực hiện rồi; "
                    f"không cần lặp lại",
                )

        # 2. schema validation, before the grounding check (Decision 7): a schema-invalid
        # call is rejected by the environment anyway, so reporting that is more useful.
        error = schema_error(self.split, name, coerced, self.allowed)
        if error:
            return GuardResult(False, name, coerced, error)

        error = self.check_grounded(name, coerced)
        if error:
            return GuardResult(False, name, coerced, error)

        error = self.check_one_of(name, coerced)
        if error:
            return GuardResult(False, name, coerced, error)

        return GuardResult(True, name, coerced, None, repaired)


def _stable(arguments: dict) -> str:
    """A stable key for an argument dict, so two equal calls compare equal."""
    import json

    return json.dumps(arguments, sort_keys=True, ensure_ascii=False)


def collect_ids(value, into: set[str]) -> set[str]:
    """Collect every string in a tool result that looks like an entity id.

    The ids observed in a result are what a later x-ref argument may legitimately use.
    """
    if isinstance(value, dict):
        for item in value.values():
            collect_ids(item, into)
    elif isinstance(value, list):
        for item in value:
            collect_ids(item, into)
    elif isinstance(value, str):
        if value and len(value) <= 64 and " " not in value:
            into.add(nfc(value))
    return into
