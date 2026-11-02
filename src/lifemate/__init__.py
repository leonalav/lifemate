"""LifeMate — offline voice-to-action agent for the Zalo AI Challenge 2026 LifeMate track.

Public surface:
    python -m lifemate train      --data-dir DIR --output-dir DIR --config FILE [...]
    python -m lifemate infer      --data-dir DIR --split NAME --checkpoint DIR --output-file FILE [...]
    python -m lifemate score      --split train --submission FILE [--report FILE]
    python -m lifemate transcribe --split NAME --data-dir DIR --out FILE

The CLI contract is frozen in docs/PRD.md §6.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
