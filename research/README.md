# Research scripts

Measurement scripts behind every number quoted in [`../docs/PRD.md`](../docs/PRD.md) and
[`../docs/VERIFIED_FACTS.md`](../docs/VERIFIED_FACTS.md). They are read-only with respect to the
dataset and reproducible against the competition kit.

## Prerequisites

The competition kit must be present at `data/train+public_test/` (train + public_test splits with
their `zaic_tools` runtime and `<split>_environment/` state). It is **not** part of this repository.

```bash
python -m pip install numpy        # everything else is stdlib
```

## Scripts

| Script | What it establishes |
|---|---|
| `scorer.py` | **The instrument.** Re-implements the official metric (state gate → maximum bipartite matching → per-argument fuzzy/exact comparison → F1) and proves it: gold replay scores **1.000000** with **1,545/1,545** perfect, and reproduces `label.final_state` on all 1,545 samples. Also produces the error-cost table (required argument, optional argument, wrong tool, type errors, extra call, order, fuzzy perturbation). **Every metric number in the PRD comes from here.** |
| `strategy.py` | Decomposes the score: out-of-scope share, the read-only population where the state gate is free, the write population where it is not; `ask_missing_info` placement; x-ref ID grounding rate. |
| `analyze.py` | Dataset statistics: catalog sizes, tool kinds, fuzzy/x-ref/x-one-of coverage, argument types, gold call counts, history and context shape, audio duration distribution. |
| `verify_metric.py` | Independent first pass at the metric, including what the state hash actually depends on (free-text keys and live apps are excluded from it). |
| `verify_metric2.py` | Order sensitivity of the state hash and the distribution of write-heavy trajectories. |
| `final_checks.py` | Confirms `label.final_state` is the gold state (so the local scorer needs no secret) and that an empty submission scores exactly the out-of-scope rate. |
| `test_notebook.py` | Local tests for the Colab launcher: static checks (no machine-local paths, valid magics, list-argv subprocess calls), execution of the data-setup logic against **both** possible zip layouts, the missing-zip error path, and CLI consistency between the notebook and PRD §6. |
| `make_notebook.py` | Generates `../colab_launcher.ipynb` from source, so the notebook is reviewable as code rather than as JSON. |

## Portal snapshots

`description.md`, `rule.md`, `evaluationMetric.md`, `evaluation_full.md`, `submissionGuidance.md` are
text renderings of the competition portal's four tabs, fetched from the public config API on
1 November 2026. The raw API payloads are kept alongside as `.json`. `metric_0.png` is the official
formula image from the Evaluation tab. `leaderboard.json` / `lb_full.json` are leaderboard snapshots
(357 teams as of 1 Nov 2026). `qwen35_index.json` is the Hugging Face weight index used to confirm
Qwen3.5-4B's exact parameter count.

To re-fetch the portal tabs:

```bash
for k in description evaluationMetric rule submissionGuidance; do
  curl -s -H "Origin: https://challenge.zalo.ai" -H "Referer: https://challenge.zalo.ai/" \
    "https://api-challenge.zalo.ai/fe/api/config-detail?name=$k&title=1" -o $k.json
done
curl -s -H "Origin: https://challenge.zalo.ai" -H "Referer: https://challenge.zalo.ai/" \
  "https://api-challenge.zalo.ai/fe/api/leaderboard?from=0&count=1000&title=1" -o lb_full.json
```

## Regenerating the notebook

```bash
python research/make_notebook.py colab_launcher.ipynb
python research/test_notebook.py
```
