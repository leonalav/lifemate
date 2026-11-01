# LifeMate

Solution for the **LifeMate** track of the [Zalo AI Challenge 2026](https://challenge.zalo.ai/portal/lifemate).

An offline, voice-driven agent that turns a spoken Vietnamese request into an ordered list of tool calls across
smart-home, banking, calendar, messaging, ride-hailing and 40+ other apps — reproducing the exact final app state
the user asked for, under the competition's official metric.

## Status

| Item | State |
|---|---|
| Rules, metric and constraints | **Verified** against the live portal API, 1 Nov 2026 |
| Local scorer | **Exact** — gold replay scores 1.000000 on 1,545/1,545 train samples |
| Dataset analysis | **Complete** — structure, distributions and error costs measured |
| Build plan | [`docs/PRD.md`](docs/PRD.md) (technical) · [`docs/PRD_NONTECHNICAL_VI.md`](docs/PRD_NONTECHNICAL_VI.md) (Vietnamese, non-technical) |
| Verified facts | [`docs/VERIFIED_FACTS.md`](docs/VERIFIED_FACTS.md) |
| Design decisions | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Implementation | Begins after plan approval — Docker freeze **16 Nov 2026, 23:59 GMT+7** |

## Running on Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonalav/lifemate/blob/main/colab_launcher.ipynb)

Training and inference run on Colab GPUs — **no model training or inference runs on a local machine.**

1. Open the notebook: [`colab_launcher.ipynb`](colab_launcher.ipynb) — or use the badge above.
2. **Runtime → Change runtime type → GPU** (T4 minimum; L4 or A100 preferred).
3. Put the dataset at `MyDrive/data_lifemate/data.zip` (~800 MB). It is never committed to this repository.
4. Add optional secrets in the Colab sidebar (key icon): `HF_TOKEN`, `WANDB_API_KEY`. Missing secrets are ignored.
5. Run the cells from the top. The clone cell updates the repo, the data cell skips if already prepared, and the
   training cell resumes from the last checkpoint if a run was interrupted.

Checkpoints, logs and inference outputs are written to Drive under
`MyDrive/lifemate_runs/<RUN_NAME>/`; the repository and the dataset stay on the session's local disk.

## Repository layout

```
colab_launcher.ipynb      # Colab GPU launcher (training + inference)
configs/                  # agent and SFT configuration
src/lifemate/             # package: asr/, agent/, evaluation/, cli
scripts/                  # train / predict / serve drivers
docker/                   # offline inference image for the final submission
docs/                     # PRD, verified facts, decision log
research/                 # measurement scripts used to produce the cited numbers
```

## Reproducing the research

```bash
python research/scorer.py        # metric fidelity: gold replay == 1.000000, error-cost table
python research/strategy.py      # score-gate decomposition, gold trajectory structure
python research/analyze.py      # dataset statistics
python research/test_notebook.py # local tests for the Colab launcher logic
```

The dataset lives outside this repository. Point the scripts at the kit directory before running them.

## Rules compliance summary

| Constraint | Limit | This solution |
|---|---|---|
| Total system parameters | < 6B | ~4.73B (Qwen3.5-4B 4.66B + gipformer 68M) |
| LLM parameters | < 5B | 4.66B, including the unused vision tower |
| Context length | ≤ 16,384 | capped at 16,384 |
| Latency | 30 s / sample | internal budget 25 s; reference measurement 10.91 s |
| Network at inference | none | fully offline, models baked into the image |
| Environment access | one `ToolEnv` per sample, no `state.json` reads | enforced by the guard |
| Model release date | before 26 Oct 2026 | 2026-02-27 (LLM), 2026-08-21 (ASR) |
| Test-set labelling | forbidden | only the labelled `train` split is used |

## License

Code in this repository is released under the MIT license. Third-party models are used under their own licenses:
Qwen3.5-4B (Apache-2.0), gipformer1.5-68M-rnnt (MIT).
