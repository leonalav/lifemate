# LifeMate — Product Requirements Document (Technical)

**Zalo AI Challenge 2026, LifeMate track · Author: build team · Date: 1 Nov 2026 · Status: plan of record**
**Deadline binding this plan: final Docker solution due 16 Nov 2026, 23:59 GMT+7 (15 days).**

This PRD is the build contract. Anything numbered here (entry points, CLI flags, file paths, gates) is normative:
later build steps, including the Colab launcher, must conform to it.

---

## 1. Objective

Ship an offline voice-to-action agent that converts a spoken Vietnamese request into an ordered list of
tool calls that (a) reproduces the exact final app state the user asked for and (b) matches the reference
call sequence under the official metric.

**Award-winning bar.** The public field on 1 Nov 2026 is 357 teams: top score **99.37**, median **48.88**,
41 teams at or above 90. Rank is decided on the private test, not the public board. The build target is:

- Milestone 1 — reproducible local score **above the public median** on a faithful leave-apps-out split.
- Milestone 2 — public submission that confirms the local instrument predicts the board.
- Milestone 3 — a private-test-frozen Docker image with headroom under every compliance cap.

---

## 2. Verified competition facts (with source and date)

All facts below were re-verified on 1 Nov 2026 against the live portal API, the released kit v3.2.0,
and the public leaderboard. Full evidence lives in `docs/VERIFIED_FACTS.md` and `research/`.

| # | Fact | Source |
|---|---|---|
| F1 | Two rounds. Public test 26 Oct–16 Nov (≤5 submissions/day). Final solution = Docker image (train code + inference code + data), Drive link + checksum, ≤2 submissions, last counts, deadline 16 Nov 23:59 GMT+7. Private test released 20 Nov 09:00, results due 23 Nov 23:59. Top 2 announced 2 Dec. | Portal `submissionGuidance`, fetched 1 Nov 2026 |
| F2 | Model caps: **whole system < 6B params, LLM < 5B, context ≤ 16,384 tokens.** Hardware: 1× A30, 16-core Xeon Gold 6442Y, 64 GB RAM. **30 s per sample**, timeout/failure = 0. Fully offline at inference. | Portal `rule` |
| F3 | **Must not read `state.json` or anything under `<split>_environment/`. Exactly one `ToolEnv` per sample; every call made on it must appear in the submission.** Organisers mount private test at `/data` and collect `result/submission.jsonl`. Organisers replay submitted `tool_calls` in order with their `zaic_tools`; their result is final. | Portal `rule` + kit README |
| F4 | Only public data and open-source models **released before 26 Oct 2026** with permitting licences. Synthetic data may be generated with other services/models during development but **not used at inference**. Labelling `public_test`/`private_test` (manual or model) to train or tune is **forbidden**. Reproducibility: fixed seeds, deterministic decoding; divergence → organisers run the image 5× and average. Tie-break: re-score with F1 ignoring `state_hash`, then lower mean latency. | Portal `rule` |
| F5 | **Metric.** `state_hash` must equal gold, else the sample scores **0**. Tool-call match: same tool name, every non-fuzzy argument identical, every fuzzy argument token-F1 **> 0.6** (lowercase + Unicode NFC); one call matches at most one call; order ignored for matching. Gold empty → 1 iff submitted list is also empty. `sys.ask_missing_info` is scored like any other tool. `P = M/(M+U)`, `R = M/|G|`, `F1 = 2PR/(P+R)`, `S = 100/N · ΣSᵢ`. `U` includes schema-rejected calls. | Portal `evaluationMetric` (text + formula) |
| F6 | Train: **1,545** labelled samples, 5,036 audio utterances, ~6.2 h of 16 kHz mono speech (mean 4.40 s, median 3.41 s, p99 12.77 s). Public test: **634** samples, audio + context + history only (no transcripts, no labels). Kit `VERSION`: release v3.2.0, built 24 Oct 2026. | Kit, measured 1 Nov 2026 |
| F7 | Train catalog: **46 apps / 147 tools**. Public-test catalog: **36 apps / 122 tools**. Installed apps per sample: mean 9.2, max 18; installed tools per sample: mean **32.5**, median 31, max 62. Tool kinds in train catalog: read 114, write 153, meta 2. `x-match: fuzzy` on 32 tools; `x-ref` on 65 tools; `x-one-of` on 6 tools. | Kit, measured |
| F8 | **11 public_test apps never appear in train**: `ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks, train_bus_tickets, translator`. Private test will very likely contain unseen apps. Leave-apps-out validation is therefore the honest generalisation proxy. | Kit, measured |
| F9 | Gold structure: empty (out-of-scope) **7.2%**; max **6** calls per sample; multi-call **68.9%**. `sys.ask_missing_info` appears in **14.0%** of samples, **always as the last call (217/217)**, always with exactly `{tool_name, params}`. **27.3%** of gold calls carry no arguments. Gold is schema-perfect: 0/3,541 calls with out-of-schema args, 0 missing a required arg. | Kit, measured |
| F10 | **57.9%** of gold sequences need an ID from a prior read (x-ref args); 1,243/1,278 x-ref values appear verbatim in an earlier `tool_results`. ID hallucination = wrong state = 0 for the sample. | Kit, measured |
| F11 | **51.8%** of transcripts contain Vietnamese number words ("hai triệu tám", "một nửa", "30 phần trăm") while gold numeric args are typed integers (banking/ewallet `amount`, climate `temperature`, light `brightness`, …). Spoken-form → typed-number normalisation is mandatory, not cosmetic. | Kit, measured |
| F12 | State-hash canonicalisation is **split-specific**: `FREE_TEXT_KEYS` and `LIVE_APPS` differ between the train and public_test `zaic_tools` (e.g. `pickup`/`expression` dropped from the hash in train but not in public test; `text` dropped only in public test; `exchange_rate`/`lottery` live only in train). Consequence: always compute `state_hash` with the split's own runtime at submission time; never copy a hash implementation into our code. | Kit source diff, 1 Nov 2026 |
| F13 | **Metric instrument verified.** Replaying every gold call through the real `ToolEnv` reproduces `label.final_state` on **1,545/1,545** train samples and scores exactly **1.000000** (perfect on all 1,545) under our re-implementation of the official formula — including per-tool fuzzy sets, exact argument key sets, type-sensitive equality, and maximum bipartite matching. The local scorer needs no secret: labels carry `final_state`, so the exact score is computable offline. An empty submission scores **7.18/100**, exactly the out-of-scope rate. | Measured 1 Nov 2026 |
| F13b | **Cost of each error class** (n=400, state gate on; "gate-killed" = samples scoring 0 because the hash moved): control 1.000 (0 killed) · drop one **required** arg → **0.069** (355 killed) · perturb one **non-fuzzy** arg → **0.213** (268 killed) · drop last call → **0.192** (292 killed) · drop one **optional** arg → **0.511** (164 killed) · wrong first tool name → **0.596** (50 killed) · int passed as string → **0.773** (91 killed) · add one extra meta call → **0.818** (0 killed) · reverse call order → **0.928** (29 killed) · perturb one **fuzzy** arg → **1.000** (0 killed). | Measured 1 Nov 2026 |
| F14 | Prompt budget fits 16K. Installed-tool schemas ≈ 3.1k tokens mean / 6.0k max; full prompt (schemas + context + history + transcript) ≈ **3.9k tokens mean / 7.4k max**. Full per-sample schemas are affordable — no tool retrieval is structurally required. | Measured |
| F15 | Model facts (Hugging Face API, 1 Nov 2026). `Qwen/Qwen3.5-4B`: 4,659,865,088 params, Apache-2.0, 2026-02-27 (pre-cutoff), **including its vision tower — still under the 5B LLM cap** (this resolves param-counting design risk without needing an organiser ruling). `Qwen/Qwen3-4B-Instruct-2507`: 4,022,468,096 params, Apache-2.0, text-only. `g-group-ai-lab/gipformer1.5-68M-rnnt`: MIT, 2026-08-21, ships `encoder/decoder/joiner` ONNX (+ int8), `tokens.txt`, `bpe.model`, `model.pt`. Recommended stack totals **≈ 4.73B < 6B**. | HF API |
| F16 | Real-time budget. A reference team measured **10.91 s/sample** on public inference against the 30 s cap. gipformer transcribes at ~10× real time on CPU (0.15–0.3 s per utterance). The plan must budget transcription of up to 24 history turns plus the final request. | Competitor measurement + local check |
| F17 | Competitor evidence (public repos, used as measurement only). Baseline Qwen3-4B-Q4 heuristic: public **7.85**; hard/soft retrieval variants scored lower on the board despite higher local proxies — local instruments that ignore the state gate and the exact-arg-set rule mislead. Measured error taxonomy of a weak baseline: **WRONG_TOOL 70.3%**, MISSING_INFO 17.6%, ASR_SLOT 5.5%, WRONG_ARGUMENT 4.5%. Even with the exact first *two* gold calls injected, a 4B model still missed final state on 29/30 probes: single-step imitation is insufficient; the multi-step policy must be trained. | Competitor repos, Oct 2026 |

---

## 3. Strategy: what actually moves the score

The metric (F5) plus the data (F9–F13) dictate the strategy. No part of this section is a guess; each
rule traces to a numbered fact.

1. **Maximise P(call set == gold set). There is no safe partial credit on write samples.**
   On the 75.9% of samples whose gold performs a write, any missing, extra, or wrong write changes
   the hash → the whole sample scores 0. Hedging with extra calls is self-harm: one extra call costs
   ~18 F1 points (F13b) and a state-changing extra call costs the sample. Dropping a **required**
   argument is the single most destructive error (mean 0.069 — 355/400 samples killed by the gate),
   which is why validation must run *before* dispatch, not after.
2. **Tool selection is the bottleneck (70.3% of baseline errors).** Everything in the plan is ranked by
   whether it fixes tool choice: installed-app-constrained prompting, next-action SFT on real
   trajectories, rejection-sampled fine-tuning against the verified scorer, and a deterministic guard.
3. **Exact argument key sets, and no invented optional arguments.** The matcher compares the full
   argument set, so adding an optional argument the user never stated breaks the call match; dropping
   one costs ~49 F1 points (F13b) and 41% of samples die on the gate. 27.3% of gold calls have zero
   arguments (F9). A fuzzy argument perturbed beyond token-F1 0.6 still scores, and a fuzzy argument
   never touches the hash (F13b) — so free text is a *precision-only* concern, while every typed
   argument is a *gate* concern. That asymmetry sets the priority order: get typed arguments exactly
   right first, then polish wording.
4. **Read-before-write for IDs.** 57.9% of sequences need a grounded ID (F10); x-ref args must come from
   an observed tool result. The guard enforces this and blocks ungrounded writes.
5. **Reads are cheap, writes are decisive.** Reads never change state; a speculative read costs only
   precision, while a speculative write costs the sample. The guard therefore allows cheap
   information reads but gates every write on schema validity + ID grounding.
6. **Vietnamese numbers and dates are decided deterministically, never by the model.**
   A spoken-form→typed normaliser handles number words, percentages, fractions, money units
   (nghìn/triệu/đồng), and a relative-day table (`hôm nay …`, weekday-anchored) removes all date
   arithmetic from the model (F11). This matters because a single perturbed non-fuzzy argument costs
   ~79 F1 points on average (F13b).
7. **Fuzzy fields copy the user, never paraphrase.** 32 tools carry free-text fuzzy args matched at
   token-F1 > 0.6 (lowercase + NFC only — diacritics are significant, and a wrong tone drops the score
   to 0 on that argument); the instruction is to reuse the user's exact wording.
8. **`ask_missing_info` policy is learned, and stopping is part of it.** It appears in 14.0% of gold,
   always last with `{tool_name, params}` — the agent must stop after emitting it (F9).
9. **Metric fidelity is the deliverable before the model.** The local scorer is proven exact on gold
   (F13); every later number is trusted only through it. Local validation that ignores the gate or the
   key-set rule is banned by this PRD.
10. **Design for unseen apps.** 11 public-test apps are absent from train (F8); private test will add
    more. Full per-sample schemas (F14) plus app-agnostic policy wording plus leave-apps-out
    validation is how we avoid fitting the 46 train apps.

---

## 4. System architecture

```
audio (+ history audio) ──▶ ASR (gipformer, CPU) ──▶ transcript (+ normalised numbers/dates)
                                                            │
context + installed schemas + history (text+calls+results) ─┴─▶ PromptBuilder
                                                            │
                        Prompt ──▶ LLM policy (Qwen3.5-4B + LoRA, vLLM, greedy, schema-constrained JSON)
                                                            │  one action per step: {"name","arguments"} | {"name":"finish"}
                                                            ▼
                        Guard (deterministic): installed? validate()? coerce types? ground x-ref?
                        block duplicate reads? ──▶ session.call(name, args) ──▶ compacted observation ──▶ loop
                                                            │
                        stop: finish | after ask_missing_info | max_steps | time budget
                                                            ▼
                        submission line: {"id", session.get_tool_calls(), session.get_state_hash()}
```

### 4.1 ASR — `g-group-ai-lab/gipformer1.5-68M-rnnt`, stock, via sherpa-onnx on CPU

- Why: 68M params, MIT, pre-cutoff, ONNX (+ int8) published upstream — no export step. Measured by a
  competitor at 16.1% WER stock (15.5% spoken form) on 878 held-out real-voice utterances, CPU,
  ~10× real time. Fine-tuning reaches ~10.1% but costs a k2/icefall setup for ~5 WER points that a
  competitor showed do not reliably transfer downstream.
- Decision: **ship stock gipformer first**; fine-tune is an explicit stretch goal, gated on E0 (the
  diagnostic that separates ASR-caused from policy-caused errors). Budget impact: 0.068B params, no
  A30 VRAM contention (ASR runs on CPU, LLM loads after).
- Determinism: greedy decoding, fixed seed; transcribe once per audio path and cache by path
  (`TranscriptStore`) so re-runs are identical.
- Post-processing (deterministic, tested): NFC normalisation, sentence-case recasing that preserves
  numeric forms (`200.000` is not a sentence end), Vietnamese number-word → typed-number mapping,
  money-unit expansion. Never fold diacritics on values that reach the model: the matcher is
  diacritic-sensitive.

### 4.2 LLM policy — Qwen3.5-4B (+ LoRA), served by vLLM, one action per step

- Why this model: 4.659B total params **including** the (unused) vision tower — inside the 5B LLM cap
  under any counting convention, and 4.73B system total with gipformer — inside the 6B cap (F15).
  Apache-2.0, released 2026-02-27 (pre-cutoff).
- Fallback, kept warm in parallel: `Qwen/Qwen3-4B-Instruct-2507` (4.02B, text-only, 4.09B system
  total): weaker policy, zero param-count ambiguity. Promote it only if Qwen3.5-4B underperforms on
  the leave-apps-out split by more than the audit headroom is worth.
- Serving: vLLM with `max_model_len=16384` (hard cap per F2), temperature 0, fixed seed,
  structured output constrained to an `anyOf` schema over the sample's installed tools plus `finish`.
  Native function calling is not required; the wire format is our choice, the submission format is fixed.
- Prompt: Vietnamese system policy encoding the labelling rules (§3); per-sample installed tool
  schemas in full (affordable per F14); relative-day table; compacted history (text, calls, trimmed
  results); few-shot examples retrieved per app-pattern. One model step emits exactly one action, so
  every dispatched call is independently attributable in the journal and independently fixable.
- Latency ledger (30 s cap): model rounds are the cost centre; measured reference is 10.91 s/sample.
  `max_steps = 8` and a 25 s per-sample time guard leave margin (gold never exceeds 6 calls).

### 4.3 Guard — deterministic code between model and environment (no params, no learning)

Order of checks (order matters: report the cause, not the consequence):

1. Tool belongs to an installed app (`context.installed_apps`, plus `sys.ask_missing_info`).
2. `session.validate()` (unjournaled — a rejected proposal costs nothing against F1) before any dispatch.
3. Argument key-set policy: drop unknown keys; drop empty optional values; never invent optional args.
4. Bidirectional type coercion toward the declared schema type (`"26"`→`26` for integer, `26`→`"26"`
   for string; `"2.000.000"`→`2000000`); date-time normalisation to `YYYY-MM-DDTHH:MM:SS+07:00`;
   enum/range/format repair where the schema makes the fix unambiguous.
5. `x-ref` grounding: every identifier argument must have appeared in a prior observation (history
   results or this session's results). Ungrounded writes are blocked, never dispatched.
6. Duplicate-read suppression: an identical read returns the cached observation without touching the
   session again (precision protection).
7. `x-one-of` group presence stated back to the model when missing.

### 4.4 Environment and scorer — the kit's code, never ours

- Exactly one session per sample, created once, from the split's own `zaic_tools` loaded under a
  unique module name when two splits share a process (each split ships its own version, F12).
- Never read, copy, or parse anything under `<split>_environment/`; app data is visible only through
  tool results. `get_state()` is used only in offline diagnostics on train, never at inference.
- The offline scorer replays with the split's `zaic_tools`, applies the state gate, matches with
  maximum bipartite assignment (Kuhn), token-F1 > 0.6 after lowercase + NFC, type-sensitive exact
  equality otherwise, exact argument key sets. Proven: gold-vs-gold = 100.00 on all 1,545 train
  samples (F13). Submitted-row validation (id uniqueness, manifest order, schema validity of every
  proposed call) is a separate pre-flight step before any upload.

---

## 5. Training plan (all on Colab GPUs — no training on this device)

### 5.1 Data: next-action trajectories with real observations (teacher-forced SFT)

For each train sample, unroll the gold trajectory against a live `ToolEnv`: at step *t* the input is
(prompt, history of gold actions + real observations so far), the target is gold action *t+1* (or
`finish`). Multi-call samples yield multiple examples (~3.5k action examples from 1,545 samples).
Transcripts used in training are the **stock-gipformer outputs** (test-time distribution), with a
gold-transcript mix for stability. ASR-fine-tuned transcripts are deliberately *not* used for LLM
training (a fine-tuned ASR memorises train audio and hides real error modes from the policy).

### 5.2 Method: QLoRA SFT, then rejection-sampled fine-tuning (RFT)

- Phase A — QLoRA SFT (rank 64) on the trajectory data, `max_seq_len` ≤ 16,384, on a single Colab
  GPU. Validation is leave-apps-out (hold out ~10 apps / ~270 samples mirroring the public-test
  unseen-app structure) plus a seed split; near-duplicate transcripts (char similarity ≥ 0.92) are
  grouped so validation is honest.
- Phase B — rejection sampling against the verified scorer: sample rollouts from the Phase-A policy
  on train, keep trajectories scoring exactly 1.0, fine-tune on those (on-policy correction for the
  multi-step drift the competitor's oracle study proved: exact first two calls still missed state on
  29/30). Optional Phase C — preference optimisation on kept-vs-rejected pairs.
- ASR fine-tune (icefall + k2, ONNX export) is scheduled **only** if diagnostic E0 shows speech in
  the dominant error path. Default: stock gipformer.

### 5.3 Evaluation protocol (binding)

- Splits: leave-apps-out holdout + seed split + near-dup grouping; the holdout apps approximate the
  11 unseen public-test apps.
- Metrics per run: exact-score (gate × F1), tool-call F1, state-match rate, and the error taxonomy
  (WRONG_TOOL / MISSING_INFO / ASR_SLOT / WRONG_ARGUMENT / OUT_OF_SCOPE). Every change is judged on
  the taxonomy delta, never on a single scalar.
- Diagnostics: E0 reference-vs-ASR transcript ablation (isolates speech), oracle-first-call probes
  (isolates tool selection), gold-replay = 99.99+ (isolates pipeline bugs from model quality).
- Submission discipline: ≤5 public uploads/day, only after a locally justified change; the board is a
  measurement, never a target.

---

## 6. Repo layout and CLI contract (normative)

The repository (`github.com/leonalav/lifemate`, branch `main`) will contain exactly this surface.
The Colab launcher and the Docker image both call these entry points; flags are frozen by this PRD.

```
lifemate/
├── colab_launcher.ipynb          # GPU training + inference launcher (conforms to §6 flags)
├── configs/
│   ├── agent.yaml                # max_steps, time budget, decoding, context caps
│   └── sft_qlora.yaml            # rank, lr schedule, seq len, mix ratios
├── requirements.txt              # no torch/torchvision/jax pins (Colab provides); see §7
├── .gitignore                    # data/, *.pt, *.pth, *.ckpt, runs/, wandb/, .env, secrets
├── src/lifemate/                 # asr/ agent/ evaluation/ scripts per §4; clean-room, no copied code
├── scripts/
│   ├── train_sft.sh              # Phase A/B driver
│   ├── predict_public.sh         # public_test inference → submission.jsonl + validation
│   └── serve.sh                  # local vLLM server for dev
├── docker/
│   ├── Dockerfile                # offline image: code + merged model + gipformer ONNX + sherpa-onnx
│   └── entrypoint.sh             # discovers split under /data, runs inference, writes result/submission.jsonl
└── docs/                         # PRD.md, PRD_NONTECHNICAL_VI.md, VERIFIED_FACTS.md, DECISIONS.md
```

CLI (frozen):

```bash
python -m lifemate train   --data-dir DIR --output-dir DIR --config configs/sft_qlora.yaml \
                           --resume [PATH] --epochs N --batch-size N [--smoke]
python -m lifemate merge   --checkpoint DIR --output-dir DIR [--model-path BASE]
python -m lifemate infer   --data-dir DIR --split NAME --checkpoint DIR --output-file FILE \
                           [--limit N] [--time-budget-s 25] [--max-steps 8]
python -m lifemate score   --split train --submission FILE [--ids-file FILE] [--report FILE]
python -m lifemate validate --split NAME --submission FILE
python -m lifemate transcribe --split NAME --data-dir DIR --out FILE
```

- `train` saves a `last` checkpoint every epoch and a `best` checkpoint on leave-apps-out improvement;
  `--resume` restores model, optimiser, scheduler, and epoch. Checkpoints, logs, and inference
  outputs live on Drive (Colab) or the mounted volume (Docker); dataset and repo live on local disk.
- `infer` auto-discovers the split directory under `--data-dir` (private test arrives as a new folder;
  never hardcode split names), loads the split's own `zaic_tools`, transcribes all user turns once,
  runs the agent loop, and writes UTF-8 JSONL with `{"id","tool_calls","state_hash"}` in manifest order.
- `merge` merges the LoRA adapter into the base model, because vLLM serves a merged model and the
  Docker image bakes the result in.
- `validate` is a pre-flight check: every sample present exactly once, every proposed call
  schema-valid against the split's catalog and the sample's installed apps.
- Docker `entrypoint.sh` calls `infer --data-dir /data --checkpoint /models/final
  --output-file /app/result/submission.jsonl` with `--network none`; model files are baked into the
  image (no downloads at runtime).

---

## 7. Dependency and secret rules

- `requirements.txt` lists only what Colab/Docker lack: `vllm`, `sherpa-onnx`, `peft`, `trl`,
  `transformers`, `datasets`, `nbformat` (launcher validation), test tooling. **Never pin**
  `torch`/`torchvision`/`tensorflow`/`jax`; never reinstall a Colab-provided core package without a
  runtime restart note.
- `.gitignore` excludes `data/`, `*.pt`, `*.pth`, `*.ckpt`, `runs/`, `wandb/`, `.env`, checkpoints,
  and model dirs. The ~800 MB dataset is never committed; it travels Drive → Colab local disk.
- Secrets (`HF_TOKEN`, `WANDB_API_KEY`) load from Colab `userdata` into `os.environ` inside
  try/except; they are never hardcoded, never printed, never committed. No API keys or tokens in the repo.

---

## 8. Colab execution plan (the GPU strategy, since local training is forbidden)

One notebook, `colab_launcher.ipynb`, at repo root; one code cell per step; every cell idempotent:

1. Markdown: set Runtime → GPU (L4/A100 preferred; T4 minimum for QLoRA-4B).
2. GPU check: `!nvidia-smi` + `torch.cuda.is_available()` with a hard error if no GPU.
3. Mount Drive at `/content/drive`.
4. **Config cell (single source of truth):** `REPO_URL`, `BRANCH`, `DRIVE_DATA_ZIP`,
   `RUN_NAME`, `OUTPUT_DIR` (Drive), `EPOCHS`, `BATCH_SIZE`, LoRA rank, seq len, checkpoint path.
   All later cells reference these variables via `{var}`.
5. Secrets cell (userdata → `os.environ`, try/except, never echo).
6. Clone-or-update cell (safe re-run: clone if absent, else fetch/checkout/pull).
7. `pip install -q -r requirements.txt` + restart-runtime note.
8. Data cell: copy zip Drive → local, unzip to `/content/data_local`, delete local zip, skip if
   present; auto-handle single-top-level-`data/` vs flat layout; print contents + file count; fail
   loudly if the Drive zip is missing.
9. Smoke cell: 1-epoch / few-step CPU-GPU dry run on a tiny subset (gates the long run).
10. Train cell: `train` with `--data-dir /content/data_local`, `--output-dir` on Drive,
    `--resume` from last checkpoint (disconnect-safe).
11. Monitoring cell: TensorBoard on the Drive log dir, or W&B if enabled.
12. Inference cell: `infer` on public_test with checkpoint + input paths from the config cell;
    outputs to Drive; display a sample row inline.
13. Troubleshooting markdown (OOM → batch size/mixed precision/restart; disconnect → re-run from
    top, resume continues; stale code → re-run clone cell; missing Drive file).

The notebook JSON must validate with `nbformat.read` + `nbformat.validate`, ship with cleared
outputs, and undergo a CPU dry-run of the `train`/`infer` CLIs on a tiny sample before any long run.
README carries an "Open in Colab" badge linking to the notebook on `main`.

---

## 9. Milestones (15 days to the Docker freeze)

| By | Milestone | Exit criterion |
|---|---|---|
| 3 Nov | Metric + replay scorer + submission validator green | Gold-vs-gold = 100.00; empty submission scores exactly the out-of-scope rate |
| 5 Nov | Baseline agent (stock gipformer + base Qwen3.5-4B + guard) on leave-apps-out | Taxonomy E0 measured; local score above public median on the holdout |
| 8 Nov | Phase-A QLoRA SFT done; first public submission | Board confirms the local instrument predicts rank direction |
| 12 Nov | Phase-B RFT done; ablations (policy wording, few-shot, guard rules) closed | Best checkpoint frozen; public submission #2 max |
| 14 Nov | Docker image built, offline-verified (`--network none`), latency < 30 s/sample mean | `result/submission.jsonl` reproduces the public submission byte-for-byte modulo ordering |
| 16 Nov | Final Docker submitted (Drive link + checksum) before 23:59 GMT+7 | Submission receipt confirmed |
| 20–23 Nov | Private test: run frozen image, submit JSONL | No code changes after freeze except organiser-mandated fixes |
| 2 Dec | Results | Top-2 target judged here, not on the public board |

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| Private-test apps unseen in both train and public_test | App-agnostic policy, full per-sample schemas, leave-apps-out tuning; no per-app special cases in prompts |
| Split-specific hash canonicalisation (F12) | Never reimplement the hash; always the split's own runtime; pre-flight validates hashes recompute identically |
| 30 s/sample incl. up-to-24-turn transcription | Transcribe once + cache; int8 ONNX on CPU; 25 s internal budget; skip already-cached paths |
| Colab disconnects mid-training | `--resume` + `last`/`best` checkpoints on Drive; smoke cell gates long runs |
| k2/icefall build pain for ASR fine-tune | Defer ASR training; stock gipformer is the plan of record |
| Local-proxy vs board divergence (observed by competitors) | Board submissions only after locally justified changes; cap at allowance |

Questions already drafted for the organisers (timing boundary incl. startup, `result/` working dir,
private-`zaic_tools` injection, lifecycle-method journaling, vision-tower counting — now moot per
F15 — loopback policy, cloud-GPU data policy) are in `VERIFIED_FACTS.md` §10; send the unresolved ones.

## 11. Definition of done

- [ ] Offline scorer reproduces gold = 100.00 and the empty-submission rate exactly.
- [ ] Leave-apps-out local score justifies every submission; taxonomy improves monotonically.
- [ ] Docker runs fully offline on A30-class hardware, mean < 30 s/sample, deterministic re-runs.
- [ ] Param audit: system ≈ 4.73B, LLM = 4.66B (incl. vision tower), context capped at 16,384.
- [ ] No compliance violation: one session per sample, no environment reads, no test labelling,
      all models/secrets pre-cutoff and clean-room.
- [ ] Final Docker submitted by 16 Nov 23:59 GMT+7 with checksum; frozen through 23 Nov.
