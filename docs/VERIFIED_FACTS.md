# LifeMate — Verified Contract Facts

**Collected:** 29 October 2026 · **Sources:** public portal API (`api-challenge.zalo.ai`), two public competitor repositories
(`MrPickle200/zalo-ai-2026-lifemate`, `aihacker111/Zalo-AI-Challenges-LifeMate`), Hugging Face model metadata.

This document supersedes the speculative parts of `LifeMate_PRD_and_Build_Plan.md` where they conflict. The PRD was written
before the kit existed; the facts below were verified from live sources. **Anything not listed here is still unverified.**

---

## 1. What the public portal actually says

The portal is a JS SPA. Content is fetched per-tab from
`https://api-challenge.zalo.ai/fe/api/config-detail?name=<key>&title=<apiTitle>` with `Origin`/`Referer` headers set.

Tab-name mapping (extracted from the app bundle, not guessed):

| Tab | `name` param | Auth required |
|---|---|---|
| Description | `description` | no |
| Evaluation | `evaluationMetric` | no |
| Rule | `rule` | no |
| Submission Guidance | `submissionGuidance` | no |
| **Data** | **`dataExam`** | **yes** (`error_code: -2`, "Authen Failed") |

The Data tab is the registration wall. `news`, `quota`, `solution-list` and `submission-list` also require
authentication. **The public leaderboard does not** — see section 11.

### Scoring formula (verbatim from Evaluation tab)

- **State match gate:** submitted `state_hash` must equal gold `state_hash`, else sample scores **0**.
  Hash is produced by `env.get_state_hash()` and is reproducible by the organisers, who **replay the submitted `tool_calls`
  in order** with the matching `zaic_tools` version. Organiser result is final.
- **Tool-call match:** same tool name, every non-fuzzy argument identical, every fuzzy argument token-F1 **> 0.6**
  (lowercase + Unicode NFC). One call matches at most one call. **Order is not considered for matching.**
- **Out-of-scope:** gold list empty → score 1 only if submitted list is also empty.
- **`sys.ask_missing_info` is scored like any other tool.**
- `P = M/(M+U)`, `R = M/|G|`, `F1 = 2PR/(P+R)`, `S = 100/N · Σ Sᵢ`. `U` includes calls rejected by the schema check.

### Rule tab — the constraints that bind

- Model size: **whole system < 6B total, LLM < 5B, context ≤ 16,384 tokens.**
- **30 seconds per sample**; timeout or failure = 0. Hardware: 1× A30, 16-core Xeon Gold 6442Y, 64 GB RAM.
- **Must not read `state.json` directly.** All reads/writes through the execution environment.
- **Exactly one `ToolEnv` per sample; every call made on it must appear in the submission.**
- Must not read, copy or parse any file under `<split>_environment/`.
- Offline at inference. Public data and open-source models must be **released before 26 October 2026**.
- Synthetic data may be generated with other services during development, **but not used at inference**.
- Labelling `public_test` or `private_test` (manually or with external models) to train or tune is **not allowed**.
- Reproducibility: fixed seeds, deterministic decoding. Divergence → organisers run the image 5× and average.
- Tie-break: re-score with the F1 formula ignoring `state_hash`; then lower average processing time wins.
- Organisers mount private test at **`/data`**, collect **`result/submission.jsonl`**.

---

## 2. Kit layout and the `zaic_tools` API (G02, G03, G05 — closed)

The competition data package is a folder of splits. Each split is self-contained:

```
train+public_test/
├── train/
│   ├── train.jsonl                 # one JSON object per sample
│   ├── train_tools_catalog.json    # ALL tool schemas for this split
│   ├── train_environment/          # initial app state  ← never read directly
│   ├── audio/                      # .wav, 16 kHz mono
│   └── zaic_tools/                 # per-split Python package
└── public_test/
    ├── public_test.jsonl
    ├── public_test_tools_catalog.json
    ├── public_test_environment/
    ├── audio/
    └── zaic_tools/
```

**Each split ships its own `zaic_tools` version.** They are all imported as `zaic_tools` but must be loaded under unique
module names if two splits live in one process (`importlib.util.spec_from_file_location`).

### `ToolEnv` construction — two confirmed signatures

```python
# mrpickle: ToolEnv(split, sample_id, root=data_root)
from zaic_tools import ToolEnv
env = ToolEnv(split, sample_id, root=data_root)

# aihacker111: Executor(env_dir, catalog_path).open(sample_id)
from zaic_tools import Executor
session = Executor(str(env_dir), str(catalog_path)).open(sample_id)
```

`Executor` loads every split's initial state once; `open(sample_id)` returns a fresh session on that sample's initial
state. **Open exactly one session per sample.**

### Session API (the surface the agent is allowed to touch)

| Method | Purpose | Journaled? |
|---|---|---|
| `session.call(name, arguments)` → `{"ok": bool, ...}` | Execute a tool | **Yes** — every call is recorded, including schema-rejected ones |
| `session.validate(name, arguments)` → `{"ok": bool, "error": str\|None}` | Dry-run schema check | **No** — safe to call before dispatch |
| `session.get_tool_calls()` → `list[{name, arguments}]` | The exact submission payload | — |
| `session.get_state_hash()` → 64-char hex string | Final state hash | — |

`session.validate()` not being journaled is the single most useful fact for precision: it lets you reject a malformed
call without it counting against you.

### Sample record shape (`<split>.jsonl`)

```json
{
  "id": "train_0001",
  "audio": "audio/_u3.wav",
  "audio_text": "…gold transcript, train only…",
  "context": {
    "current_time": "2026-11-20T14:30:00+07:00",
    "timezone": "Asia/Ho_Chi_Minh",
    "location": {"address": "…"},
    "user_profile": {"home": "…", "work": "…"},
    "installed_apps": {"smart_home": {"tools": ["smart_home.get_status", "smart_home.climate_set"]}}
  },
  "history": [
    {"role": "user", "audio": "audio/_u1.wav", "audio_text": "…"},
    {"role": "assistant", "tool_calls": [{"name": "…", "arguments": {}}], "tool_results": [{…}], "text": "…"}
  ],
  "label": {"tool_calls": [{"name": "…", "arguments": {}}]}
}
```

`context.installed_apps` gives the **exact tool list for that sample** — the set of installed apps varies per sample.

### Tool catalog structure (`<split>_tools_catalog.json`)

```json
{"apps": [{"id": "smart_home", "tools": [{
  "name": "smart_home.climate_set",
  "app": "smart_home",
  "kind": "read" | "write" | "meta",
  "description": "…",
  "parameters": {"type": "object", "properties": {…}, "required": [...]},
  "returns": "…"
}]}]}
```

Non-standard JSON-Schema extensions, confirmed in use:

| Extension | Meaning | Use |
|---|---|---|
| `x-match: "fuzzy"` | This argument is fuzzy-matched at token-F1 > 0.6 | Only these are fuzzy; **all other arguments must be byte-identical** |
| `x-ref: "<tool>"` | This argument is an ID that must come from a prior tool result | Hallucinated IDs ⇒ wrong state ⇒ sample scores 0 |
| `x-one-of: [["a","b"],…]` | At least one argument from each group is required | Prompt must state it |

`kind` is supplied by the catalog, so read/write classification comes from the kit, not from name guessing.

---

## 3. The metric, implemented (G04 — closed)

Reconstructed scorer verified against `gold vs gold = 100.00` and against the published formula:

```python
Fuzzy threshold = 0.6 (strictly greater)
Normalisation   = NFC + lowercase
Tokens          = re.findall(r"[^\W_]+", normalised)      # letters/digits, diacritics KEPT
token_f1        = bag-of-tokens F1 (SQuAD style, Counter intersection)

def calls_match(pred, gold, fuzzy):
    if pred["name"] != gold["name"]:            return False
    if set(pred_args) != set(gold_args):        return False   # ← EXACT KEY SET
    for k, gv in gold_args.items():
        pv = pred_args[k]
        if k in fuzzy:
            if not (isinstance(pv,str) and isinstance(gv,str) and token_f1(pv,gv) > 0.6): return False
        else:
            if type(pv) is not type(gv) or pv != gv: return False   # type-sensitive
    return True
```

**Matching is maximum bipartite (Kuhn's algorithm)** — a greedy one-to-one assignment is not equivalent when calls are
ambiguous.

Two facts that the PRD did not capture and that dominate scoring strategy:

1. **The argument key set must match exactly.** Passing one extra or missing one argument fails the match entirely, even if
   every value is right. This makes "do not add optional arguments the user did not state" a hard requirement, not a style
   preference.
2. **Non-fuzzy arguments are type-sensitive.** `26` does not match `"26"`; `true` does not match `1`.

Also confirmed: **matching ignores order, but replay does not.** The organisers replay in submission order, so an
equivalent-effect but differently-ordered sequence can match while producing a different hash.

---

## 4. Real dataset scale (G01 — partially closed)

Observed by competitors who already hold the kit:

| Quantity | Value |
|---|---|
| Train samples | **1,545** (mrpickle split: 1,236 train / 309 val, seed 20261028) |
| Train audio utterances (incl. history turns) | **5,036** (4,158 train / 878 val under leave-apps-out) |
| Public test samples | **634** |
| Distinct apps | **~50** (not 5 — see below) |
| Out-of-scope samples | **~7.1%** train / 7.4% val |
| `sys.ask_missing_info` as first intent | **~4.0%** train / 3.6% val |
| Multi-call vs single-call | ~69% multi / 31% single (val) |
| Split grouping method | grouped greedy stratification over first intent, primary app, call count, missing-info, out-of-scope; near-duplicate transcripts grouped at char similarity ≥ 0.92 |

**The PRD's app list is badly understated.** The portal mentions five illustrative domains (smart home, banking,
ride-hailing, food delivery, email). The actual catalog spans **~50 apps**: `smart_home`, `bill_payment`, `weather`,
`chat_ott`, `contacts`, `reminders`, `calendar`, `alarm`, `maps`, `bike_rental`, `learning`, `home_security`,
`subscriptions`, `robot_vacuum`, `phone`, `gold_price`, `loyalty`, `door_lock`, `news`, `home_energy`, `exchange_rate`,
`podcast_radio`, `music`, `notes`, `tv`, `package_tracking`, `calculator`, `food_delivery`, `flights`, `library`,
`shopping_list`, `household_chores`, `email`, `medication`, `video_meeting`, `tasks`, `ewallet`, `fitness`,
`fuel_price`, `appliances`, `ride_hailing`, `lottery`, `home_automation`, `banking`, `sys`.

Two consequences:
- A "leave-apps-out" holdout is the right generalisation proxy, because **11 public_test apps do not appear in train** —
  so private test almost certainly contains unseen apps.
- The PRD's "40 apps" figure is wrong in both directions; plan for ~50 and design for unseen apps.

### Known label noise

`train_1696` has a wrong gold label (sends a message to `chat_6142`, an ID no tool ever returned). A strict guard will
correctly block that call and lose the sample. Treat it as noise, not as a bug to chase.

---

## 5. ASR: the PRD's recommendation is superseded (G-appendix)

The PRD proposed PhoWhisper-medium (769M) as the ASR baseline and PhoWhisper-large (1.55B) as challenger. Measured results
from a competitor on 878 held-out Vietnamese utterances (real voices, multiple regions):

| ASR | WER | WER (spoken form) | Params | Notes |
|---|---|---|---|---|
| **gipformer1.5-68M** (not fine-tuned) | 16.1% | 15.5% | **68M** | MIT, ONNX via sherpa-onnx |
| gipformer1.5-68M fine-tuned on train audio | — | **10.1%** | 68M | 6.8 h on 5,036 utterances |
| Qwen3-ASR-0.6B base | 36.1% | — | 938M | Apache-2.0 |
| Qwen3-ASR-0.6B fine-tuned | 27.0% | — | 938M | |
| Whisper Small (mrpickle) | — | — | ~244M | used by the weaker team |

**gipformer1.5-68M dominates on quality and cost simultaneously.** At 68M parameters it changes the budget arithmetic
completely:

- gipformer (0.068B) + Qwen3.5-4B (4.660B) = **4.73B total**, LLM 4.66B — comfortably under both caps.
- It removes the PRD's awkward "PhoWhisper-large pushes us to 5.57B" tension entirely.
- It runs on **CPU** via sherpa-onnx, so ASR does not contend for A30 VRAM with the LLM. The Docker entrypoint can
  transcribe every user turn first, free the ASR runtime, then load the LLM.

Model availability verified: `g-group-ai-lab/gipformer1.5-68M-rnnt`, MIT licence, created **2026-08-21** (pre-cutoff),
ships `encoder.onnx` / `decoder.onnx` / `joiner.onnx` (+ int8 variants), `tokens.txt`, `bpe.model`, `model.pt`.
Fine-tuning requires icefall + k2, then export to ONNX.

Note the parameter-count caveat: gipformer is 68M *by the published name*. Its exact tensor count must still be audited
before the compliance certificate (G07).

---

## 6. Measured baselines and the strategy that failed

### What actually scored on the public leaderboard

| Team | System | Public LB |
|---|---|---:|
| mrpickle | Ollama Qwen3-4B Q4_K_M (heuristic baseline) | **7.85** |
| mrpickle | + hard tool shortlist (top-3 TF-IDF, max 5 tools) | 6.66 |
| mrpickle | + soft retrieval (full schemas visible, demos as reference) | 7.51 |

### What the other team measured locally

| Configuration | Tool-call F1 | Final-state match |
|---|---:|---:|
| Whisper Small + Q4_K_M, baseline | 0.1155 | 0.2492 |
| + oracle transcripts | 0.2429 | 0.3301 |
| + soft retrieval | 0.2435 | 0.2977 |

**Two lessons, both expensive:**

1. **The local proxy did not predict the public score.** Soft retrieval beat baseline on *every* local metric (F1 0.2435 vs
   0.1155, state 29.8% vs 24.9%, 86 samples improved vs 32 regressed) and still **lost 0.34 points on the public
   leaderboard**. The hard-shortlist version did the same. Local validation that ignores the state-hash gate and the
   exact-argument-set rule will systematically mislead. The local scorer's fidelity to the official formula is the thing
   to invest in, not the split's statistical power.
2. **Baseline error taxonomy (290 non-perfect samples):** `WRONG_TOOL` 204 (70.3%), `MISSING_INFO_ERROR` 51 (17.6%),
   `ASR_SLOT_ERROR` 16 (5.5%), `WRONG_ARGUMENT` 13 (4.5%), `OUT_OF_SCOPE_ERROR` 6 (2.1%). The bottleneck is **tool
   selection**, not speech and not argument formatting.

### What demonstrably worked for the other team

- A deterministic **guard** between model and environment: tool-installed check, unknown/empty-arg dropping, type coercion
  (`"2.000.000"` → `2000000`), date-time normalisation to `YYYY-MM-DDTHH:MM:SS+07:00`, duplicate-read blocking, `x-ref`
  enforcement, and `session.validate()` before dispatch.
- **One model step per tool call** (not a batch), with a compacted observation fed back. Simpler to reason about, easier to
  debug, and each step is independently journaled.
- **A relative-day table in the prompt** (`hôm nay (thứ Ba) = 2026-11-24` …) so the model never does date arithmetic.
- **Constrained JSON decoding**: the action is a single object `{"name","arguments"}` or `{"name":"finish"}`, constrained
  via an `anyOf` schema over installed tools. Native function calling is not required.
- A **Vietnamese system prompt encoding labelling policy learned from train**: finish immediately for out-of-scope, only
  fill optional args the user actually stated, read-before-write for IDs, ask only when genuinely ambiguous.
- **Stop after `sys.ask_missing_info`** — gold always ends the turn there.
- Agent replaying gold calls through the whole pipeline scored **99.99** on train, isolating the pipeline from the model.

### Runtime measured

Full public inference: 6,914.7 s for 634 samples = **10.91 s/sample** average, inside the 30 s cap with room. The
`time_budget_s: 25.0` guard with `max_steps: 8` was sufficient.

---

## 7. Model parameter facts (verified via Hugging Face API)

| Model | Safetensor params | Licence | Created | Revision |
|---|---:|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | 4,022,468,096 | Apache-2.0 | 2025-08-05 | `cdbee75f17c0…` |
| `Qwen/Qwen3.5-4B` | 4,659,865,088 | Apache-2.0 | 2026-02-27 | `851bf6e806ef…` |
| `Qwen/Qwen3-ASR-0.6B` | 938,008,576 | Apache-2.0 | 2026-01-28 | `5eb144179a02…` |
| `g-group-ai-lab/gipformer1.5-68M-rnnt` | (name implies 68M — audit required) | MIT | 2026-08-21 | — |
| `vinai/PhoWhisper-medium` | 769M (bin 3.06 GB) | BSD-3-Clause | 2024-02-18 | — |
| `vinai/PhoWhisper-large` | 1.55B (bin 6.17 GB) | BSD-3-Clause | 2023-12-19 | — |

All pre-cutoff. Qwen3.5-4B at 4.66B is the largest LLM that fits under the 5B cap while leaving room for ASR.
Note that Qwen3.5-4B's full checkpoint includes a vision tower; whether that counts against the LLM budget is **still an
open question for the organisers** (G07).

---

## 8. Revised parameter budget

| Configuration | Total | LLM | Verdict |
|---|---:|---:|---|
| Qwen3.5-4B + gipformer-68M | **4.73B** | 4.66B | **Recommended** — most headroom, best ASR, CPU-side ASR |
| Qwen3.5-4B + Qwen3-ASR-0.6B | 5.60B | 4.66B | Legal but ASR 2× worse and steals VRAM |
| Qwen3-4B + gipformer-68M | 4.09B | 4.02B | More headroom, weaker policy |
| Qwen3.5-4B + PhoWhisper-large | 6.21B | 4.66B | Excluded |
| Qwen3.5-4B + PhoWhisper-medium | 5.43B | 4.66B | Legal, but WER far worse than gipformer |

The PRD's "PhoWhisper is the ASR baseline" assumption should be dropped.

---

## 9. Open gates that remain

| Gate | Status | Action |
|---|---|---|
| G01 dataset files/metadata | **partly closed** | Scale, layout and split known; audio duration distribution still unknown until the kit lands |
| G02 schemas / tool semantics | **closed** | `<split>_tools_catalog.json` with `kind`, `x-match`, `x-ref`, `x-one-of` |
| G03 `ToolEnv` API / hash lifecycle | **closed** | `ToolEnv`/`Executor`, `call`/`validate`/`get_tool_calls`/`get_state_hash` confirmed |
| G04 fuzzy fields / matching | **closed** | Exact implementation above |
| G05 Docker entrypoint / layout | **closed** | `/data` in, `result/submission.jsonl` out; layout confirmed |
| G06 timing boundary (startup included?) | **open** | Ask organisers |
| G07 parameter counting (vision tower? tied weights?) | **open** | Ask organisers; audit tensors exactly |
| G08 loopback server acceptability | **open** | Ask organisers |
| G09 mixed supported/unsupported, partial state | **open** | Inspect train examples once the kit lands |
| G10 training-data inclusion vs distribution ban | **open** | Ask organisers before publishing any archive |
| G11 post-freeze SDK replacement | **open** | Ask organisers |
| G12 result-selection UI / reset timezone | **open** | Portal |
| G13 cloud compute / hosted assistants | **open** | Ask organisers before sending competition data anywhere |

---

## 10. Questions to send the organisers

1. Does the 30 s per sample include container startup and model loading?
2. Which directory is the working directory that `result/` resolves against — `/app/result`?
3. How is the private `zaic_tools` version injected into the frozen image?
4. Does "every call must appear in the submission" exclude lifecycle methods like `get_state_hash()`?
5. Does the public Drive link have a licence exception for required training assets (G10)?
6. Does the Qwen3.5-4B vision tower count against the 5B LLM limit (G07)?
7. Is a loopback HTTP server inside the container acceptable (G08)?
8. Is cloud GPU compute permitted, and may competition data be processed there (G13)?

---

## 11. The public leaderboard is readable without authentication

**Verified 29 October 2026, evening.** The earlier note that the leaderboard returned an empty array was wrong: it
needed the correct `title` parameter, not a token. The endpoint is public:

```
GET https://api-challenge.zalo.ai/fe/api/leaderboard?from=0&count=100&title=1
  headers: Origin: https://challenge.zalo.ai, Referer: https://challenge.zalo.ai/
```

`title=1` is the **LifeMate** track; `title=2` is DataMate (scores there are 0–1, a different metric). Each row carries
`rank`, `score`, `name` (team), `members` (a JSON string of member names) and `created_time`. Pagination is by `from`
offset with `count` per page.

### Snapshot of the LifeMate public leaderboard (225 teams)

| Rank | Score | Team |
|---:|---:|---|
| 1 | 97.63 | Hasagi |
| 2 | 96.49 | getout |
| 3 | 95.82 | Tấn Cường - Tèo |
| 4 | 95.67 | Spartans-ETL |
| 5 | 95.63 | SpETL |
| 9 | 94.93 | Kaseton |
| 23 | 88.45 | ZAIC_2_25 |
| — | 43.06 | KAIZEN (rank 113) |
| — | 42.13 | V2Act (rank 114) |
| 138 | 31.12 | ZAIC_Celestials |
| 225 | 0.00 | Odylytics |

Distribution: best 97.63, **median 43.06**, worst 0.00.

| Score band | Teams | Share |
|---|---:|---:|
| 90–100 | 22 | 9.8% |
| 80–90 | 8 | 3.6% |
| 70–80 | 8 | 3.6% |
| 60–70 | 22 | 9.8% |
| 50–60 | 35 | 15.6% |
| 40–50 | 23 | 10.2% |
| 30–40 | 23 | 10.2% |
| 20–30 | 26 | 11.6% |
| 10–20 | 21 | 9.3% |
| 0–10 | 37 | 16.4% |

38 teams are at 70 or above; 22 are at 90 or above. The shape is a wide cliff: the top 22 teams are separated from the
median by roughly 50 points, and a third of the field is below 30.

### How to read this

- The public leaderboard is **provisional**. The portal itself warns it "may not accurately reflect the actual performance
  of the participating systems", and the private test decides the final standing.
- The leaderboard is **ordered chronologically, not by score**, for the submission list; the score column is what ranks.
- A rank for a *hypothetical* score can be computed (strictly-higher count + 1), but ties are broken by submission time,
  which is not knowable in advance.
- Our own position is not knowable until we submit a result. `scripts/leaderboard.py` computes the placement of any
  candidate score and prints the neighbouring teams.

### Consequences for our targets

The reference team's public score of **7.85** (section 6) sits in the bottom 20% of a field that has since moved sharply
upward. Their measured local proxy at that time was F1 ≈ 0.24. The teams now at 90+ are therefore running something
materially better than a Qwen3-4B-Q4 heuristic with retrieval.

Two readings, and we should not pick the comfortable one:

1. The easy reading is that the leaderboard has improved because every team iterated over three days.
2. The harder reading is that **7.85 was a pipeline result, not a model result** — 70% of the errors were `WRONG_TOOL`,
   and a wrong tool means wrong state means zero for the sample. Teams that fixed tool selection rather than model
   quality would move the score a long way.

Either way the target is set by the leaders, not by our reference point. A defensible first milestone is **a reproducible
local score above the public median**, then a public submission to establish where we actually stand.

---

## 12. SECOND VERIFICATION PASS — 1 November 2026 (kit in hand)

The competition kit arrived locally (`data/train+public_test/`). Every fact below was **measured directly against the
released files**, not taken from a competitor. Where it corrects an earlier section, the correction is stated.
Reproduce with `research/scorer.py`, `research/strategy.py`, `research/analyze.py`, `research/verify_metric*.py`.

### 12.1 Corrections to earlier sections

| Earlier claim | Measured truth |
|---|---|
| §4 "Distinct apps ~50" (observed by competitors) | **46 apps in the train catalog, 36 in the public_test catalog**; 45 distinct apps actually installed across train samples, 35 across public test. ~50 was a rounded guess in the right direction. |
| §4 "audio duration distribution unknown" | **Measured**: mean 4.40 s, median 3.41 s, p90 8.25 s, p99 12.77 s, max 15.69 s. 16 kHz mono 16-bit. Train total ≈ 6.2 h. |
| §4 out-of-scope "~7.1%" | **7.18%** (111/1,545) — and an empty submission scores exactly **7.18/100**, confirming the out-of-scope rule is the only thing an empty submission earns. |
| §7 gipformer "audit required" | **68,625,511 params** (`config.json: num_parameters`), MIT, `zipformer-transducer`, Vietnamese, 16 kHz, ONNX + int8 for sherpa-onnx, `model.pt` (icefall) for fine-tuning. |
| §7/§9 G07 "vision tower counting open" | **Moot.** Qwen3.5-4B's full checkpoint is 4,659,865,088 params *including* its vision tower — under the 5B LLM cap under any counting convention. With gipformer: 4.73B system total. No organiser ruling needed. |
| §11 leaderboard snapshot (225 teams, best 97.63, median 43.06) | **Superseded.** Now **357 teams, best 99.37, median 48.88, 41 teams ≥ 90**. The field moved up sharply in three days. |

### 12.2 The metric instrument is now proven exact

Replaying gold through the real `ToolEnv` and scoring with our own implementation of the official formula:

| Check | Result |
|---|---|
| Gold replay reproduces `label.final_state` | **1,545 / 1,545** |
| Gold replay score under our scorer | **1.000000**, perfect on **1,545 / 1,545** |
| Empty-submission score | **7.18 / 100** (= out-of-scope rate, exactly) |

**Consequence:** `label.final_state` *is* the gold state, so the exact gold `state_hash` is computable offline — the
local scorer needs no secret. Every subsequent number is trusted only through this scorer. Implementation note that
matters: **fuzzy-ness is a property of (tool, argument)**, so the fuzzy set must be selected per call — pooling fuzzy
keys across tools inflates the score and hides real errors.

### 12.3 Measured cost of each error class (n=400, state gate on)

| Perturbation | Mean score | Samples killed by the state gate |
|---|---:|---:|
| control (gold) | 1.0000 | 0 / 400 |
| drop one **required** argument | **0.0688** | 355 / 400 |
| perturb one **non-fuzzy** argument | **0.2125** | 268 / 400 |
| drop the last call | **0.1915** | 292 / 400 |
| drop one **optional** argument | **0.5107** | 164 / 400 |
| wrong first tool name | **0.5964** | 50 / 400 |
| int argument passed as string | **0.7725** | 91 / 400 |
| add one extra `sys.ask_missing_info` | **0.8179** | 0 / 400 |
| reverse call order | **0.9275** | 29 / 400 |
| perturb one **fuzzy** argument | **1.0000** | 0 / 400 |

Read this as the priority order for engineering effort: **required arguments and typed values first** (a single error
there costs the whole sample), **argument key sets next** (never add an optional argument the user did not state),
**fuzzy wording last** (it never touches the hash).

Fuzzy boundary behaviour (token-F1, NFC + lowercase, diacritics kept):
`"Mai Anh"` vs `"mai anh"` → 1.000 (match) · `"Bật điều hòa"` vs `"bật điều hoà"` → 0.667 (match) ·
`"Hương đi học"` vs `"huong di hoc"` → 0.000 (**no match** — diacritics are significant) ·
`"abc def"` vs `"abc xyz"` → 0.500 (no match).

### 12.4 The gate decomposes the task into two very different populations

| Gold shape | Count | Share | What it takes to score |
|---|---:|---:|---|
| empty (out of scope) | 111 | 7.2% | Emit **nothing**. Any call → 0. |
| reads only, no state change | 124 | 8.0% | Pure F1. The hash gate is free. |
| ≥ 1 write | 1,172 | 75.9% | The hash must match **exactly** — one wrong write = 0 for the sample. |

So ~76% of the score is gated on exact write correctness, ~15% is pure call-matching, and ~7% is free if the agent
correctly recognises out-of-scope requests and does nothing.

### 12.5 Gold trajectory structure (train, measured)

- Calls per sample: 0 → 111, 1 → 369, 2 → 428, 3 → 359, 4 → 161, 5 → 107, 6 → 10. **Max 6** — `max_steps = 8` covers
  100% of gold trajectories with margin.
- `sys.ask_missing_info` appears in **217 samples (14.0%)** and is **always the last call (217/217)**, always with
  exactly `{tool_name, params}`. It is also the first call in only 3.9% of samples — i.e. the agent usually performs
  what it *can* first, then asks.
- **55.1%** of samples carry an ID-shaped argument that must come from a prior read; **1,243 / 1,278** x-ref values
  appear verbatim in an earlier `tool_results`. ID hallucination is a guaranteed zero.
- **27.3%** of gold calls carry **no arguments at all**.
- Gold is schema-perfect: **0 / 3,541** calls have out-of-schema arguments; **0** miss a required argument.
- **51.8%** of transcripts contain Vietnamese number words ("hai triệu tám", "một nửa", "30 phần trăm") while gold
  numeric arguments are typed integers. Spoken-form → typed-number normalisation is mandatory.
- Tool kinds in the train catalog: read 114, write 153, meta 2. `x-match: fuzzy` on 32 tools, `x-ref` on 65,
  `x-one-of` on 6. Parameter types: string 204, integer 25, boolean 6, array 10, number 1.

### 12.6 The hash is split-specific — a real trap

`train/zaic_tools/utils.py` and `public_test/zaic_tools/utils.py` **differ**:

```
train:       LIVE_APPS = {exchange_rate, fuel_price, gold_price, lottery, weather}
public_test: LIVE_APPS = {fuel_price, gold_price, weather}
train:       FREE_TEXT_KEYS = {address, body, content, dropoff, expression, label, location, name, note, pickup, subject, title}
public_test: FREE_TEXT_KEYS = {address, body, content, dropoff, label, location, name, note, subject, text, title}
```

`pickup` and `expression` are hash-relevant in public_test but not in train; `text` is the reverse; two live apps differ.
Every other app module is byte-identical between the splits. **Never reimplement the hash — always call the split's own
`env.get_state_hash()`.** A hash copied from one split will silently mismatch on another.

### 12.7 Prompt budget (measured, drives the architecture)

| Component | Mean | Max |
|---|---:|---:|
| Installed tool schemas (per sample) | 3,125 tok | 6,032 tok |
| History | 407 tok | 2,406 tok |
| **Full prompt** (schemas + context + history + transcript) | **3,913 tok** | **7,393 tok** |

Installed tools per sample: mean **32.5**, median 31, max 62. Installed apps: mean 9.2, max 18.
**Full per-sample schemas fit comfortably in the 16,384-token cap** — hard tool retrieval is not structurally
necessary, and the plan should not pay its complexity cost. (A competitor found retrieval variants *lowered* their public
score; §6 corroborates.)

### 12.8 Audio (measured)

16 kHz mono, 16-bit. Train: 1,545 sample directories, 5,036 utterances, mean 4.40 s, median 3.41 s, p99 12.77 s,
max 15.69 s, ≈6.2 h total. Public test: 634 samples, 419 with history, up to 24 history turns per sample.

### 12.9 The 11 apps that appear in public_test but never in train

`ecommerce, event_tickets, grocery, habit_tracker, pantry, parking, public_transit, recipes, stocks,
train_bus_tickets, translator` — confirmed exactly, and 21 train apps are absent from public test.
Leave-apps-out validation is the honest generalisation proxy for the private test.

### 12.10 Inference-time data shapes

`public_test.jsonl` carries only `id`, `audio`, `context`, `history` — **no transcripts, no labels**. History user turns
carry `audio` only (must be transcribed); assistant turns carry `text`, `tool_calls`, and `tool_results`
(1,106 of 1,299 assistant turns have tool results). The environment folder holds 634 `state.json` files, and
`_meta.tools` in the state matches the sample's installed tool count exactly.

### 12.11 Leaderboard, 1 November 2026

**357 teams.** Best **99.37** (Xcen), median **48.88**, 41 teams ≥ 90, 80 teams ≥ 70, 170 teams ≥ 50. The reference
team's 7.85 (section 6) now sits in the bottom ~5%. The field is not a soft distribution any more — the gap between the
median and the leaders is ~50 points, so a competitive entry must fix tool selection and write-exactness, not
incrementally improve phrasing.

### 12.12 Gate status after this pass

| Gate | New status |
|---|---|
| G01 dataset files/metadata | **closed** — everything measured in §12.1–12.8 |
| G07 parameter counting | **closed** — Qwen3.5-4B at 4.66B *including* the vision tower is under the 5B cap |
| G09 mixed supported/unsupported, partial state | **closed** — see §12.4 (7.2% out-of-scope, gate decomposition) |
| G06 timing boundary, G08 loopback server, G10 training-data licensing, G11 post-freeze SDK, G13 cloud-GPU policy | **still open** — ask the organisers |
