# Decision Log

## Decision 1 — Clean-room implementation, reference repos used as spec only

**What was changed:** All code in `src/lifemate/` was written from the verified contract
in `docs/VERIFIED_FACTS.md`. Two public competitor repositories
(`MrPickle200/zalo-ai-2026-lifemate`, `aihacker111/Zalo-AI-Challenges-LifeMate`) were read
to establish the kit's API and metric behaviour, but no code, prompt, or file was copied.

**Reason:** The repositories hold the only public descriptions of `zaic_tools`, the metric
and the split layout. Reading them as a specification is far cheaper than rediscovering the
contract by inspection, and the metric in particular must match exactly rather than
approximately. Reimplementing from a spec also leaves us owning the architecture, which
matters for a competition entry reviewed for source code.

**Alternatives:**
- *Fork the MIT-licensed repo.* Fastest path to a working submission, but adopts another
  team's architecture wholesale, including its choices we might not want (hard tool
  shortlists) and its unverified compliance items.
- *Pure black-box reconstruction from the kit.* Cleanest provenance, but serialises behind
  data arrival that has not happened yet.

**Tradeoff:** We accept some duplicated effort — the guard, prompt builder and scorer will
look superficially similar to the reference because they solve the same problem — in exchange
for an architecture we designed against the metric rather than inherited from it.

---

## Decision 2 — Stock gipformer1.5-68M as the ASR, no fine-tuning yet

**What was changed:** The PRD's PhoWhisper recommendation is dropped. The ASR is
`g-group-ai-lab/gipformer1.5-68M-rnnt` (MIT), used stock, via sherpa-onnx.

**Reason:** Measured on 878 held-out Vietnamese utterances, stock gipformer reaches 15.5%
WER against 27–36% for Qwen3-ASR-0.6B. At 68M parameters it also frees the parameter
budget (Qwen3.5-4B + gipformer = 4.73B total, inside the <6B cap) and runs on **CPU**, so
ASR does not contend for A30 VRAM with the LLM. Fine-tuning reaches 10.1% but costs ~7
GPU-hours plus an icefall/k2 setup, and its benefit cannot be measured until the training
audio arrives.

**Verified locally:** the upstream repository already publishes an ONNX export
(`encoder/decoder/joiner` + `tokens.txt` + `bpe.model`), so no export step is needed — the
PRD's assumption that one would be required is wrong. On three real Vietnamese utterances
(synthesised via `edge-tts`, decoded to 16 kHz mono) the model transcribed all three
correctly, with diacritics intact, in 0.15–0.3 s per utterance on CPU. That is roughly
10x faster than real time, which fits the 30 s per-sample budget with room to spare.

**Alternatives:**
- *Fine-tune gipformer now.* Roughly 5 points of WER, but the val split cannot be built
  without the kit, and the training audio is not yet in hand.
- *Qwen3-ASR-0.6B.* Better-documented and simpler to fine-tune later, but 938M parameters
  and roughly double the WER. We keep it as the fallback if gipformer's ONNX export turns
  out to be awkward on the target container.

**Tradeoff:** We give up ~5 points of WER now. If speech turns out to dominate the error
budget, that was the wrong call — which is exactly what the E0 diagnostic (reference
transcript vs ASR transcript) is designed to reveal before we commit.

---

## Decision 3 — The fuzzy comparison is diacritic-sensitive; do not fold when matching

**What was changed:** `text.fold()` (diacritic stripping) exists but is used **only** for
entity candidate filtering. The metric's fuzzy comparison uses NFC + lowercase only, and
`text.py` documents the distinction explicitly.

**Reason:** Verified numerically: `token_f1("Hương đi học", "huong di hoc") == 0.0`. The
evaluator's normalisation is lowercase + NFC; diacritics survive into the token stream.
A Vietnamese tone error therefore costs the full fuzzy score on that argument, and a
free-text argument that differs only in tone can drop from 1.0 to 0.0.

**Alternatives:**
- *Fold diacritics in matching.* Produces more "helpful" matches locally, but diverges
  from the official evaluator — precisely the failure mode that made another team's
  local gains lose points on the public leaderboard.

**Tradeoff:** We cannot be lenient about Vietnamese spelling in fuzzy fields. This raises
the value of high-quality ASR diacritics and of preserving the user's exact wording, and
it means `fold` must never be used on a value that will be submitted.

---

## Decision 6 — Recase the ASR transcript to sentence case before it reaches the model

**What was changed:** gipformer emits uppercase text with no punctuation. The transcript
is recased to sentence case — lowercased, with the leading letter of each sentence
restored — before it is shown to the model. A full stop between digits is treated as a
thousands separator rather than a sentence end, so "200.000 ĐỒNG" does not become
"200.000 Đồng".

**Reason:** A non-fuzzy string argument is compared with exact equality. A transcript
copied into a title or content field as "BẬT ĐIỀU HÒA" does not match a gold value of
"Bật điều hòa", and one wrong argument costs the whole sample. Showing the model an
uppercase request also invites it to write uppercase arguments.

**Alternatives:**
- *Leave the transcript uppercase.* Guaranteed mismatch on any non-fuzzy free-text
  argument, and the model is likely to reproduce the shouting.
- *Title-case every word.* "Mai Anh" survives but "tắt đèn bếp" becomes "Tắt Đèn Bếp",
  which is not how Vietnamese is written and not how the gold is written.

**Tradeoff:** Proper nouns lose their capital — "Mai Anh" becomes "mai anh". This is a real
cost, and it is only safe because free-text arguments are usually marked `x-match: fuzzy`,
where the comparison lowercases both sides and the token F1 still clears 0.6. If the
competition data turns out to have many non-fuzzy free-text fields, this decision should
be revisited against a name-aware recasing pass.

---

## Decision 7 — Schema validation runs before the grounding check

**What was changed:** In the guard, `session.validate()` is checked before the `x-ref`
grounding check.

**Reason:** A malformed call reported as "unknown identifier" sends the model to read a
device list instead of fixing the argument it already got wrong. The wrong repair wastes a
turn and can still leave the call invalid, so the more fundamental defect has to be
reported first.

**Alternatives:**
- *Grounding first.* It is the more semantically interesting failure, so it is tempting —
  but it reports a consequence rather than the cause.

**Tradeoff:** A call that is both malformed and ungrounded is told about the malformation
first. The model may take one extra turn to discover the grounding problem, but each turn
reports something the model can actually fix.

---

## Decision 8 — Argument coercion is bidirectional towards the declared type

**What was changed:** The guard coerces a proposal towards the type its schema declares,
in both directions: `"26"` → `26` for an integer parameter, and `26` → `"26"` for a
string parameter.

**Reason:** The gold calls were produced against the same schema, so a gold value for a
`string` parameter is a string and a gold value for an `integer` parameter is an integer.
Moving a proposal onto the declared type therefore moves it towards gold in either
direction. The evaluator compares types exactly, so this matters: a `"26"` where the gold
says `26` is a mismatch.

**Alternatives:**
- *One-directional coercion only.* Preserves the caller's intent more literally, but
  leaves a mismatch that the schema itself tells us how to fix.

**Tradeoff:** The schema is trusted as the authority on a value's type. If the catalog
ever disagrees with the gold data on a type, coercion would push the proposal towards the
wrong form. That is worth checking against the real catalog when it arrives.

---

## Decision 4 — One model step per tool call, with a deterministic guard in between

**What was changed:** The agent emits exactly one action per model round
(`{"name", "arguments"}` or `{"name": "finish"}`), and a deterministic guard sits between
the model and the environment.

**Reason:** It makes every dispatched call independently attributable in the journal,
keeps the trace compact enough for 16K context, and lets `session.validate()` reject a
malformed proposal *before* dispatch — validation is not journaled, so a rejected proposal
costs nothing against the tool-call F1. The reference team measured replaying gold calls
through the same pipeline at 99.99 on train, which isolates pipeline bugs from model
quality.

**Alternatives:**
- *Batch multiple calls per model round.* Fewer rounds, less prefill. But it forces
  speculative execution of calls whose arguments depend on results not yet observed, which
  risks wrong state — and wrong state zeroes the whole sample.
- *Native function calling.* The submission format is independent of the wire format, so
  schema-constrained JSON over the same transport gives us the same output with fewer
  server-parsing assumptions.

**Tradeoff:** More model rounds means more prefill and more latency per sample. Measured
reference runtime was 10.9 s/sample against a 30 s cap, so the budget has room.

---

## Decision 5 — Metric fidelity is the deliverable, not the model

**What was changed:** The first build phase is the metric, the split loader and the
replay-based scorer — before any agent code. `tests/test_metric.py` pins every rule the
official evaluator applies, including the ones that are easy to get wrong.

**Reason:** The reference team's local proxy improved on every metric (F1 0.1155 → 0.2435,
state match 24.9% → 29.8%, 86 samples improved vs 32 regressed) and still **lost 0.34
points on the public leaderboard**. The local scorer ignored the state-hash gate and the
exact-argument-key-set rule, so it optimised the wrong objective. A metric that is
provably identical to the official one is what makes every later measurement trustworthy.

**Alternatives:**
- *Start on the agent and validate the metric later.* Feels faster, but it means the first
  real result arrives with an unverified measuring instrument — and would likely be
  discarded for being misleading.

**Tradeoff:** A few days before any model runs. Recovered many times over if the metric
is right, and we lose the whole schedule if it is subtly wrong.

---

## Decision 9 — The public leaderboard is the reference frame, and it is readable without a token

**What was changed:** Added `scripts/leaderboard.py`, which fetches the portal's public leaderboard endpoint and reports
the distribution, the top teams, and the rank a hypothetical score would take. Recorded the snapshot and the distribution
bands in `docs/VERIFIED_FACTS.md` section 11.

**Reason:** The earlier belief that the leaderboard needed authentication was wrong — it needed the correct `title`
parameter. That changes what we can plan against: the field is observable continuously, so our score can be compared to
competitors without waiting for the kit. It also raises the bar. The reference team's public 7.85 sits in the bottom
fifth of a 225-team field whose top is at 97.63 and whose median is 43.06. Planning against our own reference point
rather than against the leaders would be self-deception.

**Alternatives:**
- *Ignore the leaderboard until we submit.* Fewer moving parts, but we would calibrate against a stale impression.
- *Poll the leaderboard automatically as a pipeline stage.* Tempting, but leaderboard reading is a manual judgement call:
  a script that reports "we are rank 87" without the score distribution would encourage optimising a number instead of
  the underlying error taxonomy.

**Tradeoff:** Watching the field creates pressure to submit for the leaderboard's sake. We cap public submissions at the
five-per-day allowance and only submit after a locally justified change, so the leaderboard stays a measurement rather
than a target. The tool deliberately cannot report our own rank, because a team's own score is only knowable after a
submission.

---

## Decision 10 — The local scorer is the instrument; every number is quoted through it

**What was changed:** `research/scorer.py` implements the official formula (state gate →
maximum bipartite matching → per-argument fuzzy/exact comparison → F1) and is now the only
source of truth for internal measurement. The PRD quotes no metric number that has not been
produced by it.

**Reason:** A competitor improved every one of their local proxies and still lost 0.34 points
on the public board, because their proxy measured set-based F1 and final-state equality but
not the official combination. Our instrument is verified exact: replaying all 1,545 gold
trajectories reproduces `label.final_state` on 1,545/1,545 and scores **1.000000**, and an
empty submission scores **7.18/100** — exactly the out-of-scope rate. Since `label.final_state`
*is* the gold state, the exact gold `state_hash` is computable offline; the scorer needs no
secret and no organiser cooperation.

One implementation detail is load-bearing and was initially wrong: **fuzzy-ness belongs to
the (tool, argument) pair**, so the fuzzy set must be selected per call. Pooling fuzzy keys
across tools inflated the score and hid real errors.

**Alternatives:**
- *Trust the leaderboard only.* Cannot be used: 5 submissions/day, and the board is explicitly
  provisional.
- *Use set-based F1 as a proxy.* Cheaper, and demonstrably misleading (see above).

**Tradeoff:** The scorer costs a full environment replay per sample (~50 s for 1,545 samples
plus degraded variants). That is acceptable because the instrument must be trusted more than
it must be fast, and the replay is the only way to compute the true state hash.

---

## Decision 11 — Never reimplement the state hash; always use the split's own runtime

**What was changed:** The plan calls `env.get_state_hash()` from the split's `zaic_tools`
package and nothing else. No hash logic lives in our code.

**Reason:** The hash canonicalisation is **split-specific**. `train/zaic_tools/utils.py` and
`public_test/zaic_tools/utils.py` differ in `LIVE_APPS` (`exchange_rate` and `lottery` are
live only in train) and in `FREE_TEXT_KEYS` (`pickup` and `expression` are hash-relevant in
public_test but not train; `text` is the reverse). Every other app module is byte-identical.
A hash copied from one split silently mismatches on another, and the private test ships its
own version.

**Alternatives:**
- *Reimplement the hash for speed.* Avoidable: the environment already computes it, and any
  divergence from the organisers' version is fatal by rule.
- *Assume the splits are identical.* Now disproved by source diff.

**Tradeoff:** We depend on the split's package being importable and on two versions of it
coexisting when we score train and public_test in one process. We load each under a unique
module name to avoid a package-name collision.

---

## Decision 12 — Ship stock gipformer; ASR fine-tuning is gated behind a diagnostic

**What was changed:** Confirmed Decision 2 against the kit. The ASR plan of record is
**stock** gipformer1.5-68M-rnnt via sherpa-onnx on CPU. Fine-tuning is a stretch goal that
is only unlocked if diagnostic E0 (reference transcript vs ASR transcript ablation) shows
speech in the dominant error path.

**Reason:** The measured error taxonomy of a weak baseline is **WRONG_TOOL 70.3%**, ASR_SLOT
5.5% — speech is not the bottleneck. Fine-tuning gipformer also costs an icefall/k2 setup and
~7 GPU-hours for ~5 WER points, and a competitor measured that such a fine-tune *lowered*
their downstream score despite improving WER (their fine-tuned ASR had memorised train audio,
so the policy never saw realistic ASR errors). Separately, the ASR runs on CPU while the LLM
owns the GPU, so it does not contend for the 30 s budget.

**Verified locally:** upstream publishes `encoder/decoder/joiner` ONNX plus int8 variants and
`tokens.txt`/`bpe.model` — no export step needed. `config.json` reports
`num_parameters: 68,625,511`, MIT, `zipformer-transducer`, Vietnamese, 16 kHz.

**Alternatives:**
- *Qwen3-ASR-0.6B.* Better-documented, easier to fine-tune, but 938M params and roughly double
  the WER (36.1% vs 16.1%). Kept as fallback.
- *Fine-tune immediately.* Spends the scarce GPU-hours before the error budget says where to
  spend them.

**Tradeoff:** ~5 WER points of headroom left on the table. E0 is the release valve: if speech
does dominate on our own split, we fine-tune and the cost is a known, bounded one.

---

## Decision 13 — LLM = Qwen3.5-4B; the vision-tower question is moot

**What was changed:** The LLM is `Qwen/Qwen3.5-4B` (4,659,865,088 params, Apache-2.0,
2026-02-27). `Qwen/Qwen3-4B-Instruct-2507` (4,022,468,096) is the fallback. System total with
gipformer: **4.73B**, inside the <6B cap.

**Reason:** Qwen3.5-4B's checkpoint is 4.66B **including its unused vision tower**, so it fits
under the <5B LLM cap under any counting convention — the open organiser question (G07)
disappears without needing a ruling. Apache-2.0 and a February 2026 release both satisfy the
"released before 26 Oct 2026" rule. The fallback exists because the text-only 4B is stronger on
parameter accounting (4.09B system total) if the 4B policy quality proves insufficient.

**Alternatives:**
- *Qwen3-4B-Instruct-2507 as primary.* Zero parameter ambiguity, but a materially weaker
  policy, and policy quality is the binding constraint (70.3% of errors are tool selection).
- *A larger model + smaller ASR.* Every combination above 6B is excluded by rule.

**Tradeoff:** ~0.34B of headroom is spent on a vision tower we never run. We accept that to
keep the strongest policy that legally fits.

---

## Decision 14 — No hard tool retrieval; show every installed schema

**What was changed:** The prompt carries the **full** set of installed tool schemas for the
sample (mean 32.5 tools, 3,125 tokens mean / 6,032 max), not a retrieved shortlist.

**Reason:** Measured prompt budget is **3,913 tokens mean / 7,393 max** against a 16,384-token
cap — full schemas fit with room to spare, so retrieval buys nothing structurally. Empirically
it can cost: both a competitor's hard-shortlist and soft-retrieval variants scored *lower* on
the public board than their plain baseline, and their local proxy improved while the board
score fell. That is a warning about the local instrument, which we have now fixed (Decision 10).

**Alternatives:**
- *TF-IDF top-k shortlist.* Cheaper prompt, measurably worse public score in the one public
  data point available.
- *Embedding retrieval over 147 catalog tools.* Would help only if the prompt did not fit; it
  does, so this adds a failure mode (wrong shortlist) for no gain.

**Tradeoff:** Longer prompts mean more tokens per model step, against a 30 s budget with a
measured reference of 10.91 s/sample. The budget is dominated by generation, not by our input
length, so this is acceptable — but it must be re-measured once the real pipeline runs.

---

## Decision 15 — Numbers and dates are resolved by deterministic code, not by the model

**What was changed:** A spoken-form → typed-number normaliser (Vietnamese number words,
percentages, fractions, money units nghìn/triệu/đồng) and a relative-day table are part of
the pipeline; the model never performs this arithmetic.

**Reason:** **51.8%** of transcripts contain Vietnamese number words ("hai triệu tám", "một
nửa", "30 phần trăm") while gold numeric arguments are typed integers. The measured cost of a
single perturbed non-fuzzy argument is **0.213 mean score, killing 268/400 samples on the
state gate** — so a number the model has to compute is a number that can zero the sample.
Deterministic code removes the class of error entirely.

**Alternatives:**
- *Let the model do it.* Models are good at this in English; Vietnamese spoken-form arithmetic
  with money units is a known weak spot, and the failure is asymmetric in cost.
- *Drop the number from the prompt.* Unacceptable: it is required information.

**Tradeoff:** An extra deterministic component that must itself be tested against Vietnamese
number forms. It is testable offline against the 1,545 gold transcripts, which is exactly the
property that makes it safe to trust.

---

## Decision 16 — Validate before dispatch; a rejected call is not free

**What was changed:** `session.validate()` runs **before** any dispatch, and the guard repairs
arguments (types, date-times, empty optionals) rather than passing them through.

**Reason:** Measured error costs (n=400): dropping one **required** argument drops the mean
score to **0.0688** and kills 355/400 samples on the state gate. A schema-rejected call is
also counted in `U` by the official formula, so it hurts twice. Validating first converts a
fatal class of errors into a recoverable one — the model gets the validation error back and
can fix the call.

Note the counter-intuitive case that sets the design: **dropping one optional argument still
costs 0.511 mean score and kills 164/400 samples.** Because the matcher compares the full
argument key set, adding an argument the user never mentioned breaks the match just as
surely as omitting one that gold contains. The guard therefore drops optional arguments
rather than filling them in.

**Alternatives:**
- *Validate after dispatch.* Simpler, but the write has already corrupted the state.
- *Trust the model's schema compliance.* Measured evidence says otherwise.

**Tradeoff:** An extra call round-trip per proposal. `validate()` is unjournalled, so it costs
nothing against the metric — only latency.

---

## Decision 17 — The metric lives in the package, and the reference script is its test

**What was changed:** `research/scorer.py` proved the metric; `src/lifemate/metric.py` now
implements it as the shipped code, and `tests/test_metric.py` asserts that the packaged
version reproduces the reference's two published numbers — gold replay `1.000000` on all
1,545 train samples, and `7.18/100` for an empty submission.

**Reason:** The scorer is the instrument every number in the PRD is quoted from. If the
shipped code and the research script could drift, the PRD's numbers would describe a scorer
the submission does not use. The research script also hardcoded the `train` split; the
packaged version takes the split as an argument, which is what the private test needs.

**Alternatives:**
- *Ship the research script as the scorer.* It is proven, but it hardcodes paths and one
  split, and it was never written to be a library.
- *Trust the research script and write no test.* Then the number in the README is a claim
  about a file, not about the code that runs.

**Tradeoff:** The metric now exists twice — a research script and a package module. The test
that compares them is the cost of that duplication, and it is worth paying: it fails loudly
the moment either copy changes.

---

## Decision 18 — Each split's runtime is loaded under a unique module name

**What was changed:** `src/lifemate/kit.py` imports `<split>/zaic_tools` under the name
`zaic_tools__<split>` and caches it per split, instead of importing `zaic_tools` by its
plain name.

**Reason:** `train/zaic_tools/utils.py` and `public_test/zaic_tools/utils.py` define
different `FREE_TEXT_KEYS` and `LIVE_APPS` sets, and those sets change the state hash
(docs/VERIFIED_FACTS.md §12.6). A plain `import zaic_tools` binds the **first** split's
runtime for the whole process, so the second split would be scored against the wrong hash —
a silent, total-scoring failure. The scorer needs both splits in one process (replay the
gold trajectory for the hash, replay the submission against the same split), so the
collision is not hypothetical.

**Alternatives:**
- *One process per split.* Avoids the collision, but makes the scorer run the kit twice and
  invites exactly the copy-paste mistake that causes it.
- *Reimplement the hash.* Explicitly forbidden: the hash is split-specific by construction.

**Tradeoff:** A dynamic import with a non-standard module name is less obvious than a plain
import, and the package must be registered in `sys.modules` before execution or its own
relative imports fail. Both are handled in one small module, and a test asserts the two
splits load independently.

---

## Decision 19 — The agent loop is testable without a model

**What was changed:** The policy is an interface (`Policy.propose`) with two implementations:
`VllmPolicy` for inference and `ScriptedPolicy` for tests. The loop, the guard, the prompt
builder and the submission writer are exercised end to end on CPU with a scripted policy.

**Reason:** The pipeline is where most of the score is decided — the state gate, the argument
key set, the id grounding. Those are testable without a GPU, and a pipeline that only works
on a Colab GPU cannot be verified before the submission deadline. The scripted policy also
gives the decisive check: replaying the gold trajectory through the **real** loop scores
exactly `1.000000`, which isolates the pipeline from the model. The reference team measured
the same thing (99.99 on train) and it is the only way to know whether a low score is the
model's fault or ours.

**Alternatives:**
- *Test only the pieces.* Unit tests of the guard alone would not have caught the bug where
  the loop overwrote the dispatch list with the session's call log.
- *Run the real model on CPU.* Too slow, and it would need the weights locally.

**Tradeoff:** The scripted policy is a second implementation of the interface that must be
maintained. It is about twenty lines, and it buys a test suite that runs in under two
minutes without a GPU.

---

## Decision 20 — The adapter is merged explicitly before inference

**What was changed:** A separate `lifemate merge` step writes the merged model to disk, and
the inference cell points at that directory. `--model-path` names the merged model, and the
adapter's `adapter_config.json` is copied alongside it.

**Reason:** vLLM serves a merged model. Passing the adapter directory to `--checkpoint`
would load the *base* model — the adapter would be silently ignored, and the run would
score as though no training had happened at all. The failure is silent, which is the worst
kind: the submission would look plausible and be wrong.

**Alternatives:**
- *Merge inside `infer`.* Fewer steps, but every inference run would pay the merge cost,
  and the merged weights would have nowhere to live on the constrained Colab disk.
- *Serve the adapter with vLLM's LoRA support.* Possible, but it constrains the served
  graph and adds a runtime dependency to the offline image.

**Tradeoff:** One extra step in the notebook and an extra directory on Drive. In exchange,
the inference path has no silent way to ignore the trained weights, and the Docker image
bakes in exactly the model that was validated.

---

## Decision 21 — The grounding check seeds its ids from the conversation history

**What was changed:** The guard's observed-id set is seeded from the **history** turns'
`tool_results`, not only from the current turn's results.

**Reason:** Measured on the train split, 1,277 of the 1,278 gold x-ref values appear in
the conversation history, but only 964 appear in the *current* turn. The first version of
the guard tracked only current-turn results, so it would have rejected a gold call in
**254 of 1,545 samples** — a fifth of the split — for an id the model had legitimately
seen two turns earlier. That failure is silent: the model would be told the id was
unobserved and would answer "missing information", turning a correct call into a zero.

This was found by measuring the gold trajectories rather than by reading the code, and it
is pinned by `tests/test_pipeline.py::test_the_guard_grounds_ids_from_the_conversation_history`.

**Alternatives:**
- *Require an id to be observed in the current turn.* Strict, and wrong by 16% of the
  split.
- *Drop the grounding check entirely.* Measured cost: the guard is what stops a
  hallucinated id. Dispatching a wrong id does **not** error out — `sms.send` with a
  nonexistent `contact_id` returns `ok: true` and creates a message with a null phone
  number, silently corrupting the state and killing the sample on the hash gate.

**Tradeoff:** One gold sample in 1,545 (`train_1696`, whose gold uses a `chat_id` that does
not exist in the environment, so the gold call is itself a no-op) is rejected by the
guard. The full-split agent replay therefore scores **99.987** rather than 100.000. That
is the deliberate price of not letting a hallucinated id corrupt the state, and it is a
five-hundred-fold better trade than the 254-sample regression the strict variant caused.

---

## Decision 22 — Numbers and dates are repaired by the guard, targeted by declared type

**What was changed:** The guard gained a `normalize_values` step that runs before schema
validation. It repairs an argument **only when its declared type is numeric or
date-shaped**:

| Declared type | Repair |
|---|---|
| `integer` / `number` | spoken number → typed number: `bốn mươi phần trăm` → `40`, `hai triệu` → `2000000`, `2.000.000` → `2000000` |
| `string` + `format: date` | `27/9`, `ngày mười bốn` → `2026-09-27` |
| `string` + `format: date-time` | `2026-09-26 09:00` → `2026-09-26T09:00:00+07:00`; `chín giờ sáng mai` → the same |
| `string` + `format: time` | `bảy giờ tối` → `19:00` |

Free-text arguments are **never** touched: they are compared fuzzily (token-F1 > 0.6) and
the user's own wording is what scores. Dates are resolved against the **sample's**
`context.current_time`, never the wall clock, and the timezone offset is always the
sample's own.

**Reason:** Decision 15 said numbers and dates are resolved by code, but the code was
never wired into the loop — only `nfc` and the relative-day table were. Measured
consequences of leaving it out:

- The environment validates `format` strictly (`validate.py`), so a spoken value is a
  **schema error**: the call is rejected, the step is wasted, and the model has to try
  again.
- A date-time without the timezone offset is *accepted* but changes the **state hash**:
  replaying `2026-09-26T09:00:00` instead of `2026-09-26T09:00:00+07:00` produced a
  different hash on every sample checked (`train_0467`, `train_0471`, `train_0506`), so
  the sample scores **0**. The offset is not decoration.

Three properties were then measured on all 1,545 train samples
(`research/measure_normalizer.py`):

| Property | Result |
|---|---|
| Safety — gold arguments rewritten by the repair | **0 / 320 numeric, 0 / 117 date** |
| Repair — injected spelling defects restored to the gold value | **356 / 356** |
| Coverage — gold dates recoverable from the transcript by a self-contained phrase | **57 / 97** |

The 40 dates the resolver does *not* reach are all genuinely contextual: 14 are
anaphoric ("hôm đó", "hôm trước"), 17 have no date phrase in the utterance at all (the
date lives in the conversation history or in a referenced object), and 9 are relative to
another reference ("qua hạn sớm nhất một ngày"). Those belong to the model, and the
prompt now says so.

**Alternatives:**
- *Normalise the transcript before the model sees it.* Rejected: the model must copy the
  user's own words into free-text arguments, and rewriting "hai triệu" to "2000000" in
  the transcript would corrupt those arguments (token-F1 → 0).
- *Let the model do all of it.* Rejected: measured error costs (§12.3) — one perturbed
  non-fuzzy argument costs 0.213 mean and kills 268/400 samples on the state gate.
- *Normalise every argument regardless of type.* Rejected: it breaks the fuzzy free-text
  match and rewrites gold values, which is exactly the failure the safety measurement
  exists to prevent.

**Tradeoff:** A deterministic component that must be tested against Vietnamese
conventions. It is testable offline against the gold trajectories, which is the property
that makes it safe to trust. Two conventions are assumed rather than measured, and are
flagged as open: a day-of-month that has already passed rolls to the next month ("ngày 5"
said on the 25th), and a bare weekday refers to the next occurrence (measured on 13 of 14
gold cases; the exception is a past event, which is contextual).


---

## Decision 23 — Capitalisation is restored by a context model over the rules, not by more rules

**What was changed:** A new module `src/lifemate/case_model.py` layers a small
logistic-regression model on top of the rule module. It decides one thing per token —
whether that token's first character is uppercase — using the two words either side, the
rule module's own opinion, position in the sentence, and membership of the derived
vocabulary. The rule module keeps ownership of everything structural: internal capitals,
digits, punctuation, honorifics and product codes.

Two smaller changes came out of the same measurement:

- `capitalize.is_code()` recognises a product code whose letters the derived acronym set
  knows ("VOV3" from "vov", "RON95" from "ron", "E5" from "e", "RV183" from "rv"). The
  render path consults it, so a code keeps its caps instead of being lowercased.
- The model defers to the rules on honorifics and codes, where the decision is structural
  rather than contextual.

**Reason:** Decision 6 deferred the capitalisation question to "a name-aware recasing
pass" if the data turned out to have many non-fuzzy free-text fields. It does. Measured
over the training split, replacing each capitalised argument with its lowercase form
costs **9.258 points** — more than any single remaining source of loss.

The errors that survive the rule module are overwhelmingly *ambiguous* tokens: a word the
training transcripts spell in both cases ("Mai" the name against "mai" the month). A
vocabulary cannot settle those — the same word is right and wrong depending on the
sentence — but the neighbouring words can. All figures below are 5-fold cross-validated
over the whole training split, scored through the same code path the pipeline uses:

| Measurement | Rule module | With the context model |
|---|---|---|
| Metric score | 97.071 | **98.626** |
| Exactly-compared capitalised arguments recovered | 59.0% | **81.0%** |
| Arguments rescued / broken | — | 65 / 1 |

Two ablations establish that the gain is real rather than memorisation:

- **Removing the token's own identity** from the features still yields +0.789 points, so
  the gain comes from the context rather than from remembering train words.
- **Restricting to arguments whose words never appear capitalised in training** — the
  case that matters, because the public test uses contact names train never saw
  (531 of the 578 capitalised names in the public test environment are unknown to the
  train-derived table) — the model still improves them, from 4.3% to 13.8%.

**Alternatives:**
- *A larger vocabulary.* Rejected on measurement: the derived table already knows 324
  proper words, and the public test's names are largely outside it. A bigger vocabulary
  built from the same transcripts does not reach them.
- *A transformer or an embedding model over the sentence.* Rejected: the decision is
  local and per-token, the model is linear, and the artifact is 138 KiB of weights
  scored with dictionary lookups. There is no scikit-learn at inference time, and no
  GPU either.
- *Asking `is_acronym` in the render path.* Rejected on measurement: it accepts any
  unknown short uppercase word, so "giúp" became "GIÚP" — 6,974 tokens broken against
  13 fixed. The digit-bearing restriction broke none and fixed eleven.
- *Letting the model decide honorifics.* Rejected on measurement: the rules capitalise
  "Mẹ" only when a name follows it, which the model cannot reliably reproduce. All 17
  arguments it broke were honorifics or codes. Deferring on both cut that to 1 and
  raised the gain from 1.172 to 1.555 points.

**Tradeoff:** The artifact is a trained artefact that has to be rebuilt when the training
split changes, so `research/build_case_model.py` is the thing to run and
`research/verify_case_model.py` checks that the exported weights reproduce the
cross-validated score through the runtime path — currently 0 mismatches over 34,686
tokens. The model can also be wrong in a way the rules are not: it lowercases a name the
rules had right, which is why the deferral exists. And the public test is scored on
names the training split never saw, so the +1.555 points is an estimate from
cross-validation, not a guarantee.

---

## Decision 24 — The policy model casts the arguments; the recasing code leaves the pipeline

**What was changed:** The transcript now reaches the policy model in **lowercase**, and the
model is expected to write the capitalisation itself. The recasing that used to sit in the
ASR path — the rule module (`capitalize.py`) and the context model over it (`case_model.py`)
— is out of the runtime path. It is kept, because it is the baseline the ablation measures
against, and `casing.py` selects between the two settings in one place:

| mode | the transcript the model sees | who decides the case |
|---|---|---|
| `none` (default) | lowercase | the policy model |
| `rules` | restored by the rule module | the rule module |
| `model` | restored by the context model | the context model |

Four things changed with it:

- **The policy is applied where the text becomes model input** — `build_prompt`, for the
  request *and* for every history turn — so the training prompt and the inference prompt
  cannot disagree about it. Training used to show the *gold* transcript (correctly cased,
  punctuated) while inference showed a *recased* ASR transcript; both are lowercase now.
- **The SFT dataset uses the recogniser's output.** `TrainConfig.transcript_cache` points
  at the JSONL the `transcribe` command writes; `transcript_source` (`gold` / `asr` /
  `mix`) chooses it. The `transcript_source` key existed but was never wired — the trainer
  always used the gold transcript, so the model was never trained on the text it meets at
  inference time.
- **A safety net in the guard**: an argument that is *entirely lowercase* and matches an
  entity the environment already showed — in a tool result or in the conversation history —
  is snapped to the observed spelling ("mai anh" → "Mai Anh"). The snap only changes case,
  so it cannot break a fuzzy match, and it leaves an argument the model already capitalised
  alone.
- **A safety net in the prompt**: a small pretrained truecaser (`vinai/phobert-base`,
  135M parameters) contributes a *hint* — a recased copy of the request, marked as
  advisory. It is the fallback for names the model has never seen, and it is optional
  everywhere: without the weights the pipeline runs without it.

**Reason:** The recasing code can only settle a token whose spelling it has seen. The
public test's contact names are largely unknown to the train-derived table — 531 of the
578 capitalised names in the public test environment are outside it — so the context
model's gain is a cross-validation estimate, not a guarantee, and its failures land on
exactly the arguments that are compared with equality. The policy model is the one
component that already sits in the loop, it is trained on the gold arguments (which are
correctly cased), and it brings the pretrained knowledge of Vietnamese names that a
vocabulary cannot. Training it on the recogniser's lowercase output paired with the
cased gold arguments turns casing from a post-processing problem into a learned one.

**This decision is a hypothesis until it is measured.** The ablation is
`research/ablate_case_model_vs_sft.py`, three arms on the `train` split — A: the context
model recases (the baseline), B: the model casts, C: B plus the truecaser hint — each a
full train + infer + score cycle, scored on the exactly-compared capitalised arguments.
It needs a GPU, so it runs on Colab:

```
python research/ablate_case_model_vs_sft.py --arm B \
    --data-dir data/train+public_test \
    --transcripts /content/drive/transcripts_train.jsonl \
    --output-dir /content/drive/ablation
python research/ablate_case_model_vs_sft.py --summarize --output-dir /content/drive/ablation
```

If arm B loses to arm A, the recasing code comes back and this decision is reverted.

**Alternatives:**
- *Keep the recasing and add the model on top.* Rejected: the two would fight. The case
  model rewrites the transcript the policy model copies from, so the policy model would
  never learn to cast an argument, and its output would be overwritten in any case.
- *Feed the model the uppercase transcript.* Rejected: ALL CAPS tokenises poorly and
  destroys the word-boundary information the model needs to recognise a name.
- *A dedicated Vietnamese truecasing checkpoint.* None exists in a maintained
  pretrained form, so the hint is built on PhoBERT with pseudo-log-likelihood scoring
  instead of a fine-tuned checkpoint.
- *Snap every argument to an observed spelling.* Rejected: it would rewrite wording the
  model chose deliberately. The snap is limited to purely lowercase arguments, where the
  only thing missing is the capital.

**Tradeoff:** The pipeline now depends on the model getting the casing right, and the
guard can only repair a case when the environment happens to show the entity. The
transcripts are also less informative for the model: it can no longer copy the case from
the text, it has to know it. The recasing code stays in the repository, so the old
behaviour is one flag away (`--case-mode model`), and the truecaser costs 135M parameters
and a forward pass per word — it stays optional until the ablation says it pays.
