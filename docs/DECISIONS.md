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
