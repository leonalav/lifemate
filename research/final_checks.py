"""Final verification block: can we reproduce the official score locally, and what does the state gate cost?

F1. Does gold replay's state_hash equal the hash of label.final_state? (i.e. can we score offline without the gold hash?)
F2. For each sample: what is the score if F1 is perfect but state is wrong (gate) vs state right but F1 partial?
F3. Value of the read-before-write pattern: how many samples' gold contains a read that yields an id used later?
F4. Cost model for extra calls: which categories of extra call are cheap (no state change) vs fatal.
"""
import json, sys, collections, re
from pathlib import Path
ROOT = Path("data/train+public_test")
sys.path.insert(0, str(ROOT / "train"))
sys.stdout.reconfigure(encoding="utf-8")
from zaic_tools import ToolEnv, canon, state_hash

train = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8") if l.strip()]

# ---- F1: is label.final_state enough to derive the gold hash? ----
match = 0
for item in train:
    env = ToolEnv("train", item["id"], root=ROOT)
    for c in item["label"]["tool_calls"]:
        env.call(c["name"], c.get("arguments") or {})
    fs = item["label"].get("final_state", {})
    got = {a: st for a, st in env.get_state().items() if a in fs}
    if canon(got) == canon(fs):
        match += 1
print(f"F1. gold replay state == label.final_state: {match}/{len(train)}")
print("    => label.final_state IS the gold state; the offline scorer needs no secret gold hash.")

# ---- F2: the gate is a hard multiplier ----
# score model: 0 if hash mismatch else F1 (non-empty gold); 1 if gold empty and submitted empty
def f1_of(pred, gold):
    if not gold:
        return 1.0 if not pred else 0.0
    def fz(name):
        return set()
    M = 0
    used = set()
    for p in pred:
        for j, g in enumerate(gold):
            if j in used: continue
            if p["name"] == g["name"]:
                M += 1; used.add(j); break
    U = len(pred) - M
    P = M / (M + U) if (M + U) else 0.0
    R = M / len(gold)
    return 2 * P * R / (P + R) if (P + R) else 0.0

# what does an empty submission score?
z = sum(f1_of([], item["label"]["tool_calls"]) for item in train)
print(f"F2. empty submission mean score: {z/len(train):.4f}  (only out-of-scope samples score)")

# ---- F3: read-before-write ----
need_read = 0
for item in train:
    g = item["label"]["tool_calls"]
    blob = json.dumps(item["label"].get("tool_results"), ensure_ascii=False)
    ids = set()
    has_write_after = False
    for c in g:
        args = c.get("arguments") or {}
        if any(isinstance(v, str) and re.search(r"^[a-z]+_[a-z0-9_]+$", v) for v in args.values()):
            has_write_after = True
    if has_write_after:
        need_read += 1
print(f"F3. samples whose gold carries an id-shaped arg (needs a prior read): {need_read}/{len(train)} = {need_read/len(train)*100:.1f}%")

# ---- F4: cost of extra calls ----
cat = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
kind = {t["name"]: t["kind"] for a in cat["apps"] for t in a["tools"]}
read_only_gold = sum(1 for item in train if item["label"]["tool_calls"] and all(kind.get(c["name"]) == "read" for c in item["label"]["tool_calls"]))
print(f"F4. gold is read-only for {read_only_gold} samples -> any extra call there is free on the gate, costly on F1")

# ---- F5: how much of the score is reachable by getting the FIRST call right ----
first_tool_correct = 0
first_full_correct = 0
for item in train:
    g = item["label"]["tool_calls"]
    if not g: continue
    first_tool_correct += 1
print(f"F5. samples with a non-empty gold: {len(train)-sum(1 for i in train if not i['label']['tool_calls'])}")
