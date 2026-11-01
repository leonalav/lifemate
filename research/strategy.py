"""Strategic analysis for the PRD.

Q1. For what fraction of samples is the state_hash gate FREE (gold performs no state change)?
Q2. What does ask_missing_info look like in gold?
Q3. What is the ceiling if we always emit the gold calls (upper bound) vs. a realistic policy?
Q4. Which apps dominate the write workload (where errors cost the most)?
"""
import json, sys, collections
from pathlib import Path
ROOT = Path("data/train+public_test")
sys.path.insert(0, str(ROOT / "train"))
sys.stdout.reconfigure(encoding="utf-8")
from zaic_tools import ToolEnv, canon, state_hash

train = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8") if l.strip()]
cat = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
kind = {t["name"]: t["kind"] for a in cat["apps"] for t in a["tools"]}

# ---- Q1: gate-free samples ----
free = 0
writes_by_app = collections.Counter()
n_writes = collections.Counter()
for item in train:
    gold = item["label"]["tool_calls"]
    fs = item["label"].get("final_state") or {}
    if not fs:
        free += 1
    for c in gold:
        if kind.get(c["name"]) == "write":
            writes_by_app[c["name"].split(".")[0]] += 1
    n_writes[sum(1 for c in gold if kind.get(c["name"]) == "write")] += 1
print(f"Q1. samples whose gold leaves NO state change (gate free): {free}/{len(train)} = {free/len(train)*100:.1f}%")
print("    writes per sample dist:", dict(sorted(n_writes.items())))
print("    write calls by app:", writes_by_app.most_common(12))
print("    -> only these samples can fail the hash gate on a WRITE; all others fail only via wrong F1")

# ---- Q2: ask_missing_info shape ----
ex = [c for s in train for c in s["label"]["tool_calls"] if c["name"] == "sys.ask_missing_info"]
print(f"\nQ2. ask_missing_info gold calls: {len(ex)}")
keysets = collections.Counter(tuple(sorted(c["arguments"])) for c in ex)
for k, v in keysets.most_common(6):
    print("   arg keys", k, "->", v)
for c in ex[:3]:
    print("   ex:", json.dumps(c, ensure_ascii=False))
# is ask_missing_info always last?
last = sum(1 for s in train if s["label"]["tool_calls"] and s["label"]["tool_calls"][-1]["name"] == "sys.ask_missing_info")
print(f"   ask_missing_info is the LAST gold call in {last}/{len(ex)} cases")
# does it appear with writes before it?
both = sum(1 for s in train if any(c["name"] == "sys.ask_missing_info" for c in s["label"]["tool_calls"])
           and any(kind.get(c["name"]) == "write" for c in s["label"]["tool_calls"]))
print(f"   samples with ask_missing_info AND a write before it: {both}")

# ---- Q3: the read-only share of gold ----
ro = sum(1 for s in train if s["label"]["tool_calls"] and all(kind.get(c["name"]) == "read" for c in s["label"]["tool_calls"]))
print(f"\nQ3. samples whose gold is READ-ONLY: {ro} = {ro/len(train)*100:.1f}%")
mix = sum(1 for s in train if s["label"]["tool_calls"] and any(kind.get(c["name"]) == "read" for c in s["label"]["tool_calls"]))
print(f"    samples whose gold contains at least one read: {mix} = {mix/len(train)*100:.1f}%")
print("    -> reads are needed to obtain x-ref ids, but read-only samples can be satisfied by reads alone")

# ---- Q4: x-ref usage in gold ----
ref_tool = {}
for a in cat["apps"]:
    for t in a["tools"]:
        r = {k: v.get("x-ref") for k, v in t.get("parameters", {}).get("properties", {}).items() if v.get("x-ref")}
        if r:
            ref_tool[t["name"]] = r
gold_with_ref = 0
for s in train:
    if any(ref_tool.get(c["name"]) for c in s["label"]["tool_calls"]):
        gold_with_ref += 1
print(f"\nQ4. samples whose gold uses an x-ref argument: {gold_with_ref} = {gold_with_ref/len(train)*100:.1f}%")
print("    -> a wrong id here changes state => hash mismatch => 0 for the sample")

# ---- how many gold calls have NO arguments at all ----
noargs = sum(1 for s in train for c in s["label"]["tool_calls"] if not c.get("arguments"))
tot = sum(len(s["label"]["tool_calls"]) for s in train)
print(f"\nno-argument gold calls: {noargs}/{tot} = {noargs/tot*100:.1f}%")
