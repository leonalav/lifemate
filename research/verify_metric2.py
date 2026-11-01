"""Deeper metric characterisation: order sensitivity, extra-state-changing calls, arg key-set."""
import json, sys
from pathlib import Path
ROOT = Path("data/train+public_test")
sys.path.insert(0, str(ROOT / "train"))
sys.stdout.reconfigure(encoding="utf-8")
from zaic_tools import ToolEnv, canon

train = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8") if l.strip()]

def run(item, calls):
    env = ToolEnv("train", item["id"], root=ROOT)
    for c in calls:
        env.call(c["name"], c.get("arguments") or {})
    return env.get_state_hash()

n = 0
order_sensitive = 0
extra_call_zero = 0
extra_call_penalty = []
for item in train[:400]:
    gold = item["label"]["tool_calls"]
    if len(gold) < 2:
        continue
    n += 1
    h = run(item, gold)
    if run(item, list(reversed(gold))) != h:
        order_sensitive += 1
    # add a duplicate of the first write call with a changed arg -> changes state -> hash differs
    if gold:
        extra = [dict(c) for c in gold]
        e = dict(extra[0]); e["arguments"] = dict(e.get("arguments") or {})
        e["arguments"]["__bogus__"] = 1
        extra.append(e)
        h2 = run(item, extra)
        if h2 != h:
            extra_call_zero += 1
print(f"samples with >=2 gold calls: {n}")
print(f"order-sensitive (reversed order changes hash): {order_sensitive} = {order_sensitive/max(n,1)*100:.1f}%")
print(f"extra state-changing call changes hash: {extra_call_zero}/{n}")

# Which apps' writes actually affect the hash
from zaic_tools.utils import LIVE_APPS, FREE_TEXT_KEYS
import collections
affected = collections.Counter()
for item in train[:400]:
    for c in item["label"]["tool_calls"]:
        affected[c["name"].split(".")[0]] += 1
print("\napp share of gold calls:", affected.most_common(20))

# How many gold calls are reads vs writes
cat = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
kind = {t["name"]: t["kind"] for a in cat["apps"] for t in a["tools"]}
kc = collections.Counter(kind.get(c["name"], "?") for s in train for c in s["label"]["tool_calls"])
print("gold call kinds:", dict(kc))
# position of the first write
pos = collections.Counter()
for s in train:
    g = s["label"]["tool_calls"]
    if not g: continue
    for i, c in enumerate(g):
        if kind.get(c["name"]) == "write":
            pos[min(i, 3)] += 1
            break
print("index of first write call (3=3+):", dict(sorted(pos.items())))
