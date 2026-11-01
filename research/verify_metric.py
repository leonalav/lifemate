"""Verify metric fidelity: replay gold tool_calls through the real zaic_tools and score.

Checks:
 1. gold replay -> every sample scores 100 (state hash matches, F1 = 1)
 2. what the state_hash actually depends on (free-text keys dropped, LIVE_APPS dropped)
 3. the effect of dropping a call, adding a call, reordering calls, and perturbing arguments
"""
import json, sys, re, unicodedata, collections
from pathlib import Path

ROOT = Path("data/train+public_test")
sys.path.insert(0, str(ROOT / "train"))
sys.stdout.reconfigure(encoding="utf-8")
from zaic_tools import ToolEnv, state_hash, canon, FREE_TEXT_KEYS, LIVE_APPS

train = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8") if l.strip()]

# ---------- metric implementation (reconstructed per the Evaluation tab) ----------
def _norm(s):
    return unicodedata.normalize("NFC", str(s).lower())

def _tokens(s):
    return collections.Counter(re.findall(r"[^\W_]+", _norm(s)))

def token_f1(a, b):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if _norm(a) == _norm(b) else 0.0
    common = sum((ta & tb).values())
    if common == 0:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)

def calls_match(pred, gold, fuzzy_keys):
    if pred["name"] != gold["name"]:
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    for k, gv in ga.items():
        pv = pa[k]
        if k in fuzzy_keys:
            if not (isinstance(pv, str) and isinstance(gv, str) and token_f1(pv, gv) > 0.6):
                return False
        else:
            if type(pv) is not type(gv) or pv != gv:
                return False
    return True

def bipartite_max_match(preds, golds, fuzzy_keys):
    """Kuhn's algorithm for maximum bipartite matching."""
    adj = [[j for j in range(len(golds)) if calls_match(p, golds[j], fuzzy_keys)] for p in preds]
    match_g = [-1] * len(golds)
    def try_k(i, seen):
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_g[j] == -1 or try_k(match_g[j], seen):
                match_g[j] = i
                return True
        return False
    n = 0
    for i in range(len(preds)):
        if try_k(i, [False] * len(golds)):
            n += 1
    return n

def score_sample(pred_calls, gold_calls, pred_hash, gold_hash, fuzzy_keys):
    if not gold_calls:
        return 1.0 if not pred_calls else 0.0
    if pred_hash != gold_hash:
        return 0.0
    M = bipartite_max_match(pred_calls, gold_calls, fuzzy_keys)
    U = len(pred_calls) - M
    P = M / (M + U) if (M + U) else 0.0
    R = M / len(gold_calls)
    return 2 * P * R / (P + R) if (P + R) else 0.0

# ---------- fuzzy keys per tool ----------
cat = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
fuzzy = {}
for a in cat["apps"]:
    for t in a["tools"]:
        props = t.get("parameters", {}).get("properties", {})
        fz = {k for k, v in props.items() if v.get("x-match") == "fuzzy"}
        if fz:
            fuzzy[t["name"]] = fz
print("fuzzy tools:", len(fuzzy))

# ---------- 1. gold replay ----------
N = 400
s = train[:N]
gold_hash_ok = 0
scores = []
final_state_ok = 0
for item in s:
    env = ToolEnv("train", item["id"], root=ROOT)
    gold = item["label"]["tool_calls"]
    for c in gold:
        env.call(c["name"], c.get("arguments") or {})
    h = env.get_state_hash()
    # gold final_state from the label, canonicalised
    gold_fs = item["label"].get("final_state", {})
    got = env.get_state()
    changed = {a: st for a, st in got.items() if st != json.load(open(ROOT / f"train/train_environment/{item['id']}/state.json", encoding="utf-8")).get(a)}
    # compare via canon
    if canon({a: st for a, st in got.items() if a in gold_fs}) == canon(gold_fs):
        final_state_ok += 1
    scores.append(score_sample(env.get_tool_calls(), gold, h, h, fuzzy))
print(f"[1] gold replay: mean sample score = {sum(scores)/len(scores):.6f}  perfect={sum(1 for x in scores if x==1.0)}/{len(scores)}")
print(f"    gold final_state reproduced (canon): {final_state_ok}/{len(scores)}")

# ---------- 2. what the hash depends on ----------
item = train[0]
env = ToolEnv("train", item["id"], root=ROOT)
for c in item["label"]["tool_calls"]:
    env.call(c["name"], c.get("arguments") or {})
h_base = env.get_state_hash()
print(f"\n[2] state_hash sensitivity, sample {item['id']}")
print("   FREE_TEXT_KEYS:", sorted(FREE_TEXT_KEYS))
print("   LIVE_APPS (dropped from hash):", sorted(LIVE_APPS))
# perturb a fuzzy arg
gold = item["label"]["tool_calls"]
fz_call = next((c for c in gold if fuzzy.get(c["name"]) and set(c["arguments"]) & fuzzy[c["name"]]), None)
if fz_call:
    env2 = ToolEnv("train", item["id"], root=ROOT)
    for c in gold:
        a = dict(c.get("arguments") or {})
        if c is fz_call:
            for k in a:
                if k in fuzzy[c["name"]]:
                    a[k] = a[k] + " xyz"
        env2.call(c["name"], a)
    print(f"   perturbing a fuzzy arg changes hash? {env2.get_state_hash() != h_base}  (expect False)")
# reorder
env3 = ToolEnv("train", item["id"], root=ROOT)
for c in reversed(gold):
    env3.call(c["name"], c.get("arguments") or {})
print(f"   reversing call order changes hash? {env3.get_state_hash() != h_base}")

# ---------- 3. degraded variants ----------
def evaluate(mutate, label, n=200):
    tot = 0.0
    for item in train[:n]:
        env = ToolEnv("train", item["id"], root=ROOT)
        gold = item["label"]["tool_calls"]
        calls = mutate(list(gold), item)
        for c in calls:
            env.call(c["name"], c.get("arguments") or {})
        tot += score_sample(env.get_tool_calls(), gold, env.get_state_hash(), h_of(item, gold), fuzzy)
    print(f"   {label:<42} mean={tot/n:.4f}")

def h_of(item, gold):
    env = ToolEnv("train", item["id"], root=ROOT)
    for c in gold:
        env.call(c["name"], c.get("arguments") or {})
    return env.get_state_hash()

print("\n[3] degraded variants (n=200)")
evaluate(lambda g, i: g, "gold (control)")
evaluate(lambda g, i: g[:-1] if g else g, "drop last call")
evaluate(lambda g, i: g + [{"name": "sys.ask_missing_info", "arguments": {}}], "add one extra call")
evaluate(lambda g, i: list(reversed(g)), "reverse order")
def drop_opt(g, i):
    out = []
    for c in g:
        c = dict(c); args = dict(c.get("arguments") or {})
        spec = next((t for a in cat["apps"] for t in a["tools"] if t["name"] == c["name"]), None)
        if spec:
            req = set(spec["parameters"].get("required", []))
            for k in list(args):
                if k not in req:
                    del args[k]; break
        c["arguments"] = args
        out.append(c)
    return out
evaluate(drop_opt, "drop one optional arg")
def wrong_tool(g, i):
    if not g: return g
    out = [dict(c) for c in g]
    out[0] = dict(out[0]); out[0]["name"] = "weather.get_current" if out[0]["name"] != "weather.get_current" else "weather.get_forecast"
    return out
evaluate(wrong_tool, "wrong first tool name")
