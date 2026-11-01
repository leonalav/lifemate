"""Authoritative local scorer (per-tool fuzzy sets) + degradation study.

This is the reference implementation the PRD's numbers are quoted from.
Fuzzy-ness is a property of the ARGUMENT within a specific TOOL, so the
fuzzy set must be selected per (tool, argument) — not pooled across tools.
"""
import json, sys, re, collections, unicodedata
from pathlib import Path

ROOT = Path("data/train+public_test")
sys.path.insert(0, str(ROOT / "train"))
sys.stdout.reconfigure(encoding="utf-8")
from zaic_tools import ToolEnv

train = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8") if l.strip()]
cat = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))

FUZZY = {t["name"]: {k for k, v in t.get("parameters", {}).get("properties", {}).items()
                    if v.get("x-match") == "fuzzy"}
         for a in cat["apps"] for t in a["tools"]}
SPEC = {t["name"]: t for a in cat["apps"] for t in a["tools"]}


def _norm(s):
    return unicodedata.normalize("NFC", str(s).lower())


def _tok(s):
    return collections.Counter(re.findall(r"[^\W_]+", _norm(s)))


def token_f1(a, b):
    ta, tb = _tok(a), _tok(b)
    if not ta or not tb:
        return 1.0 if _norm(a) == _norm(b) else 0.0
    common = sum((ta & tb).values())
    if common == 0:
        return 0.0
    p, r = common / sum(ta.values()), common / sum(tb.values())
    return 2 * p * r / (p + r)


def calls_match(pred, gold):
    """Per-tool, per-argument: exact key set; fuzzy args by token-F1 > 0.6."""
    if pred["name"] != gold["name"]:
        return False
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    if set(pa) != set(ga):
        return False
    fz = FUZZY.get(gold["name"], set())
    for k, gv in ga.items():
        pv = pa[k]
        if k in fz:
            if not (isinstance(pv, str) and isinstance(gv, str) and token_f1(pv, gv) > 0.6):
                return False
        else:
            if type(pv) is not type(gv) or pv != gv:
                return False
    return True


def max_bipartite(preds, golds):
    """Kuhn's algorithm for maximum bipartite matching (one call matches at most one)."""
    adj = [[j for j in range(len(golds)) if calls_match(p, golds[j])] for p in preds]
    match_gold = [-1] * len(golds)

    def augment(i, seen):
        for j in adj[i]:
            if seen[j]:
                continue
            seen[j] = True
            if match_gold[j] == -1 or augment(match_gold[j], seen):
                match_gold[j] = i
                return True
        return False

    m = 0
    for i in range(len(preds)):
        if augment(i, [False] * len(golds)):
            m += 1
    return m


def sample_score(pred_calls, gold_calls, pred_hash, gold_hash):
    """Official formula: gold empty -> 1 iff pred empty; else state gate then F1."""
    if not gold_calls:
        return 1.0 if not pred_calls else 0.0
    if pred_hash != gold_hash:
        return 0.0
    m = max_bipartite(pred_calls, gold_calls)
    u = len(pred_calls) - m
    p = m / (m + u) if (m + u) else 0.0
    r = m / len(gold_calls)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def gold_hash(item):
    env = ToolEnv("train", item["id"], root=ROOT)
    for c in item["label"]["tool_calls"]:
        env.call(c["name"], c.get("arguments") or {})
    return env.get_state_hash()


if __name__ == "__main__":
    print("=" * 72)
    print("A. GOLD REPLAY over the FULL train split")
    tot = 0.0
    perfect = 0
    for item in train:
        env = ToolEnv("train", item["id"], root=ROOT)
        for c in item["label"]["tool_calls"]:
            env.call(c["name"], c.get("arguments") or {})
        s = sample_score(env.get_tool_calls(), item["label"]["tool_calls"],
                         env.get_state_hash(), env.get_state_hash())
        tot += s
        perfect += (s == 1.0)
    print(f"   mean = {tot/len(train):.6f}   perfect = {perfect}/{len(train)}")

    tot = sum(sample_score([], i["label"]["tool_calls"], "x", "y") for i in train)
    print(f"   empty submission mean = {tot/len(train):.6f} = {tot/len(train)*100:.2f}/100")

    print("=" * 72)
    print("B. DEGRADATION STUDY (n=400, state gate ON, per-tool fuzzy)")
    N = 400

    def evaluate(mutate, label):
        tot = 0.0
        gate_zero = 0
        for item in train[:N]:
            gold = item["label"]["tool_calls"]
            gh = gold_hash(item)
            calls = mutate([dict(c, arguments=dict(c.get("arguments") or {})) for c in gold], item)
            env = ToolEnv("train", item["id"], root=ROOT)
            for c in calls:
                env.call(c["name"], c.get("arguments") or {})
            s = sample_score(env.get_tool_calls(), gold, env.get_state_hash(), gh)
            tot += s
            gate_zero += (env.get_state_hash() != gh and bool(gold))
        print(f"   {label:<44} mean={tot/N:.4f}   gate-killed={gate_zero}/{N}")

    evaluate(lambda g, i: g, "gold (control)")
    evaluate(lambda g, i: g[:-1] if g else g, "drop last call")
    evaluate(lambda g, i: g + [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "x.y", "params": ["z"]}}],
             "add one extra meta call")
    evaluate(lambda g, i: list(reversed(g)), "reverse call order")

    def drop_optional(g, i):
        for c in g:
            spec = SPEC.get(c["name"])
            if not spec:
                continue
            req = set(spec["parameters"].get("required", []))
            props = set(spec["parameters"].get("properties", {}))
            for k in list(c["arguments"]):
                if k not in req and k in props:
                    del c["arguments"][k]
                    break
        return g
    evaluate(drop_optional, "drop one optional argument")

    def drop_required(g, i):
        for c in g:
            spec = SPEC.get(c["name"])
            if not spec:
                continue
            req = spec["parameters"].get("required", [])
            for k in req:
                if k in c["arguments"]:
                    del c["arguments"][k]
                    break
        return g
    evaluate(drop_required, "drop one REQUIRED argument (schema reject)")

    def wrong_first_tool(g, i):
        if not g:
            return g
        g[0] = dict(g[0])
        g[0]["name"] = "weather.get_current" if g[0]["name"] != "weather.get_current" else "weather.get_forecast"
        return g
    evaluate(wrong_first_tool, "wrong first tool name")

    def perturb_fuzzy(g, i):
        for c in g:
            fz = FUZZY.get(c["name"], set())
            for k in list(c["arguments"]):
                if k in fz and isinstance(c["arguments"][k], str):
                    c["arguments"][k] = c["arguments"][k] + " xyz"
                    return g
        return g
    evaluate(perturb_fuzzy, "perturb one fuzzy arg (token-F1 test)")

    def perturb_nonfuzzy(g, i):
        for c in g:
            for k in list(c["arguments"]):
                if k not in FUZZY.get(c["name"], set()):
                    v = c["arguments"][k]
                    if isinstance(v, str):
                        c["arguments"][k] = v + "x"
                        return g
                    if isinstance(v, int):
                        c["arguments"][k] = v + 1
                        return g
        return g
    evaluate(perturb_nonfuzzy, "perturb one non-fuzzy arg")

    def type_swap(g, i):
        for c in g:
            for k in list(c["arguments"]):
                if k not in FUZZY.get(c["name"], set()):
                    v = c["arguments"][k]
                    if isinstance(v, int):
                        c["arguments"][k] = str(v)
                        return g
        return g
    evaluate(type_swap, "int arg as string (type-sensitive)")

    print("=" * 72)
    print("C. FUZZY-KEY SENSITIVITY (token-F1 on the boundary)")
    for a, b in [("mai chạy bộ dời sang 09:00 nhé", "mai chạy bộ dời sang 09:00 nhé"),
                 ("Mai Anh", "mai anh"),
                 ("Hương đi học", "huong di hoc"),
                 ("Bật điều hòa", "bật điều hoà"),
                 ("abc def", "abc xyz")]:
        print(f"   token_f1({a!r}, {b!r}) = {token_f1(a,b):.3f}  -> {'MATCH' if token_f1(a,b)>0.6 else 'NO MATCH'}")
