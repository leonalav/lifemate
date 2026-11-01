"""Deep analysis of the LifeMate kit for PRD grounding. Read-only."""
import json, collections, statistics, re, os, sys

ROOT = "data/train+public_test"
sys.stdout.reconfigure(encoding="utf-8")

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

train = load(f"{ROOT}/train/train.jsonl")
test = load(f"{ROOT}/public_test/public_test.jsonl")
cat_tr = json.load(open(f"{ROOT}/train/train_tools_catalog.json", encoding="utf-8"))
cat_te = json.load(open(f"{ROOT}/public_test/public_test_tools_catalog.json", encoding="utf-8"))

print("=" * 70)
print("SPLIT SIZES")
print(f"train={len(train)} public_test={len(test)}")
print("train keys:", sorted(train[0].keys()))
print("test keys:", sorted(test[0].keys()))

print("=" * 70)
print("CATALOG")
for name, cat in (("train", cat_tr), ("public_test", cat_te)):
    apps = cat["apps"]
    ntools = sum(len(a["tools"]) for a in apps)
    print(f"{name}: schema_version={cat.get('schema_version')} apps={len(apps)} tools={ntools}")
    print("  catalog top keys:", sorted(cat.keys()))
print("sample app object:", json.dumps(cat_tr["apps"][0], ensure_ascii=False)[:900])

# tool kinds
kinds = collections.Counter()
for cat in (cat_tr, cat_te):
    for a in cat["apps"]:
        for t in a["tools"]:
            kinds[t["kind"]] += 1
print("tool kinds:", dict(kinds))

# extensions
ext = collections.Counter()
fuzzy_by_tool = collections.Counter()
ref_by_tool = collections.Counter()
oneof_tools = []
param_types = collections.Counter()
for cat in (cat_tr,):
    for a in cat["apps"]:
        for t in a["tools"]:
            props = t.get("parameters", {}).get("properties", {})
            f = [k for k, v in props.items() if v.get("x-match") == "fuzzy"]
            r = [k for k, v in props.items() if v.get("x-ref")]
            if f:
                fuzzy_by_tool[t["name"]] = len(f)
            if r:
                ref_by_tool[t["name"]] = len(r)
            if "x-one-of" in t.get("parameters", {}):
                oneof_tools.append(t["name"])
            for k, v in props.items():
                param_types[v.get("type")] += 1
print("tools with fuzzy args:", len(fuzzy_by_tool), "of", sum(len(a['tools']) for a in cat_tr['apps']))
print("tools with x-ref args:", len(ref_by_tool))
print("tools with x-one-of:", len(oneof_tools), oneof_tools[:12])
print("param types:", dict(param_types))
print("note:", cat_tr.get("note"))

print("=" * 70)
print("INSTALLED APPS / TOOL SET PER SAMPLE")
def apps_of(s):
    return set(s["context"]["installed_apps"].keys())
tr_apps = collections.Counter()
for s in train:
    for a in apps_of(s):
        tr_apps[a] += 1
te_apps = collections.Counter()
for s in test:
    for a in apps_of(s):
        te_apps[a] += 1
print("train distinct apps:", len(tr_apps))
print("public_test distinct apps:", len(te_apps))
unseen = set(te_apps) - set(tr_apps)
print("APPS IN PUBLIC_TEST NOT IN TRAIN (%d):" % len(unseen), sorted(unseen))
print("apps in train not in public_test:", sorted(set(tr_apps) - set(te_apps)))
print("apps count per sample: train mean %.1f median %d max %d" % (
    statistics.mean(len(apps_of(s)) for s in train),
    statistics.median(len(apps_of(s)) for s in train),
    max(len(apps_of(s)) for s in train)))
n_tools = [sum(len(v["tools"]) for v in s["context"]["installed_apps"].values()) for s in train]
print("tools available per sample: mean %.1f median %d max %d" % (statistics.mean(n_tools), statistics.median(n_tools), max(n_tools)))
print("top apps by sample count:", tr_apps.most_common(15))

print("=" * 70)
print("LABEL ANALYSIS (train)")
n_calls = collections.Counter()
first_tool = collections.Counter()
out_of_scope = 0
ask_missing = 0
ask_first = 0
apps_used = collections.Counter()
fuzzy_used = collections.Counter()
arg_keys = collections.Counter()
for s in train:
    tc = s["label"]["tool_calls"]
    n_calls[len(tc)] += 1
    if not tc:
        out_of_scope += 1
    else:
        first_tool[tc[0]["name"]] += 1
        apps_used[tc[0]["name"].split(".")[0]] += 1
    if any(c["name"] == "sys.ask_missing_info" for c in tc):
        ask_missing += 1
        if tc and tc[0]["name"] == "sys.ask_missing_info":
            ask_first += 1
print("calls-per-sample dist:", dict(sorted(n_calls.items())))
print(f"out-of-scope (empty gold): {out_of_scope} = {out_of_scope/len(train)*100:.1f}%")
print(f"samples containing ask_missing_info: {ask_missing} = {ask_missing/len(train)*100:.1f}%")
print(f"ask_missing_info as FIRST call: {ask_first} = {ask_first/len(train)*100:.1f}%")
print("first-call app distribution:", apps_used.most_common(20))
print("total distinct tools used as first call:", len(first_tool))
print("top first tools:", first_tool.most_common(15))

# multi-call breakdown
multi = sum(v for k, v in n_calls.items() if k >= 2)
print(f"multi-call samples: {multi} = {multi/len(train)*100:.1f}%")

print("=" * 70)
print("ARGUMENT KEY SET ANALYSIS")
# how often does gold use the exact required+optional key set?
cat_tool = {t["name"]: t for a in cat_tr["apps"] for t in a["tools"]}
extra = 0
missing_req = 0
tot = 0
for s in train:
    for c in s["label"]["tool_calls"]:
        spec = cat_tool.get(c["name"])
        if not spec:
            continue
        tot += 1
        props = set(spec["parameters"].get("properties", {}))
        req = set(spec["parameters"].get("required", []))
        got = set(c["arguments"])
        if got - props:
            extra += 1
        if req - got:
            missing_req += 1
print(f"gold calls with args not in schema: {extra}/{tot}")
print(f"gold calls missing a required arg: {missing_req}/{tot}")

# fuzzy usage in gold
fz = collections.Counter()
for s in train:
    for c in s["label"]["tool_calls"]:
        spec = cat_tool.get(c["name"])
        if not spec: continue
        props = spec["parameters"].get("properties", {})
        for k, v in c["arguments"].items():
            if isinstance(v, str) and props.get(k, {}).get("x-match") == "fuzzy":
                fz[k] += 1
print("fuzzy args used in gold:", fz.most_common(15))

print("=" * 70)
print("HISTORY")
h = collections.Counter(len(s["history"]) for s in train)
print("history turn counts:", dict(sorted(h.items())))
has_audio_text_hist = sum(1 for s in train for turn in s["history"] if turn.get("role") == "user" and turn.get("audio_text"))
print("user history turns with audio_text:", has_audio_text_hist)

print("=" * 70)
print("CONTEXT")
tz = collections.Counter(s["context"].get("timezone") for s in train)
print("timezones:", dict(tz))
cities = collections.Counter(s["context"].get("user_profile", {}).get("city") for s in train)
print("cities:", cities.most_common(12))
print("context keys:", sorted(train[0]["context"].keys()))
print("user_profile keys:", sorted(train[0]["context"]["user_profile"].keys()))
print("location keys:", sorted(train[0]["context"]["location"].keys()))

print("=" * 70)
print("AUDIO")
import wave
tot_dur = []
n_wav = 0
for d in sorted(os.listdir(f"{ROOT}/train/audio"))[:200]:
    p = f"{ROOT}/train/audio/{d}"
    for f in os.listdir(p):
        if f.endswith(".wav"):
            n_wav += 1
            with wave.open(os.path.join(p, f)) as w:
                tot_dur.append(w.getnframes() / w.getframerate())
                if n_wav == 1:
                    print("sample wav:", f, "rate", w.getframerate(), "ch", w.getnchannels(), "frames", w.getnframes())
print(f"audio files in first 200 dirs: {n_wav}, mean dur {statistics.mean(tot_dur):.2f}s median {statistics.median(tot_dur):.2f}s max {max(tot_dur):.2f}s p95 {sorted(tot_dur)[int(len(tot_dur)*.95)]:.2f}s")

print("=" * 70)
print("TEXT LENGTH")
al = [len(s["audio_text"]) for s in train]
print("audio_text chars: mean %.0f median %d p95 %d max %d" % (statistics.mean(al), statistics.median(al), sorted(al)[int(len(al)*.95)], max(al)))
al_te = [len(s.get("audio", "")) for s in test]
print("public_test has audio_text?", "audio_text" in test[0])

print("=" * 70)
print("SAMPLE SUBMISSION")
print(open(f"{ROOT}/public_test/sample_submission.jsonl", encoding="utf-8").read())
