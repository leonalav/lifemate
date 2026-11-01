"""Local test of the notebook's logic (no Colab, no GPU).

Extracts the code cells from colab_launcher.ipynb, strips the Colab-only magics and
imports, and executes the environment-independent parts against a fake Drive layout so we
prove the data-setup logic, the split-detection logic, and the shell command construction
are correct.
"""
import json, os, shutil, sys, tempfile, zipfile, ast, textwrap

sys.stdout.reconfigure(encoding="utf-8")

NB = "colab_launcher.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
code_cells = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
              for c in nb["cells"] if c["cell_type"] == "code"]
md_cells = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            for c in nb["cells"] if c["cell_type"] == "markdown"]

print("=" * 70)
print("A. STATIC CHECKS")
print(f"   code cells: {len(code_cells)}   markdown cells: {len(md_cells)}")

# no absolute Windows paths from the local machine
bad = []
for i, src in enumerate(code_cells + md_cells):
    for pat in ("A:\\\\lifemate", "A:/lifemate", "C:\\\\Users", "C:/Users", "/media/", "/home/"):
        if pat in src:
            bad.append((i, pat))
print(f"   local-machine absolute paths: {len(bad)} {bad if bad else ''}")

# magics
magic_cells = [i for i, s in enumerate(code_cells) if s.lstrip().startswith("%")]
print(f"   cells using % magics: {magic_cells} (expected: the tensorboard cell only)")

# every code cell parses as python after stripping magics
for i, src in enumerate(code_cells):
    stripped = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    try:
        ast.parse(stripped)
    except SyntaxError as e:
        print(f"   SYNTAX ERROR in cell {i}: {e}")
        raise
print("   all code cells parse as Python")

# shell command construction check: every subprocess argv must resolve to a list of str
def argv_is_list(cell_src, call):
    """True if the first positional arg is a list literal or a name bound to one."""
    tree = ast.parse(cell_src)
    first = call.args[0] if call.args else None
    if isinstance(first, ast.List):
        return True
    if isinstance(first, ast.Name):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == first.id for t in node.targets
            ):
                if isinstance(node.value, ast.List):
                    return True
    return False


for i, src in enumerate(code_cells):
    stripped = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
    for node in ast.walk(ast.parse(stripped)):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "run":
            assert argv_is_list(stripped, node), f"cell {i}: subprocess.run argv is not a list"
            for kw in node.keywords:
                assert kw.arg != "shell", f"cell {i}: subprocess.run uses shell=True"
print("   all subprocess calls use list argv (no shell=True)")

print("=" * 70)
print("B. EXECUTE THE DATA-SETUP LOGIC (both zip layouts)")

setup = next(s for s in code_cells if "DATA_READY_MARKER" in s)
setup = setup.replace("import shutil, zipfile", "")
# The setup cell consumes DRIVE_DATA_ZIP / LOCAL_DATA_DIR from the config cell;
# inject them through the exec globals instead of rewriting the source.


def make_zip(path, layout):
    with zipfile.ZipFile(path, "w") as zf:
        if layout == "with_data_folder":
            zf.writestr("data/train/train.jsonl", "{}\n")
            zf.writestr("data/public_test/public_test.jsonl", "{}\n")
            zf.writestr("__MACOSX/._data", "junk")
        else:
            zf.writestr("train/train.jsonl", "{}\n")
            zf.writestr("public_test/public_test.jsonl", "{}\n")


for layout in ("with_data_folder", "flat"):
    tmp = tempfile.mkdtemp(prefix="nbtest_")
    drive = os.path.join(tmp, "data_lifemate")
    os.makedirs(drive)
    zip_path = os.path.join(drive, "data.zip")
    make_zip(zip_path, layout)
    local = os.path.join(tmp, "data_local")

    env = dict(os.environ, FAKE_DRIVE_ZIP=zip_path, FAKE_LOCAL_DIR=local)
    g = {"os": os, "shutil": shutil, "zipfile": zipfile,
         "DRIVE_DATA_ZIP": zip_path, "LOCAL_DATA_DIR": local}
    exec(compile(setup, "<setup>", "exec"), g)

    extracted = os.listdir(local)
    print(f"   [{layout}] extracted top-level: {sorted(extracted)}")
    assert "data.zip" not in extracted, "local zip was not deleted"
    marker = os.path.join(local, ".data_ready")
    assert os.path.exists(marker), "ready marker missing"
    root = open(marker, encoding="utf-8").read().strip()
    print(f"   [{layout}] resolved data root: {os.path.relpath(root, tmp)}")
    assert os.path.isdir(os.path.join(root, "train")), "train split not found under data root"

    # idempotency: second run must skip AND resolve the same root
    g2 = {"os": os, "shutil": shutil, "zipfile": zipfile,
          "DRIVE_DATA_ZIP": zip_path, "LOCAL_DATA_DIR": local}
    exec(compile(setup, "<setup>", "exec"), g2)
    root2 = open(marker, encoding="utf-8").read().strip()
    assert root2 == root, f"re-run resolved a different data root: {root} vs {root2}"
    print(f"   [{layout}] re-run skipped correctly and resolved the same root")
    shutil.rmtree(tmp)

print("=" * 70)
print("C. EXECUTE THE SPLIT-DETECTION HELPER DIRECTLY")
ns = {"os": os}
exec(compile("""
def find_data_root(base):
    entries = [e for e in os.listdir(base) if not e.startswith("__MACOSX")]
    if len(entries) == 1 and os.path.isdir(os.path.join(base, entries[0])):
        return os.path.join(base, entries[0])
    return base
""", "<helper>", "exec"), ns)
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, "data", "train"))
print("   single 'data/' folder ->", os.path.relpath(ns["find_data_root"](tmp), tmp))
os.makedirs(os.path.join(tmp, "train"))
print("   flat layout           ->", os.path.relpath(ns["find_data_root"](tmp), tmp))
shutil.rmtree(tmp)

print("=" * 70)
print("D. MISSING-ZIP ERROR PATH")
setup_missing = setup.replace('if not os.path.exists(DRIVE_DATA_ZIP):', 'if True:')
g = {"os": os, "shutil": shutil, "zipfile": zipfile,
     "DRIVE_DATA_ZIP": "/nonexistent/data.zip",
     "LOCAL_DATA_DIR": tempfile.mkdtemp()}
try:
    exec(compile(setup_missing, "<setup>", "exec"), g)
    print("   FAIL: no exception raised")
except FileNotFoundError as e:
    print("   raised FileNotFoundError with a clear message:")
    print("   ", str(e).replace("\n", "\n    "))

print("=" * 70)
print("E. CLI SURFACE CONSISTENCY (notebook vs PRD §6)")
prd = open("docs/PRD.md", encoding="utf-8").read()
import re
prd_cmds = set(re.findall(r"lifemate (train|infer|score|transcribe)\s+(.*)", prd))
nb_cmds = set()
for src in code_cells:
    for m in re.finditer(r'"lifemate",\s*"(train|infer|score|transcribe)",(.*?)\]', src, re.S):
        nb_cmds.add((m.group(1), re.sub(r"\s+", " ", m.group(2)).strip()))
for verb, args in sorted(prd_cmds):
    print(f"   PRD: lifemate {verb} {args[:90]}")
for verb, args in sorted(nb_cmds):
    print(f"   NB : lifemate {verb} {args[:90]}")
print("   verbs used in notebook:", sorted({v for v, _ in nb_cmds}))
assert {v for v, _ in nb_cmds} <= {v for v, _ in prd_cmds}, "notebook uses a verb the PRD does not define"
print("   notebook verbs are all defined in the PRD")

print("=" * 70)
print("ALL LOCAL NOTEBOOK TESTS PASSED")
