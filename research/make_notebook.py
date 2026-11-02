"""Generate colab_launcher.ipynb.

The notebook calls the CLI contract frozen in docs/PRD.md §6:
    python -m lifemate train  --data-dir ... --output-dir ... --config ... [--resume] [--epochs] [--batch-size] [--smoke]
    python -m lifemate infer  --data-dir ... --split ... --checkpoint ... --output-file ... [--limit] [--max-steps] [--time-budget-s]
    python -m lifemate score  --split train --submission FILE [--ids-file] [--report]
"""
import json, sys

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md("""
# LifeMate — Colab Launcher

Runs the LifeMate training and inference pipeline on a Colab GPU.

**Runtime → Change runtime type → GPU** (T4 minimum, L4 or A100 preferred).

| What runs where | Location |
|---|---|
| Cloned repository | `/content/lifemate` (local disk — wiped on every session reset) |
| Competition dataset | `/content/data_local` (local disk — copied from Drive, unzipped) |
| Checkpoints, logs, inference outputs | Google Drive (`/content/drive/MyDrive/lifemate_runs`) |

Nothing is written back into the repository except the notebook's own outputs.
Re-running any cell is safe: the clone cell updates, the data cell skips, training resumes.
""")

md("""
## 1. GPU check
""")

code("""
import subprocess, sys

try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout)
except FileNotFoundError:
    print("nvidia-smi not found")

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA GPU detected. Go to Runtime > Change runtime type > GPU, "
        "then re-run this cell from the top."
    )

print(f"torch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
""")

md("""
## 2. Mount Google Drive
""")

code("""
from google.colab import drive

drive.mount("/content/drive")

import os

DRIVE_ROOT = "/content/drive/MyDrive"
print("Drive mounted at /content/drive")
print("MyDrive contents:", os.listdir(DRIVE_ROOT)[:20])
""")

md("""
## 3. Configuration

**Edit the values in this cell only.** Every later cell reads from these variables.
""")

code("""
import os

# --- repository ---
REPO_URL = "https://github.com/leonalav/lifemate.git"
BRANCH = "main"
REPO_DIR = "/content/lifemate"

# --- data ---
DRIVE_DATA_ZIP = "/content/drive/MyDrive/data_lifemate/data.zip"
LOCAL_DATA_DIR = "/content/data_local"

# --- run ---
RUN_NAME = "lifemate-qwen35-4b-qlora"
OUTPUT_DIR = f"/content/drive/MyDrive/lifemate_runs/{RUN_NAME}"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"
LOG_DIR = f"{OUTPUT_DIR}/logs"
INFER_OUTPUT_DIR = f"{OUTPUT_DIR}/inference"

# --- training ---
EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACC = 8
LORA_R = 64
LORA_ALPHA = 128
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 8192
SEED = 20261028

# --- inference ---
SPLIT = "public_test"
MAX_STEPS = 8
TIME_BUDGET_S = 25

# The ASR bundle (gipformer ONNX) is downloaded from Hugging Face on first use and
# cached here, so a session reset does not re-download it.
ASR_MODEL_DIR = "/content/drive/MyDrive/lifemate_runs/models/gipformer"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)

print("REPO_URL        :", REPO_URL)
print("BRANCH          :", BRANCH)
print("DATA ZIP (Drive):", DRIVE_DATA_ZIP)
print("LOCAL DATA DIR  :", LOCAL_DATA_DIR)
print("RUN NAME        :", RUN_NAME)
print("OUTPUT DIR      :", OUTPUT_DIR)
print("EPOCHS/BATCH    :", EPOCHS, "/", BATCH_SIZE, "grad_acc", GRAD_ACC)
print("SPLIT           :", SPLIT)
""")

md("""
## 4. Secrets

Loads optional tokens from Colab's secret manager (key icon in the sidebar) into the
environment. Missing secrets are ignored — nothing is printed.
""")

code("""
from google.colab import userdata

for key in ("HF_TOKEN", "WANDB_API_KEY"):
    try:
        value = userdata.get(key)
        os.environ[key] = value
        print(f"{key}: loaded ({len(value)} chars)")
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError, KeyError):
        print(f"{key}: not set (continuing without it)")
    except Exception:
        print(f"{key}: unavailable (continuing without it)")

# Never echo the values themselves.
""")

md("""
## 5. Clone or update the repository

Clones on first run; on later runs it fetches, checks out the branch, and pulls.
""")

code("""
if os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print("Repository exists — updating.")
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--all", "--tags"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only", "origin", BRANCH], check=True)
else:
    print("Cloning repository.")
    os.makedirs(REPO_DIR, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, REPO_DIR],
        check=True,
    )

subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], check=True)
print("HEAD:", subprocess.run(
    ["git", "-C", REPO_DIR, "log", "-1", "--format=%h %s"],
    capture_output=True, text=True).stdout.strip())
""")

md("""
## 6. Install dependencies

`requirements.txt` deliberately does **not** pin `torch`, `torchvision`, `tensorflow` or `jax` —
Colab already provides a working CUDA build, and reinstalling them is slow and error-prone.

**Restart the runtime only if this cell reports that a core package (for example `numpy`)
was upgraded.** Otherwise no restart is needed.
""")

code("""
requirements = os.path.join(REPO_DIR, "requirements.txt")
if not os.path.exists(requirements):
    raise FileNotFoundError(f"requirements.txt not found at {requirements}")

before = {}
for pkg in ("numpy", "torch"):
    try:
        before[pkg] = __import__(pkg).__version__
    except Exception:
        before[pkg] = None

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", requirements],
    cwd=REPO_DIR,
)
print("pip exit code:", result.returncode)

# The package itself, in editable mode, so `python -m lifemate` resolves to this clone.
# --no-deps: the dependencies are already handled above.
install = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", REPO_DIR],
    cwd=REPO_DIR,
)
print("package install exit code:", install.returncode)
if install.returncode != 0:
    raise RuntimeError("could not install the lifemate package")

after = {}
for pkg in ("numpy", "torch"):
    try:
        after[pkg] = __import__(pkg).__version__
    except Exception:
        after[pkg] = None

changed = [p for p in before if before[p] and after[p] and before[p] != after[p]]
if changed:
    print(f"RESTART RUNTIME REQUIRED: {', '.join(changed)} changed "
          f"({ {p: (before[p], after[p]) for p in changed} }). "
          f"Runtime > Restart session, then re-run from the top.")
else:
    print("No core package changed — no restart needed.")

import lifemate
print("lifemate", lifemate.__version__, "from", lifemate.__file__)
""")

md("""
## 7. Prepare the dataset

Copies `data.zip` from Drive to local disk, unzips it into `/content/data_local`, then
deletes the local copy. The zip is never unpacked inside Drive.

If the folder already exists this cell is skipped, so re-runs are fast.

The archive's internal layout is unknown, so both shapes are handled:
a single top-level `data/` folder, or files directly at the root.
""")

code("""
import shutil, zipfile

DATA_READY_MARKER = os.path.join(LOCAL_DATA_DIR, ".data_ready")


def find_data_root(base):
    \"\"\"Return the folder that actually holds the split data.

    The released kit nests the splits inside a bundle folder ('train+public_test');
    the archive may also place them at the root. Both shapes are handled.
    \"\"\"
    entries = [e for e in os.listdir(base)
               if not e.startswith("__MACOSX") and not e.startswith(".")]
    if len(entries) == 1 and os.path.isdir(os.path.join(base, entries[0])):
        return os.path.join(base, entries[0])
    return base


if os.path.exists(DATA_READY_MARKER):
    with open(DATA_READY_MARKER, encoding="utf-8") as fh:
        data_root = fh.read().strip()
    print("Data already prepared — skipping.")
    print("Data root:", data_root)
else:
    if not os.path.exists(DRIVE_DATA_ZIP):
        raise FileNotFoundError(
            f"Dataset zip not found at {DRIVE_DATA_ZIP}.\\n"
            f"Upload data.zip to Drive at MyDrive/data_lifemate/ and re-run this cell."
        )

    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    local_zip = os.path.join(LOCAL_DATA_DIR, "data.zip")

    print("Copying zip from Drive to local disk...")
    shutil.copyfile(DRIVE_DATA_ZIP, local_zip)

    print("Unzipping...")
    with zipfile.ZipFile(local_zip) as zf:
        zf.extractall(LOCAL_DATA_DIR)

    os.remove(local_zip)
    print("Removed local zip copy.")

    data_root = find_data_root(LOCAL_DATA_DIR)
    with open(DATA_READY_MARKER, "w", encoding="utf-8") as fh:
        fh.write(data_root)

print("Data root:", data_root)
print("Top-level contents:")
for entry in sorted(os.listdir(data_root)):
    path = os.path.join(data_root, entry)
    print("  ", entry, "(dir)" if os.path.isdir(path) else "(file)")

# Show which splits the CLI will discover, so a wrong layout is visible immediately.
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
from lifemate.data import discover_splits

splits = discover_splits(data_root)
for name, split in sorted(splits.items()):
    print(f"  split '{name}': {split.samples_file.name} "
          f"({len(split.ids())} samples)")

total = sum(len(files) for _, _, files in os.walk(data_root))
print(f"Total files under data root: {total}")
""")

md("""
## 8. Smoke test

Runs a few training steps on a small subset to verify the pipeline end to end before
committing to a long run. Cheapest useful gate: if this cell fails, the long run will fail.
""")

code("""
smoke_cmd = [
    sys.executable, "-m", "lifemate", "train",
    "--data-dir", data_root,
    "--split", "train",
    "--output-dir", f"{OUTPUT_DIR}/smoke",
    "--config", os.path.join(REPO_DIR, "configs/sft_qlora.yaml"),
    "--epochs", "1",
    "--batch-size", "1",
    "--smoke",
]

print(" ".join(smoke_cmd))
result = subprocess.run(smoke_cmd, cwd=REPO_DIR)
if result.returncode != 0:
    raise RuntimeError(f"Smoke test failed with exit code {result.returncode}")
print("Smoke test passed.")
""")

md("""
## 9. Train

Runs from the local dataset and writes checkpoints and logs to Drive, so a Colab disconnect
does not lose progress. `--resume` picks up from the last checkpoint if one exists.

The run writes two checkpoints under `<OUTPUT_DIR>/checkpoints`: `last` (every epoch) and
`best` (on improvement). `best` is what the inference cell uses.
""")

code("""
train_cmd = [
    sys.executable, "-m", "lifemate", "train",
    "--data-dir", data_root,
    "--split", "train",
    "--output-dir", OUTPUT_DIR,
    "--config", os.path.join(REPO_DIR, "configs/sft_qlora.yaml"),
    "--epochs", str(EPOCHS),
    "--batch-size", str(BATCH_SIZE),
    "--resume",
]

print(" ".join(train_cmd))
print("Logs and checkpoints ->", OUTPUT_DIR)

env = dict(os.environ)
env["WANDB_PROJECT"] = "lifemate"
env["WANDB_NAME"] = RUN_NAME

result = subprocess.run(train_cmd, cwd=REPO_DIR, env=env)
if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")
print("Training finished.")
""")

md("""
## 10. Monitoring

TensorBoard reads the log directory on Drive. If `WANDB_API_KEY` was loaded, the training run
also reports to Weights & Biases.
""")

code("""
%load_ext tensorboard

%tensorboard --logdir {LOG_DIR}
""")

md("""
## 11. Merge the adapter

vLLM serves a merged model, so the LoRA adapter is merged into the base weights and written
to Drive. The merged model is what the final Docker image bakes in.
""")

code("""
MERGED_MODEL_DIR = f"{OUTPUT_DIR}/merged_model"

merge_cmd = [
    sys.executable, "-m", "lifemate", "merge",
    "--checkpoint", f"{CHECKPOINT_DIR}/best",
    "--output-dir", MERGED_MODEL_DIR,
]

print(" ".join(merge_cmd))
result = subprocess.run(merge_cmd, cwd=REPO_DIR)
if result.returncode != 0:
    raise RuntimeError(f"Merge failed with exit code {result.returncode}")
print("Merged model ->", MERGED_MODEL_DIR)
""")

md("""
## 12. Inference

Runs the agent over the split using the checkpoint on Drive, writes the submission JSONL to
Drive, and prints a sample row.

`--checkpoint` points at the merged model directory from the previous cell; `--model-path`
names the base model, because the LoRA adapter is already merged into it.
""")

code("""
infer_cmd = [
    sys.executable, "-m", "lifemate", "infer",
    "--data-dir", data_root,
    "--split", SPLIT,
    "--checkpoint", MERGED_MODEL_DIR,
    "--model-path", MERGED_MODEL_DIR,
    "--output-file", f"{INFER_OUTPUT_DIR}/submission.jsonl",
    "--max-steps", str(MAX_STEPS),
    "--time-budget-s", str(TIME_BUDGET_S),
    "--transcript-cache", f"{INFER_OUTPUT_DIR}/transcripts.jsonl",
    "--asr-model-dir", ASR_MODEL_DIR,
]

print(" ".join(infer_cmd))
result = subprocess.run(infer_cmd, cwd=REPO_DIR)
if result.returncode != 0:
    raise RuntimeError(f"Inference failed with exit code {result.returncode}")

submission_path = f"{INFER_OUTPUT_DIR}/submission.jsonl"
print("Wrote", submission_path, os.path.getsize(submission_path), "bytes")

import json

with open(submission_path, encoding="utf-8") as fh:
    rows = [json.loads(line) for line in fh if line.strip()]

print(f"{len(rows)} rows")
if rows:
    print("First row:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:2000])
""")

md("""
## 13. Validate the submission

Checks what the organiser's scorer will trip on: every sample present exactly once, and
every proposed call schema-valid against the split's catalog and the sample's installed apps.
""")

code("""
validate_cmd = [
    sys.executable, "-m", "lifemate", "validate",
    "--data-dir", data_root,
    "--split", SPLIT,
    "--submission", f"{INFER_OUTPUT_DIR}/submission.jsonl",
]

print(" ".join(validate_cmd))
result = subprocess.run(validate_cmd, cwd=REPO_DIR)
print("validate exit code:", result.returncode)
""")

md("""
## 14. Score the submission (labelled splits only)

Works on `train`, where gold labels exist. `public_test` has no labels, so it cannot be
scored locally.

The score should reach **100.00** for a gold-replay submission, and **~7.18** for an empty
one (the out-of-scope rate) — use these two numbers to confirm the scorer is behaving.
""")

code("""
score_cmd = [
    sys.executable, "-m", "lifemate", "score",
    "--data-dir", data_root,
    "--split", "train",
    "--submission", f"{INFER_OUTPUT_DIR}/submission.jsonl",
    "--report", f"{OUTPUT_DIR}/scores.jsonl",
]

print(" ".join(score_cmd))
result = subprocess.run(score_cmd, cwd=REPO_DIR)
print("scorer exit code:", result.returncode)
""")

md("""
## 15. Troubleshooting

**Out of memory**
Lower `BATCH_SIZE` to 1 and raise `GRAD_ACC` to compensate, or lower `MAX_SEQ_LEN` to 4096.
Enable mixed precision / gradient checkpointing in `configs/sft_qlora.yaml`. If the GPU is
shared, wait for it to free up. As a last resort, restart the runtime — the session is
discarded but Drive keeps the checkpoints.

**Colab disconnected or the runtime was reset**
Re-run every cell from the top. The clone cell updates the repo, the data cell skips (the
folder is still on Drive's copy of the zip), and the training cell resumes from the `last`
checkpoint because `--resume` is passed. Nothing is lost except the current epoch.

**Stale code**
Re-run the clone cell after every push. It fetches, checks out `BRANCH`, and pulls.

**Dataset zip missing**
`FileNotFoundError: Dataset zip not found` means `MyDrive/data_lifemate/data.zip` is not
visible. Check that Drive is mounted (cell 2) and that the file is in MyDrive, not in
"Shared with me". The expected path is in `DRIVE_DATA_ZIP` in the config cell.

**Out of disk space on `/content`**
`/content` holds the repository, the unzipped dataset, and pip packages. If the drive is
full, remove the pip cache: `!pip cache purge`.

**The GPU cell says no CUDA**
`Runtime > Change runtime type > GPU`, then `Runtime > Restart session`, then re-run from
the top.

**A submission scores ~0**
Almost always a state-hash mismatch: a wrong ID, a wrong date/time, or a missing required
argument. Use `python -m lifemate score --split train --submission FILE --report FILE` and
read the error taxonomy in the report.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "colab_launcher.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, ensure_ascii=False, indent=1)
    fh.write("\n")
print("wrote", sys.argv[1], "cells:", len(cells))
