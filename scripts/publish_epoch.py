"""Record a new Arm 0 weight file in ledger/arm0.jsonl and, with --publish, push that line.

The weight file stays on disk. Git receives one JSON line: step, size, sha256,
and the logged loss. The script never adds *.pt, logs, or secret files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "runs" / "arm0"
LEDGER = ROOT / "ledger" / "arm0.jsonl"
LOG = ROOT / "runs" / "queue.log"
BRANCH = "arm0-epochs"
REMOTE = "origin"
STEP_RE = re.compile(
    r"STEP (\d+)/300000 loss=([0-9.eE+-]+) ce=([0-9.eE+-]+) vram_mb=([0-9.]+) nan=(\d+)"
)
GIT_USER = ["-c", "user.name=Zulu Operations", "-c", "user.email=290113042+ZuluYokohama@users.noreply.github.com"]


def existing_steps():
    if not LEDGER.exists():
        return set()
    steps = set()
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        steps.add(int(json.loads(line)["step"]))
    return steps


def log_text():
    if not LOG.exists():
        return ""
    raw = LOG.read_bytes().replace(b"\x00", b"")
    return raw.decode("utf-8", errors="ignore")


def loss_for(step, text):
    found = None
    for match in STEP_RE.finditer(text):
        if int(match.group(1)) == step:
            found = match
    if found is None:
        return None
    return {
        "loss": float(found.group(2)),
        "ce": float(found.group(3)),
        "vram_mb": float(found.group(4)),
        "nan": int(found.group(5)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable(path, expected_size):
    if time.time() - path.stat().st_mtime < 30:
        return False
    size = path.stat().st_size
    if expected_size is not None and size != expected_size:
        return False
    return size > 0


def new_rows():
    files = sorted(ARM.glob("weights_step*.pt"))
    if not files:
        return []
    done = existing_steps()
    expected = None
    for path in files:
        step = int(path.stem.replace("weights_step", ""))
        if step in done and expected is None:
            expected = path.stat().st_size
    text = log_text()
    rows = []
    for path in files:
        step = int(path.stem.replace("weights_step", ""))
        if step in done:
            continue
        if not stable(path, expected):
            continue
        row = {
            "arm": "arm0",
            "step": step,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "weight_file": path.relative_to(ROOT).as_posix(),
            "in_git": False,
            "loss_scope": "single logged step, not a mean",
        }
        logged = loss_for(step, text)
        if logged is not None:
            row.update(logged)
        rows.append(row)
        if expected is None:
            expected = row["bytes"]
    return rows


def append(rows):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def git(*args, check=True):
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise SystemExit(completed.stderr.strip() or completed.stdout.strip() or f"git {' '.join(args)} failed")
    return completed


def publish(rows):
    git("add", "--", "ledger/arm0.jsonl")
    staged = git("diff", "--cached", "--name-only")
    if "ledger/arm0.jsonl" not in staged.stdout.splitlines():
        return False
    steps = ",".join(str(row["step"]) for row in rows)
    message = f"Record Arm 0 epoch steps {steps}."
    git(*GIT_USER, "commit", "-m", message)
    git("push", REMOTE, BRANCH)
    return True


def push_if_ahead():
    status = git("status", "-sb", check=False)
    first = status.stdout.splitlines()[:1]
    if first and "ahead" in first[0]:
        git("push", REMOTE, BRANCH)
        print("EPOCH pushed")
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    rows = new_rows()
    if not rows:
        print("EPOCH none")
        if args.publish:
            push_if_ahead()
        return
    append(rows)
    print("EPOCH wrote " + ",".join(str(row["step"]) for row in rows))
    if args.publish and publish(rows):
        print("EPOCH pushed")


if __name__ == "__main__":
    main()
