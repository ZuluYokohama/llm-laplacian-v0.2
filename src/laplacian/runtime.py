"""Paths, metric files, and process environment for V0.2 runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
SRC = ROOT / "src"
CACHE = ROOT / "cache"
USER_HUB = Path(r"C:\Users\JT-DEV1\.cache\huggingface\hub")


def apply_env():
    """Keep downloads, temp files, and checkpoints on D:."""
    tmp = CACHE / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (CACHE / "hf").mkdir(parents=True, exist_ok=True)
    (CACHE / "torch").mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["PIP_CACHE_DIR"] = str(CACHE / "pip")
    os.environ["HF_HOME"] = str(CACHE / "hf")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE / "hf" / "datasets")
    os.environ["TORCH_HOME"] = str(CACHE / "torch")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


def write_metrics(path, payload):
    """Write JSON and a sibling sha256 of LF-normalized UTF-8 text.

    Windows text output may contain CRLF bytes. Historical sidecars hash the
    logical LF text, not those raw on-disk bytes; retain that convention.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()), encoding="utf-8")
