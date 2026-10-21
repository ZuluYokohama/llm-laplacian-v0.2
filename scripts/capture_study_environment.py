"""Record the observed Windows core-study dependency closure without installing.

The requirements are version pins, not a solved or artifact-hashed lockfile.
Optional extras and historical transformer-model experiments are out of scope.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def capture(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pending = ["torch", "numpy", "tokenizers", "pyarrow", "pytest", "pip"]
    found = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in found:
            continue
        distribution = metadata.distribution(name)
        found[name] = distribution.version
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                actual = metadata.version(requirement.name)
                if requirement.specifier and actual not in requirement.specifier:
                    raise ValueError(f"Installed dependency does not satisfy {requirement.name}")
                pending.append(requirement.name)
    pinned = [f"{name}=={version}" for name, version in sorted(found.items())]
    for name, lines in {
        "requirements-observed.txt": pinned,
        "requirements-pypi.txt": [line for line in pinned if not line.startswith("torch==")],
        "requirements-torch.txt": [line for line in pinned if line.startswith("torch==")],
    }.items():
        (output / name).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    report = {
        "scope": "Observed core-study dependency closure, no optional extras or historical HF model dependencies",
        "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
        "executable_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
        "packages": found, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.glob("*.txt")},
        "clean_install_verified": False,
        "limitations": "Version pins alone do not establish wheel identity, a lockfile, or clean-environment reproduction.",
    }
    (output / "observed.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.output), sort_keys=True))
