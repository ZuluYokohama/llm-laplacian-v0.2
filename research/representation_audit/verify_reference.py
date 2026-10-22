"""Run only finite representation controls in a new pinned JoVin receipt chain."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def record(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "runs/publication_study"):
        raise ValueError("Output must stay inside publication study")
    out.mkdir(parents=True, exist_ok=False)
    data = out / "jovin_data"
    data.mkdir()
    config = ROOT / "runs/publication_study/concept_reference_20261022_001/jovin_data/config.toml"
    (data / "config.toml").write_bytes(config.read_bytes())
    env = dict(os.environ, JOVIN_HARNESS_DATA_DIR=str(data), JOVIN_HARNESS_RUNTIME="wsl:JovinHarness")
    env.pop("JOVIN_HARNESS_CONTAINER_ARGV", None)
    exe = Path(sys.executable).parent / "jovin-harness.exe"
    code, tests = HERE / "reference.py", HERE / "reference_tests.py"
    expected = [node.name for node in ast.parse(tests.read_text()).body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    result = subprocess.run([str(exe), "verify", str(code), "--tests", str(tests), "--isolation", "process", "--timeout", "10"],
                            env=env, cwd=ROOT, capture_output=True, timeout=60)
    (out / "verification.json").write_bytes(result.stdout)
    (out / "verification.stderr.log").write_bytes(result.stderr)
    response = json.loads(result.stdout)
    assert result.returncode == 0 and response["passed"] is True and response["integrity"] == "process_isolated"
    stored = json.loads((data / "receipts" / (response["receipt_id"] + ".json")).read_bytes())
    named = [test for test in stored["result"]["tests"] if test["name"] != "<module>"]
    assert [row["name"] for row in named] == expected and all(row["outcome"] == "passed" for row in named)
    for name, command in [("chain_verify", ["receipts", "verify"]), ("chain_head", ["receipts", "head"])]:
        checked = subprocess.run([str(exe), *command], env=env, capture_output=True, timeout=30)
        (out / (name + ".json")).write_bytes(checked.stdout)
        assert checked.returncode == 0
        if name == "chain_verify":
            assert json.loads(checked.stdout)["valid"] is True
    sources = []
    for path in [code, tests, Path(__file__).resolve()]:
        sources.append(record(path))
        target = out / "source_archive" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    for key, path in [("code_sha256", code), ("tests_sha256", tests)]:
        assert hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() == stored["inputs"][key]
    receipt = {"kind": "representation_reference_jovin_v1", "math_cases_passed": len(named), "cases": named,
               "sources": sources, "response": response, "new_workspace_chain": True,
               "scope": "Finite representation-contract controls only; no model, held-out data or GPU execution",
               "confirmatory_claim_permitted": False}
    target = out / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    target.with_suffix(".sha256").write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "\n", encoding="ascii")
    print(json.dumps({"named_math_cases_passed": len(named), "chain_valid": True, "integrity": response["integrity"]}))


if __name__ == "__main__":
    main()
