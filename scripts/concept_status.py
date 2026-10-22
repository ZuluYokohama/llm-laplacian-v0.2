"""Read-only concept/evidence snapshot. Never schedules work or admits claims."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import study_budget as budget

ROOT = Path(__file__).resolve().parents[1]
STATES = {"finite_checks", "exploratory", "not_evaluated", "source_gap", "deferred", "restricted", "partial", "pending"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Path escapes workspace")
    return path


def checked(root, record, external=False):
    path = Path(record["path"]) if external else local(root, record["path"])
    raw = path.read_bytes()
    require(record.get("hash_convention", "raw_bytes") == "raw_bytes", "Unknown hash convention")
    require(digest(raw) == record["sha256"], f"Changed bound file: {path.name}")
    if "bytes" in record:
        require(len(raw) == record["bytes"], "Bound size differs")
    return raw


def document(root, path):
    path = local(root, path)
    raw = path.read_bytes()
    require(path.with_suffix(".sha256").read_text().strip() == digest(raw), "Document sidecar differs")
    return json.loads(raw)


def inspect_audits(root, paths):
    """Validate local descriptive audit receipts; never infer scientific merit.

    A valid failure receipt is evidence of the failed attempt. It is not a
    successful diagnostic. Records bind raw bytes only, including the manifest.
    """
    observed, receipt_paths, manifest_paths, manifest_digests = [], set(), set(), set()

    def audit_record(record):
        require(isinstance(record, dict) and set(("path", "bytes", "sha256")) <= record.keys(),
                "Audit record lacks path/size/digest")
        require(type(record["bytes"]) is int and record["bytes"] >= 0, "Invalid audit record size")
        path = local(root, record["path"])
        raw = checked(root, record)
        return {"path": str(path), "bytes": len(raw), "sha256": digest(raw), "hash_convention": "raw_bytes"}

    for name in paths:
        path = local(root, name)
        require(path not in receipt_paths, "Duplicate audit receipt")
        receipt_paths.add(path)
        row = document(root, path)
        require(row.get("kind") == "representation_audit_receipt_v1", "Unknown audit receipt kind")
        require(row.get("state") in {"completed", "failed", "resource_exhausted"}, "Unknown audit state")
        require(type(row.get("complete")) is bool and row["complete"] == (row["state"] == "completed"),
                "Audit state and completion differ")
        require(row.get("scope") == "host_cpu_training_only_diagnostics" and row.get("device") == "cpu",
                "Audit is not a host CPU training-only diagnostic")
        require(row.get("confirmatory_claim_permitted") is False and row.get("publication_admitted") is False,
                "Audit cannot admit confirmatory or publication claims")
        manifest = audit_record(row["manifest"])
        manifest_path = Path(manifest["path"])
        require(manifest_path not in manifest_paths and manifest["sha256"] not in manifest_digests, "Duplicate audit manifest")
        manifest_paths.add(manifest_path)
        manifest_digests.add(manifest["sha256"])
        require(isinstance(row["sources"], list) and isinstance(row["outputs"], list), "Audit inventory is not a list")
        require(bool(row["sources"]), "Audit source inventory is empty")
        require(not row["complete"] or bool(row["outputs"]), "Completed audit has no outputs")
        sources = [audit_record(record) for record in row["sources"]]
        outputs = [audit_record(record) for record in row["outputs"]]
        observed.append({"receipt": {"path": str(path), "bytes": path.stat().st_size,
                                      "sha256": digest(path.read_bytes()), "hash_convention": "raw_bytes"},
                         "manifest": manifest, "sources": sources, "outputs": outputs,
                         "state": row["state"], "diagnostic_state": "completed_descriptive" if row["complete"] else row["state"],
                         "scope": row["scope"], "device": row["device"],
                         "confirmatory_claim_permitted": False, "publication_admitted": False})
    return observed


def attach_audits(report, audits):
    """Return a report copy, preserving prior scientific and region statuses."""
    result = {**report, "regions": [dict(region) for region in report["regions"]], "audits": audits}
    if audits:
        states = {audit["diagnostic_state"] for audit in audits}
        state = "failed" if "failed" in states else "resource_exhausted" if "resource_exhausted" in states else "completed_descriptive"
        for region in result["regions"]:
            if region["id"] in {"CXM-GEOM", "CXM-EXEC"}:
                region["audit_evidence"] = [audit["receipt"] for audit in audits]
                region["diagnostic_state"] = state
    return result


def inspect_map(root, mapping):
    require(type(mapping.get("schema_version")) is int and mapping["schema_version"] == 1, "Unknown map schema")
    ids = [r["id"] for r in mapping["regions"]]
    require(len(ids) == len(set(ids)), "Duplicate region")
    source_ids = {s["id"]: s for s in mapping["sources"]}
    require(len(source_ids) == len(mapping["sources"]), "Duplicate source")
    for source in mapping["sources"]:
        checked(root, source, external=True)
    for record in mapping["artifacts"].values():
        raw = checked(root, record)
        if record.get("checks"):
            data = json.loads(raw)
            for check in record["checks"]:
                value = data
                for part in check["selector"]:
                    value = value[part]
                require(budget.canonical(value) == budget.canonical(check["equals"]), "Evidence assertion differs")
    for region in mapping["regions"]:
        require(region["status"] in STATES, "Unknown region state")
        require(set(region["evidence"]) <= mapping["artifacts"].keys(), "Unknown evidence reference")
        for reference in region.get("source_refs", []):
            require(reference["document"] in source_ids, "Unknown source reference")
            require(all(type(page) is int and 1 <= page <= source_ids[reference["document"]]["pages"]
                        for page in reference["pages"]), "Source page outside document")
        for anchor in region["implementation"]:
            checked(root, anchor)
    return mapping["regions"]


def execution(root, mapping):
    """Reconcile receipts with accounting, without importing Torch or loading checkpoints.

    Checkpoints/sampling are attested by the existing controller; this reporter
    validates receipt bindings, not their full binary contents or process liveness.
    """
    manifest_path = local(root, mapping["execution"]["continuation_manifest"])
    manifest = document(root, manifest_path)
    require(manifest["schema"] == "ce_parent_continuation_v1", "Unknown continuation schema")
    checked(root, manifest["parent_stage"])
    parent_path = local(root, manifest["parent_stage"]["path"])
    parent = document(root, parent_path)
    for record in parent["source_records"] + manifest["configs"] + [manifest["self"], manifest["python"]]:
        checked(root, record)
    for row in parent["ce_parents"]:
        checked(root, row["config"])
    folder, parent_folder = manifest_path.parent, parent_path.parent
    ledger = local(root, mapping["execution"]["ledger"])
    records = budget.read_ledger(ledger)
    runs = budget.accounting(records)
    cap, _ = budget.approval(local(root, mapping["execution"]["grant"]), ledger)
    blocks, per_seed = [], {seed: 0 for seed in range(5)}
    jobs = [(n, job, parent_folder) for n, job in enumerate(parent["segments"]) if job["seed"] == 0]
    jobs += [(job["number"], job, folder) for job in manifest["jobs"]]
    require([n for n, _, _ in jobs] == list(range(50)), "Segment order differs")
    for number, job, directory in jobs:
        marker = directory / f"block_{number:03d}.json"
        if not marker.exists():
            continue
        # Original controller markers predate continuation sidecars. Their raw
        # digests are bound by seed0_admission and their settlements by the ledger.
        row = json.loads(marker.read_bytes()) if directory == parent_folder else document(root, marker)
        require(row["job"] == job, "Segment job differs")
        rid = f"{directory.name}-b{number:03d}"
        require(row["budget_run_id"] == rid and rid in runs, "Segment reservation differs")
        run = runs[rid]
        require(run["settled"] and run["settlement"]["status"] == "completed", "Segment has no successful settlement")
        require(row["settlement_sha256"] == run["settlement"]["sha256"] and row["charge_seconds"] == run["charge_seconds"], "Segment settlement differs")
        checked(root, row["training_receipt"])
        require(job["from_step"] == per_seed[job["seed"]], "Noncontiguous parent receipts")
        per_seed[job["seed"]] = job["to_step"]
        blocks.append(number)
    require(blocks == list(range(len(blocks))), "Nonserial completed block inventory")
    admission = folder / "seed0_admission.json"
    if admission.exists():
        admitted = document(root, admission)
        require(per_seed[0] == 50000 and admitted["parent"]["updates"] == 50000, "Seed 0 admission differs")
        for row in admitted["parent"]["block_receipts"]:
            checked(root, row)
    for seed in range(1, 5):
        if per_seed[seed] == 50000:
            completed = document(root, folder / f"parent_seed{seed}_complete.json")
            require(completed["seed"] == seed and completed["updates"] == 50000, "Parent completion differs")
            for record in completed["block_receipts"]:
                checked(root, record)
    status = json.loads((folder / "status.json").read_bytes())
    open_ids = [rid for rid, run in runs.items() if not run["settled"]]
    require(len(open_ids) <= 1, "Multiple open reservations")
    if status["state"] == "training":
        require(admission.exists(), "Continuation training lacks seed 0 admission")
        require(status["budget_run_id"] in open_ids, "Training status and open reservation differ")
        expected = next((j for n, j, _ in jobs if n == len(blocks)), None)
        require(status["job"] == expected, "Active job is not the next serial segment")
    if status["state"] == "completed":
        require(len(blocks) == 50 and not open_ids, "Completion lacks reconciled inventory")
    charge = sum(r["charge_seconds"] for r in runs.values())
    own_charge = sum(r["charge_seconds"] for rid, r in runs.items() if rid.startswith(folder.name + "-b"))
    require(charge <= cap and own_charge <= manifest["ceiling_seconds"], "Budget ceiling exceeded")
    return {"controller_reported": status, "pause_requested": (folder / "PAUSE").exists(),
            "completed_blocks": len(blocks), "updates_by_seed": per_seed,
            "completed_ce_parents": sum(v == 50000 for v in per_seed.values()),
            "open_reservations": open_ids, "global_charged_or_reserved_hours": charge / 3600,
            "global_cap_hours": cap / 3600, "continuation_charged_or_reserved_hours": own_charge / 3600,
            "ledger_head": records[-1]["sha256"] if records else None,
            "scope": "Receipt/accounting snapshot; no checkpoint deserialization, liveness or held-out quality attestation."}


def snapshot(root, map_path, audits=()):
    raw = map_path.read_bytes()
    mapping = json.loads(raw)
    regions = inspect_map(root, mapping)
    audit_evidence = inspect_audits(root, audits)
    # A segment can finish between reads. Only accept an unchanged ledger/status view.
    status = local(root, mapping["execution"]["continuation_manifest"]).parent / "status.json"
    ledger = local(root, mapping["execution"]["ledger"])
    for attempt in range(3):
        before = (status.read_bytes(), ledger.read_bytes())
        try:
            observed = execution(root, mapping)
        except (ValueError, OSError, KeyError, budget.BudgetError):
            if before == (status.read_bytes(), ledger.read_bytes()):
                raise
        else:
            if before == (status.read_bytes(), ledger.read_bytes()):
                return attach_audits({"schema_version": 1, "utc": datetime.now(timezone.utc).isoformat(),
                        "map_sha256": digest(raw), "evidence_integrity": "passed", "execution": observed,
                        "regions": regions, "scientific_gates": mapping["scientific_gates"],
                        "confirmatory_claim_permitted": False, "publication_admitted": False}, audit_evidence)
        time.sleep(.1)
    raise ValueError("Controller/accounting changed during all three snapshot attempts")


def markdown(report):
    lines = ["# Concept-to-experiment snapshot", "", report["utc"], "",
             "Evidence bindings passed. Scientific claims and publication are not admitted.", "",
             f"CE parents complete: {report['execution']['completed_ce_parents']}/5; reconciled blocks: {report['execution']['completed_blocks']}/50.", "",
             "Controller reports: " + report["execution"]["controller_reported"]["state"] + ".", "",
             "| Region | Evidence state | Next gate |", "|---|---|---|"]
    for row in report["regions"]:
        lines.append(f"| {row['name']} | {row['status']} | {row['next_gate']} |")
    if report.get("audits"):
        lines += ["", "## Representation audits", "",
                  "These are host CPU training-only diagnostics. Completion is descriptive evidence; scientific gates remain unchanged.", "",
                  "| Receipt | Diagnostic state | Scope |", "|---|---|---|"]
        for audit in report["audits"]:
            lines.append(f"| {Path(audit['receipt']['path']).name} ({audit['receipt']['sha256'][:12]}) | {audit['diagnostic_state']} | {audit['scope']} |")
    lines += ["", "This is a timestamped, read-only snapshot. Open reservations are charged at their full reserved amount.",
              "Receipt reconciliation relies on the existing controller's checkpoint/sampling audits; it does not repeat those binary audits.", ""]
    return "\n".join(lines)


def save(output, report):
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in {"status.json": json.dumps(report, indent=2, allow_nan=False).encode() + b"\n",
                      "status.md": markdown(report).encode()}.items():
        (output / name).write_bytes(raw)
        (output / (name + ".sha256")).write_text(digest(raw) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit", action="append", type=Path, default=[], help="Hash-bound host CPU representation-audit receipt; repeatable")
    args = parser.parse_args()
    try:
        report = snapshot(ROOT, local(ROOT, args.map), args.audit)
        if args.output_dir:
            save(local(ROOT, args.output_dir), report)
        else:
            print(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, budget.BudgetError) as exc:
        print(json.dumps({"evidence_integrity": "invalid", "error": str(exc), "publication_admitted": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
