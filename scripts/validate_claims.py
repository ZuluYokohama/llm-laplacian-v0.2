"""Validate explicit claim/evidence bindings, not the scientific interpretation.

Draft manifests may contain planned findings. Publication validation additionally
requires every listed finding to be resolved and a recorded human review. Neither
mode proves causality, novelty, citation support, or independence of replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]


def safe_file(root, relative):
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("evidence path must be nonempty")
    win = PureWindowsPath(relative)
    if win.is_absolute() or win.drive or "\\" in relative:
        raise ValueError(f"evidence path must be repository relative: {relative}")
    base = Path(root).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"evidence must be an existing file within the repository: {relative}")
    return path


def file_hash(path, convention="raw_bytes"):
    data = Path(path).read_bytes()
    if convention == "lf_normalized_utf8":
        data = data.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    elif convention != "raw_bytes":
        raise ValueError(f"unknown hash convention: {convention}")
    return hashlib.sha256(data).hexdigest()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON value")
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)


def lookup(payload, selector):
    if not isinstance(selector, list) or not selector:
        raise ValueError("selector must be a nonempty list of keys/indices")
    current = payload
    for key in selector:
        if isinstance(current, dict) and isinstance(key, str):
            current = current[key]
        elif isinstance(current, list) and type(key) is int and key >= 0:
            current = current[key]
        else:
            raise ValueError(f"invalid selector component: {key}")
    return current


def validate_manifest(manifest, root=ROOT, publication=False):
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported claim manifest schema")
    artifacts = manifest.get("artifacts")
    claims = manifest.get("claims")
    if not isinstance(artifacts, dict) or not artifacts or not isinstance(claims, list) or not claims:
        raise ValueError("nonempty artifacts and claims required")
    loaded = {}
    for key, spec in artifacts.items():
        path = safe_file(root, spec.get("path"))
        if file_hash(path, spec.get("hash_convention", "raw_bytes")) != spec.get("sha256"):
            raise ValueError(f"artifact hash mismatch: {key}")
        if spec.get("kind") in ("json", "metrics"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            finite_json(payload)
            if spec["kind"] == "metrics":
                if not isinstance(payload, dict) or payload.get("complete") is not True:
                    raise ValueError(f"incomplete metrics: {key}")
                side = path.with_name(path.name + ".sha256")
                if not side.is_file() or side.read_text(encoding="utf-8").strip() != file_hash(path, "lf_normalized_utf8"):
                    raise ValueError(f"metrics sidecar mismatch: {key}")
            loaded[key] = payload
        elif spec.get("kind") == "file":
            if path.stat().st_size == 0:
                raise ValueError(f"empty source: {key}")
            loaded[key] = None
        else:
            raise ValueError(f"unknown artifact kind: {key}")
    ids = set()
    for claim in claims:
        identity = claim.get("id")
        if not isinstance(identity, str) or not identity or identity in ids:
            raise ValueError("missing or duplicate claim id")
        ids.add(identity)
        for field in ("statement", "scope", "limitations"):
            if not isinstance(claim.get(field), str) or not claim[field].strip():
                raise ValueError(f"claim {identity} missing {field}")
        status = claim.get("status")
        if status not in ("observed", "failed", "inconclusive", "planned"):
            raise ValueError(f"invalid claim status: {identity}")
        evidence = claim.get("evidence", [])
        if status != "planned" and not evidence:
            raise ValueError(f"unbound claim: {identity}")
        for binding in evidence:
            key = binding.get("artifact")
            if key not in loaded:
                raise ValueError(f"unknown evidence artifact: {key}")
            if "selector" in binding:
                actual = lookup(loaded[key], binding["selector"])
                if "equals" not in binding or type(actual) is not type(binding["equals"]) or actual != binding["equals"]:
                    raise ValueError(f"evidence value mismatch: {identity}/{key}")
        if publication:
            review = claim.get("human_review", {})
            if status == "planned" or review.get("approved") is not True or not review.get("reviewer") or not review.get("date"):
                raise ValueError(f"publication claim unresolved or not human-reviewed: {identity}")
    if publication:
        manuscript = manifest.get("manuscript", {})
        path = safe_file(root, manuscript.get("path"))
        if file_hash(path) != manuscript.get("sha256"):
            raise ValueError("human review must bind the exact manuscript bytes")
    return {
        "artifact_count": len(artifacts), "claim_count": len(claims),
        "mode": "publication" if publication else "draft",
        "integrity_passed": True,
        "limitation": "Host-side SHA-256 binding/integrity checks only; this report is not a Jovin isolated validation receipt. Does not prove scientific support or audit unlisted prose.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_manifest(json.loads(args.manifest.read_text(encoding="utf-8")), publication=args.publication)
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        parser.exit(1, f"CLAIM VALIDATION FAILED: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
