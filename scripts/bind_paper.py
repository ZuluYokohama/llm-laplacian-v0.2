"""Build the report only from a publication-reviewed claim manifest.

Placeholders {{m:RUN:path}} must resolve inside a complete hashed metrics file.
Placeholders {{c:KEY}} must name a reference with a retrieved source file and a URL, DOI, or local_source.
Literature target strings are rejected unless the same line says they were not measured.
Publication inputs must be manifest artifacts. resolved_manuscript_sha256 binds
the exact substituted UTF-8/LF text; the emitted resolved file uses those bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BAITS = ("1.466", "0.866", "1e-17", "10.9 ms", "220 ms", "7 kB", "0.67 s", "0.003 ms")
ALLOW = ("literature_target", "not measured", "not run", "unverified")
PLACEHOLDER = re.compile(r"\{\{m:([A-Za-z0-9_/-]+):([A-Za-z0-9_.]+)\}\}|\{\{c:([A-Za-z0-9_]+)\}\}")


def queue_is_done():
    return (ROOT / "runs" / "QUEUE_DONE").exists()


def bait_hits(text):
    found = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(token in lowered for token in ALLOW):
            continue
        for bait in BAITS:
            if bait.lower() in lowered:
                found.append(bait)
    return found


def judge_hits(text):
    found = []
    for line in text.splitlines():
        if "gpt-4o-mini" in line.lower() and "not" not in line.lower():
            found.append(line.strip())
    return found


def metrics_ok(path):
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    if not path.exists() or not side.exists():
        return {"ok": False}
    logical = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    digest = hashlib.sha256(logical.encode("utf-8")).hexdigest()
    recorded = side.read_text(encoding="utf-8").strip()
    payload = json.loads(logical)
    return {"ok": payload.get("complete") is True and recorded == digest, "payload": payload}


def lookup(payload, path):
    current = payload
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            raise KeyError(path)
    if isinstance(current, (dict, list)):
        raise KeyError(path)
    return current


def metric_text(run_id, path):
    source = (ROOT / "runs" / run_id / "metrics.json").resolve()
    if not source.is_relative_to((ROOT / "runs").resolve()):
        raise SystemExit("metric source escapes runs directory")
    state = metrics_ok(source)
    if not state["ok"]:
        raise SystemExit(f"placeholder {run_id}:{path} is not in a complete hashed metrics file")
    try:
        value = lookup(state["payload"], path)
    except KeyError:
        raise SystemExit(f"placeholder path missing {run_id}:{path}")
    return str(value)


def cite_text(references, key):
    entry = references.get(key)
    if not entry:
        raise SystemExit(f"citation {key} is not in paper/references.json")
    linked = bool(entry.get("url") or entry.get("doi") or entry.get("local_source"))
    if not linked:
        raise SystemExit(f"citation {key} has no url, doi, or local_source")
    relative = entry.get("source_file")
    if not isinstance(relative, str) or not relative.strip():
        raise SystemExit(f"citation {key} has no retrieved source file")
    source = (ROOT / relative).resolve()
    if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or source.stat().st_size == 0:
        raise SystemExit(f"citation {key} has no retrieved source file")
    expected = entry.get("source_sha256")
    if not expected or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"citation {key} has no matching source byte hash")
    if entry.get("citation_support_reviewed") is not True:
        raise SystemExit(f"citation {key} has not been checked for support")
    return entry.get("label") or key


def substitute(text, references):
    pieces = []
    cursor = 0
    for match in PLACEHOLDER.finditer(text):
        pieces.append(text[cursor : match.start()])
        if match.group(1):
            pieces.append(metric_text(match.group(1), match.group(2)))
        else:
            pieces.append(cite_text(references, match.group(3)))
        cursor = match.end()
    pieces.append(text[cursor:])
    resolved = "".join(pieces)
    if "{{" in resolved:
        raise SystemExit("unresolved placeholder remains")
    hits = bait_hits(resolved) + judge_hits(resolved)
    if hits:
        raise SystemExit(f"manuscript contains unmeasured target language: {hits}")
    return resolved


def xml_escape(text):
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("&lt;sub&gt;", "<sub>").replace("&lt;/sub&gt;", "</sub>")
    return escaped.replace("&lt;super&gt;", "<super>").replace("&lt;/super&gt;", "</super>")


def build_pdf(text, path):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    styles = getSampleStyleSheet()
    story = []
    table_rows = []

    def flush_table():
        if not table_rows:
            return
        data = []
        for row in table_rows:
            cells = [cell.strip() for cell in row.strip("|").split("|")]
            if cells and all(set(cell) <= set("-: ") and cell for cell in cells):
                continue
            data.append([Paragraph(xml_escape(cell), styles["BodyText"]) for cell in cells])
        table_rows.clear()
        if data:
            story.append(Table(data, hAlign="LEFT"))
            story.append(Spacer(1, 8))

    for line in text.splitlines():
        if line.startswith("|"):
            table_rows.append(line)
            continue
        flush_table()
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 6))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(xml_escape(stripped[4:]), styles["Heading3"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(xml_escape(stripped[3:]), styles["Heading2"]))
        elif stripped.startswith("# "):
            story.append(Paragraph(xml_escape(stripped[2:]), styles["Heading1"]))
        else:
            story.append(Paragraph(xml_escape(stripped), styles["BodyText"]))
    flush_table()
    document = SimpleDocTemplate(str(path), pagesize=letter, title="Spectral LLM and condition-state measurements")
    document.build(story)


def render_pages(pdf, folder):
    folder.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["pdftoppm", "-png", str(pdf), str(folder / "page")],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"pdftoppm failed: {proc.stderr}")


def require_bound_artifact(manifest, path, kinds, *, raw_bytes=False):
    """Require an input in the artifact set already checked by validate_manifest."""
    path = Path(path).resolve()
    for artifact in manifest["artifacts"].values():
        if (ROOT / artifact["path"]).resolve() != path:
            continue
        if artifact.get("kind") not in kinds:
            continue
        if raw_bytes and artifact.get("hash_convention", "raw_bytes") != "raw_bytes":
            continue
        return
    raise SystemExit(f"publication input is not bound as a validated artifact: {path}")


def resolve_publication(manifest):
    """Check all publication gates before returning any text for output.

    This verifies recorded review bindings, not the authenticity or scientific
    judgment of a human review. It never creates or updates review declarations.
    """
    from validate_claims import validate_manifest

    try:
        validate_manifest(manifest, root=ROOT, publication=True)
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        raise SystemExit(f"publication evidence gate failed: {exc}")
    manuscript = ROOT / "paper" / "manuscript.md"
    if (ROOT / manifest["manuscript"]["path"]).resolve() != manuscript.resolve():
        raise SystemExit("reviewed manuscript does not match the build input")
    reference_path = ROOT / "paper" / "references.json"
    require_bound_artifact(manifest, reference_path, {"json", "file"}, raw_bytes=True)
    references = json.loads(reference_path.read_text(encoding="utf-8"))
    text = manuscript.read_text(encoding="utf-8")
    resolved = substitute(text, references)
    for match in PLACEHOLDER.finditer(text):
        if match.group(1):
            source = ROOT / "runs" / match.group(1) / "metrics.json"
            require_bound_artifact(manifest, source, {"metrics"})
        else:
            source = ROOT / references[match.group(3)]["source_file"]
            require_bound_artifact(manifest, source, {"file"}, raw_bytes=True)
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    if manifest.get("resolved_manuscript_sha256") != digest:
        raise SystemExit("human review must bind the exact resolved manuscript UTF-8/LF bytes")
    return resolved


def main():
    # A queue marker cannot establish that the manuscript's claims are ready.
    manifest_path = ROOT / "paper" / "claim_evidence.json"
    if not manifest_path.is_file():
        raise SystemExit("refusing publication build without paper/claim_evidence.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = resolve_publication(manifest)
    (ROOT / "paper" / "manuscript.resolved.md").write_bytes(resolved.encode("utf-8"))
    pdf = ROOT / "paper" / "spectral_llm_report.pdf"
    build_pdf(resolved, pdf)
    render_pages(pdf, ROOT / "paper" / "pages")
    print(f"DONE paper {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
