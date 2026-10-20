"""Build the report only after runs/QUEUE_DONE.

Placeholders {{m:RUN:path}} must resolve inside a complete hashed metrics file.
Placeholders {{c:KEY}} must name a reference with a retrieved source file and a URL, DOI, or local_source.
Literature target strings are rejected unless the same line says they were not measured.
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
PLACEHOLDER = re.compile(r"\{\{m:([A-Za-z0-9_]+):([A-Za-z0-9_.]+)\}\}|\{\{c:([A-Za-z0-9_]+)\}\}")


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
    state = metrics_ok(ROOT / "runs" / run_id / "metrics.json")
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
    source = ROOT / entry.get("source_file", "")
    if not source.exists():
        raise SystemExit(f"citation {key} has no retrieved source file")
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


def main():
    if not queue_is_done():
        raise SystemExit("refusing to build the paper before runs/QUEUE_DONE")
    manuscript = ROOT / "paper" / "manuscript.md"
    references = json.loads((ROOT / "paper" / "references.json").read_text(encoding="utf-8"))
    resolved = substitute(manuscript.read_text(encoding="utf-8"), references)
    (ROOT / "paper" / "manuscript.resolved.md").write_text(resolved, encoding="utf-8")
    pdf = ROOT / "paper" / "spectral_llm_report.pdf"
    build_pdf(resolved, pdf)
    render_pages(pdf, ROOT / "paper" / "pages")
    print(f"DONE paper {pdf}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
