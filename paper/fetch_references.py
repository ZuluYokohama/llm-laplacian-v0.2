"""Retrieve citation and funding pages. Drop a seed that does not resolve.

Exit 0 when the local source documents exist, even if a remote page is dropped.
A dropped page is recorded in paper/dropped.json and is absent from paper/references.json.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from laplacian.runtime import apply_env, write_metrics  # noqa: E402

SOURCES = ROOT / "paper" / "sources"
FUNDING = ROOT / "paper" / "funding"
MAX_SOURCE_BYTES = 50 * 1024 * 1024

SEEDS = [
    {
        "key": "spielman_srivastava_2008",
        "title": "Graph Sparsification by Effective Resistances",
        "year": 2008,
        "url": "https://arxiv.org/abs/0803.0929",
        "doi": "10.48550/arXiv.0803.0929",
        "label": "Spielman and Srivastava (2008)",
    },
    {
        "key": "belkin_niyogi_2006",
        "title": "Convergence of Laplacian Eigenmaps",
        "year": 2006,
        "url": "https://papers.nips.cc/paper_files/paper/2006/file/5848ad959570f87753a60ce8be1567f3-Paper.pdf",
        "doi": "10.7551/mitpress/7503.003.0021",
        "label": "Belkin and Niyogi (2006)",
    },
    {
        "key": "magma_2024",
        "title": "arXiv:2412.02871",
        "year": 2024,
        "url": "https://arxiv.org/abs/2412.02871",
        "doi": "10.48550/arXiv.2412.02871",
        "label": "arXiv:2412.02871",
    },
    {
        "key": "achlioptas_2003",
        "title": "Database-friendly random projections: Johnson-Lindenstrauss with binary coins",
        "year": 2003,
        "url": "https://doi.org/10.1016/S0022-0000(03)00025-4",
        "doi": "10.1016/S0022-0000(03)00025-4",
        "label": "Achlioptas (2003)",
    },
    {
        "key": "tinystories_2023",
        "title": "TinyStories",
        "year": 2023,
        "url": "https://arxiv.org/abs/2305.07759",
        "doi": "10.48550/arXiv.2305.07759",
        "label": "Eldan and Li (2023)",
    },
    {
        "key": "ifeval_2023",
        "title": "Instruction-Following Evaluation for Large Language Models",
        "year": 2023,
        "url": "https://arxiv.org/abs/2311.07911",
        "doi": "10.48550/arXiv.2311.07911",
        "label": "Zhou et al. (2023)",
    },
    {
        "key": "smollm2_135m",
        "title": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "year": 2024,
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct",
        "label": "SmolLM2-135M-Instruct model card",
    },
    {
        "key": "qwen25_1p5b_instruct",
        "title": "Qwen/Qwen2.5-1.5B-Instruct",
        "year": 2024,
        "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        "label": "Qwen2.5-1.5B-Instruct model card",
    },
]

LOCALS = [
    {
        "key": "spec_laplacian_llm",
        "title": "Laplacian LLM Spec and Plan",
        "path": r"C:\Users\JT-DEV1\Downloads\Laplacian LLM Spec and Plan.pdf",
        "label": "Laplacian LLM Spec and Plan (local unpublished file)",
    },
    {
        "key": "spec_guidance",
        "title": "SPEC GUIDANCE",
        "path": r"C:\Users\JT-DEV1\Downloads\SPEC GUIDANCE.md",
        "label": "SPEC GUIDANCE (local unpublished file)",
    },
]

FUNDING_PAGES = [
    {
        "key": "nsf_mfai",
        "url": "https://www.nsf.gov/funding/opportunities/mfai-mathematical-foundations-artificial-intelligence",
    },
    {
        "key": "nsf_mfai_listing",
        "url": "https://simpler.grants.gov/opportunity/508e8ee7-6925-4593-a548-66578974572f",
    },
    {
        "key": "doe_sc_continuation",
        "url": "https://simpler.grants.gov/opportunity/34942f3a-232f-48f4-a624-123dd6d042a0",
    },
    {"key": "doe_ascr", "url": "https://science.osti.gov/ascr/Funding-Opportunities"},
]


def retrieve(url):
    request = urllib.request.Request(url, headers={"User-Agent": "laplacian-v0.2-citation-fetch"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            final = response.geturl()
            body = response.read(MAX_SOURCE_BYTES + 1)
            content_type = response.headers.get("Content-Type", "application/octet-stream")
        if len(body) > MAX_SOURCE_BYTES:
            raise ValueError("source exceeds byte limit; refusing a truncated citation")
        pdf = body.startswith(b"%PDF-")
        return {"ok": status < 400, "status": status, "url": final, "body": body,
                "content_type": content_type, "pdf": pdf,
                "text": "" if pdf else body.decode("utf-8", errors="replace")}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "url": url, "error": str(exc)[:300], "text": ""}
    except Exception as exc:
        return {"ok": False, "status": None, "url": url, "error": f"{exc.__class__.__name__}: {exc}"[:300], "text": ""}


def page_title(text, fallback):
    match = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return fallback
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title[:240] or fallback


def keep_local(seed):
    path = Path(seed["path"])
    if not path.exists():
        return None
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    destination = SOURCES / f"{seed['key']}{path.suffix or '.bin'}"
    destination.write_bytes(data)
    return {
        "title": seed["title"],
        "label": seed["label"],
        "local_source": True,
        "path": str(path),
        "sha256": digest,
        "source_file": destination.relative_to(ROOT).as_posix(),
        "source_sha256": digest,
        "citation_support_reviewed": False,
    }


def main():
    apply_env()
    SOURCES.mkdir(parents=True, exist_ok=True)
    FUNDING.mkdir(parents=True, exist_ok=True)
    references = {}
    dropped = []
    for seed in LOCALS:
        entry = keep_local(seed)
        if entry is None:
            print(f"FAILED missing local source {seed['path']}", flush=True)
            return 1
        references[seed["key"]] = entry
        print(f"KEEP local {seed['key']}", flush=True)
    for seed in SEEDS:
        page = retrieve(seed["url"])
        if not page["ok"] or len(page.get("body", b"")) < 200:
            dropped.append({"key": seed["key"], "url": seed["url"], "status": page.get("status"), "error": page.get("error")})
            print(f"DROP {seed['key']} status={page.get('status')}", flush=True)
            continue
        extension = ".pdf" if page["pdf"] else ".html" if "html" in page["content_type"] else ".txt"
        destination = SOURCES / f"{seed['key']}{extension}"
        destination.write_bytes(page["body"])
        references[seed["key"]] = {
            "title": page_title(page["text"], seed["title"]),
            "year": seed.get("year"),
            "label": seed["label"],
            "url": seed["url"],
            "retrieved_url": page["url"],
            "doi": seed.get("doi"),
            "local_source": False,
            "source_file": destination.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(page["body"]).hexdigest(),
            "content_type": page["content_type"],
            "citation_support_reviewed": False,
        }
        print(f"KEEP {seed['key']}", flush=True)
    funding = []
    for page_seed in FUNDING_PAGES:
        page = retrieve(page_seed["url"])
        (FUNDING / f"{page_seed['key']}.txt").write_text(
            page["text"][:80_000] or page.get("error") or "",
            encoding="utf-8",
        )
        funding.append({"key": page_seed["key"], "url": page_seed["url"], "status": page.get("status"), "ok": page["ok"]})
        print(f"FUNDING {page_seed['key']} ok={page['ok']} status={page.get('status')}", flush=True)
    (ROOT / "paper" / "references.json").write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")
    (ROOT / "paper" / "dropped.json").write_text(json.dumps(dropped, indent=2) + "\n", encoding="utf-8")
    (FUNDING / "index.json").write_text(json.dumps(funding, indent=2) + "\n", encoding="utf-8")
    digest = write_metrics(
        ROOT / "runs" / "fetch_references" / "metrics.json",
        {
            "run_id": "fetch_references",
            "complete": True,
            "host_gpu": False,
            "retrieved": sorted(references),
            "dropped": [row["key"] for row in dropped],
            "funding_pages": funding,
            "note": "Funding pages are saved for a later crosswalk. This file does not say which solicitations are open.",
        },
    )
    print(f"DONE fetch_references sha256={digest} dropped={len(dropped)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
