import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


binder = load("scripts/bind_paper.py", "paper_binder_citation_test")
fetch = load("paper/fetch_references.py", "paper_fetch_citation_test")


@pytest.mark.parametrize("source", [None, "", ".", "missing.txt"])
def test_url_alone_and_directory_cannot_satisfy_citation(tmp_path, monkeypatch, source):
    monkeypatch.setattr(binder, "ROOT", tmp_path)
    entry = {"url": "https://example.org", "source_file": source}
    with pytest.raises(SystemExit, match="retrieved source"):
        binder.cite_text({"test": entry}, "test")


def test_citation_requires_intact_source_and_support_review(tmp_path, monkeypatch):
    monkeypatch.setattr(binder, "ROOT", tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Actual retrieved source")
    entry = {"url": "https://example.org", "source_file": "source.txt",
             "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    with pytest.raises(SystemExit, match="checked for support"):
        binder.cite_text({"test": entry}, "test")
    entry["citation_support_reviewed"] = True
    assert binder.cite_text({"test": entry}, "test") == "test"
    source.write_text("Changed source")
    with pytest.raises(SystemExit, match="hash"):
        binder.cite_text({"test": entry}, "test")


def test_retrieve_preserves_full_pdf_bytes(monkeypatch):
    data = b"%PDF-1.7\n" + bytes(range(256)) * 600

    class Response:
        status = 200
        headers = {"Content-Type": "application/pdf"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def geturl(self): return "https://example.org/paper.pdf"
        def read(self, limit): return data[:limit]

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: Response())
    result = fetch.retrieve("https://example.org/paper.pdf")
    assert result["body"] == data
    assert result["pdf"] is True
    assert result["text"] == ""
    monkeypatch.setattr(fetch, "MAX_SOURCE_BYTES", 100)
    oversized = fetch.retrieve("https://example.org/paper.pdf")
    assert oversized["ok"] is False
    assert oversized["error"] == "ValueError: source exceeds byte limit; refusing a truncated citation"


@pytest.mark.parametrize("extra_byte", [False, True])
def test_retrieve_accepts_exact_limit_and_rejects_one_extra_byte(monkeypatch, extra_byte):
    maximum = 256
    body = b"%PDF-" + b"x" * (maximum - 5 + int(extra_byte))
    read_limits = []
    class Response:
        status = 200
        headers = {"Content-Type": "application/pdf"}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def geturl(self): return "https://example.org/exact.pdf"
        def read(self, limit):
            read_limits.append(limit)
            return body[:limit]
    monkeypatch.setattr(fetch, "MAX_SOURCE_BYTES", maximum)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    result = fetch.retrieve("https://example.org/exact.pdf")
    assert read_limits == [maximum + 1]
    if extra_byte:
        assert result["ok"] is False
        assert result["error"] == "ValueError: source exceeds byte limit; refusing a truncated citation"
        assert "body" not in result
    else:
        assert result["ok"] is True
        assert result["body"] == body
        assert len(result["body"]) == maximum


@pytest.mark.parametrize("content_type, body, extension", [
    ("application/pdf", b"%PDF-1.7\r\n" + bytes(range(256)), ".pdf"),
    ("text/html; charset=utf-8", b"<title>Fixture</title>\r\n" + b"\xff" * 220, ".html"),
    ("text/plain", b"Synthetic source\r\n" + b"\xfe" * 220, ".txt"),
])
def test_remote_main_preserves_source_extension_bytes_hash_and_unreviewed_status(tmp_path, monkeypatch, content_type, body, extension):
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(fetch, "SOURCES", tmp_path / "paper/sources")
    monkeypatch.setattr(fetch, "FUNDING", tmp_path / "paper/funding")
    monkeypatch.setattr(fetch, "apply_env", lambda: None)
    monkeypatch.setattr(fetch, "LOCALS", [])
    monkeypatch.setattr(fetch, "FUNDING_PAGES", [])
    monkeypatch.setattr(fetch, "SEEDS", [{"key": "fixture", "title": "Synthetic source", "label": "Fixture",
                                        "url": "https://example.org/requested"}])
    pdf = body.startswith(b"%PDF-")
    monkeypatch.setattr(fetch, "retrieve", lambda url: {"ok": True, "status": 200, "url": "https://example.org/final",
        "content_type": content_type, "pdf": pdf, "body": body, "text": "" if pdf else body.decode("utf-8", errors="replace")})
    assert fetch.main() == 0
    entry = json.loads((tmp_path / "paper/references.json").read_text(encoding="utf-8"))["fixture"]
    source = tmp_path / entry["source_file"]
    assert source.suffix == extension
    assert source.read_bytes() == body
    assert entry["source_sha256"] == hashlib.sha256(body).hexdigest()
    assert entry["citation_support_reviewed"] is False
    assert entry["local_source"] is False
    assert entry["retrieved_url"] == "https://example.org/final"
    assert entry["content_type"] == content_type


def test_local_source_is_copied_not_replaced_with_hash_stub(tmp_path, monkeypatch):
    source = tmp_path / "input.pdf"
    source.write_bytes(b"%PDF-1.7\nexact local bytes")
    destination = tmp_path / "paper" / "sources"
    destination.mkdir(parents=True)
    monkeypatch.setattr(fetch, "ROOT", tmp_path)
    monkeypatch.setattr(fetch, "SOURCES", destination)
    entry = fetch.keep_local({"path": str(source), "key": "test", "title": "test", "label": "test"})
    assert (tmp_path / entry["source_file"]).read_bytes() == source.read_bytes()
    assert entry["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert entry["citation_support_reviewed"] is False


@pytest.fixture
def publication(tmp_path, monkeypatch):
    """Synthetic review declarations are test fixtures, never study approvals."""
    monkeypatch.setattr(binder, "ROOT", tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    paper = tmp_path / "paper"
    paper.mkdir()
    source = paper / "source.txt"
    source.write_bytes(b"Synthetic citation source.\n")
    references = paper / "references.json"
    references.write_text(json.dumps({"fixture": {
        "url": "https://example.org/fixture", "label": "Fixture citation",
        "source_file": "paper/source.txt", "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "citation_support_reviewed": True,
    }}), encoding="utf-8")
    metric = tmp_path / "runs" / "nested" / "pilot" / "metrics.json"
    metric.parent.mkdir(parents=True)
    write_fixture_metrics(metric, 1.25)
    manuscript = paper / "manuscript.md"
    manuscript.write_bytes(b"Pilot rate {{m:nested/pilot:rate}} ({{c:fixture}}).\r\n")
    resolved = "Pilot rate 1.25 (Fixture citation).\n"

    def artifact(path, kind):
        return {"path": path.relative_to(tmp_path).as_posix(), "kind": kind,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "hash_convention": "raw_bytes"}

    manifest = {"schema_version": 1, "artifacts": {
        "timing": artifact(metric, "metrics"), "references": artifact(references, "json"),
        "citation": artifact(source, "file"),
    }, "claims": [{"id": "fixture", "status": "observed", "statement": "Synthetic fixture rate.",
                   "scope": "CPU unit test", "limitations": "No real experiment or review",
                   "evidence": [{"artifact": "timing", "selector": ["rate"], "equals": 1.25}],
                   "human_review": {"approved": True, "reviewer": "synthetic test fixture", "date": "2026-10-21"}}],
        "manuscript": {"path": "paper/manuscript.md", "sha256": hashlib.sha256(manuscript.read_bytes()).hexdigest()},
        "resolved_manuscript_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest()}
    return manifest, metric, references, paper, resolved


def write_fixture_metrics(path, rate):
    data = (json.dumps({"complete": True, "rate": rate}) + "\n").encode("utf-8")
    path.write_bytes(data)
    path.with_name("metrics.json.sha256").write_text(hashlib.sha256(data).hexdigest())


def test_publication_build_emits_exact_reviewed_resolved_bytes(publication, monkeypatch):
    manifest, _, _, paper, resolved = publication
    (paper / "claim_evidence.json").write_text(json.dumps(manifest))
    rendered = []
    monkeypatch.setattr(binder, "build_pdf", lambda text, path: rendered.append(text))
    monkeypatch.setattr(binder, "render_pages", lambda *args: None)
    assert binder.main() == 0
    assert rendered == [resolved]
    assert (paper / "manuscript.resolved.md").read_bytes() == resolved.encode("utf-8")


@pytest.mark.parametrize("binding, reason", [
    ("old_hash", "publication evidence gate failed: artifact hash mismatch: timing"),
    ("new_hash", "human review must bind the exact resolved manuscript UTF-8/LF bytes"),
    ("unlisted", "unlisted"),
])
def test_changed_metrics_with_valid_sidecar_cannot_reuse_review(publication, monkeypatch, binding, reason):
    manifest, metric, _, paper, _ = publication
    old_manuscript_hash = manifest["manuscript"]["sha256"]
    write_fixture_metrics(metric, 9.75)
    assert binder.metrics_ok(metric)["ok"]
    if binding == "new_hash":
        manifest["artifacts"]["timing"]["sha256"] = hashlib.sha256(metric.read_bytes()).hexdigest()
        manifest["claims"][0]["evidence"][0]["equals"] = 9.75
    elif binding == "unlisted":
        del manifest["artifacts"]["timing"]
        manifest["claims"][0]["evidence"] = [{"artifact": "citation"}]
    (paper / "claim_evidence.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(binder, "build_pdf", lambda *args: pytest.fail("Rejected inputs reached PDF build"))
    with pytest.raises(SystemExit) as caught:
        binder.main()
    if binding == "unlisted":
        reason = f"publication input is not bound as a validated artifact: {metric.resolve()}"
    assert str(caught.value) == reason
    assert hashlib.sha256((paper / "manuscript.md").read_bytes()).hexdigest() == old_manuscript_hash
    assert not (paper / "manuscript.resolved.md").exists()


@pytest.mark.parametrize("artifact", ["timing", "references", "citation"])
def test_every_substitution_input_must_be_bound(publication, artifact):
    manifest, metric, references, paper, _ = publication
    del manifest["artifacts"][artifact]
    manifest["claims"][0]["evidence"] = [{"artifact": next(iter(manifest["artifacts"]))}]
    with pytest.raises(SystemExit) as caught:
        binder.resolve_publication(manifest)
    missing_path = {"timing": metric, "references": references, "citation": paper / "source.txt"}[artifact]
    assert str(caught.value) == f"publication input is not bound as a validated artifact: {missing_path.resolve()}"


def test_reference_metadata_change_requires_new_binding(publication):
    manifest, _, references, _, _ = publication
    updated = json.loads(references.read_text())
    updated["fixture"]["url"] = "https://example.org/other-source"
    references.write_text(json.dumps(updated))
    with pytest.raises(SystemExit, match="artifact hash mismatch"):
        binder.resolve_publication(manifest)


def test_updated_reference_label_cannot_reuse_resolved_review(publication):
    manifest, _, references, _, _ = publication
    updated = json.loads(references.read_text())
    updated["fixture"]["label"] = "Unreviewed label"
    references.write_text(json.dumps(updated))
    manifest["artifacts"]["references"]["sha256"] = hashlib.sha256(references.read_bytes()).hexdigest()
    with pytest.raises(SystemExit, match="exact resolved manuscript"):
        binder.resolve_publication(manifest)


@pytest.mark.parametrize("missing, reason", [
    ("resolved_digest", "human review must bind the exact resolved manuscript UTF-8/LF bytes"),
    ("human_review", "publication evidence gate failed: publication claim unresolved or not human-reviewed: fixture"),
    ("planned", "publication evidence gate failed: publication claim unresolved or not human-reviewed: fixture"),
])
def test_incomplete_review_cannot_build_publication(publication, missing, reason):
    manifest, _, _, _, _ = publication
    if missing == "resolved_digest":
        del manifest["resolved_manuscript_sha256"]
    elif missing == "human_review":
        del manifest["claims"][0]["human_review"]
    else:
        manifest["claims"][0]["status"] = "planned"
    with pytest.raises(SystemExit) as caught:
        binder.resolve_publication(manifest)
    assert str(caught.value) == reason
