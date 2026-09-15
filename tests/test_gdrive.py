from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pdftomd import MANIFEST_NAME, Reason, Syncer, convert_file
from pdftomd.gdrive import (
    MIME_DOC,
    MIME_MD,
    MIME_PDF,
    MIME_SHEET,
    MIME_SLIDES,
    MIME_XLSX,
    GoogleDriveError,
    GoogleDriveResolver,
    Stub,
    read_stub,
    xlsx_to_markdown,
)
from tests.test_sync import FakeConverter


def make_xlsx(sheets: dict[str, list[list]]) -> bytes:
    from openpyxl import Workbook

    wb = Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        ws = wb.create_sheet(title)
        for row in rows:
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


class FakeDrive:
    """In-memory Drive: doc_id -> (mime, name, version, exports{mime: bytes})."""

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.metadata_calls = 0
        self.export_calls: list[tuple[str, str]] = []

    def add(self, doc_id: str, mime: str, name: str, exports: dict[str, bytes], version: int = 1) -> None:
        self.docs[doc_id] = {"mime": mime, "name": name, "version": version, "exports": exports}

    def bump(self, doc_id: str, **exports: bytes) -> None:
        self.docs[doc_id]["version"] += 1
        self.docs[doc_id]["exports"].update(exports)

    def metadata(self, stub: Stub) -> dict:
        self.metadata_calls += 1
        d = self.docs.get(stub.doc_id)
        if d is None:
            raise GoogleDriveError(f"Google Drive API 404 for {stub.doc_id}")
        return {"id": stub.doc_id, "name": d["name"], "mimeType": d["mime"], "version": str(d["version"]), "modifiedTime": f"2026-01-0{d['version']}T00:00:00Z"}

    def export(self, stub: Stub, mime: str) -> bytes:
        self.export_calls.append((stub.doc_id, mime))
        return self.docs[stub.doc_id]["exports"][mime]


def write_stub(path: Path, doc_id: str, resource_key: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"": "WARNING! DO NOT EDIT", "doc_id": doc_id, "resource_key": resource_key, "email": "x@y"}))
    return path


@pytest.fixture
def drive() -> FakeDrive:
    d = FakeDrive()
    d.add("DOC1", MIME_DOC, "Meeting notes", {MIME_MD: b"# Notes\n\n\n\nHello\r\n"})
    d.add("SHEET1", MIME_SHEET, "Budget", {MIME_XLSX: make_xlsx({"Q1": [["Item", "Cost"], ["Rent", 1000.0], ["Food|Drinks", 200.5]], "Empty": []})})
    d.add("SLIDES1", MIME_SLIDES, "Deck", {MIME_PDF: b"%PDF-fake"})
    return d


@pytest.fixture
def tree(tmp_path: Path):
    src = tmp_path / "src"
    write_stub(src / "notes.gdoc", "DOC1")
    write_stub(src / "finance" / "budget.gsheet", "SHEET1")
    write_stub(src / "deck.gslides", "SLIDES1")
    (src / "scan.pdf").write_bytes(b"pdf")
    return src, tmp_path / "dst"


# -- unit ----------------------------------------------------------------------


def test_read_stub_variants(tmp_path):
    p = write_stub(tmp_path / "a.gdoc", "ABC", resource_key="rk")
    s = read_stub(p)
    assert s == Stub("ABC", "rk") and s.headers == {"X-Goog-Drive-Resource-Keys": "ABC/rk"}
    (tmp_path / "old.gdoc").write_text(json.dumps({"url": "https://docs.google.com/document/d/OLDID/edit"}))
    assert read_stub(tmp_path / "old.gdoc").doc_id == "OLDID"
    (tmp_path / "bad.gdoc").write_text("not json")
    with pytest.raises(GoogleDriveError):
        read_stub(tmp_path / "bad.gdoc")


def test_xlsx_to_markdown():
    md = xlsx_to_markdown(make_xlsx({"Q1": [["Item", "Cost", None], ["Rent", 1000.0, None], [None, None, None], ["Food|Drinks", "a\nb", None]], "Empty": []}), title="Budget")
    assert md == (
        "# Budget\n\n"
        "## Q1\n\n"
        "| Item | Cost |\n|---|---|\n| Rent | 1000 |\n| Food\\|Drinks | a<br>b |\n\n"
        "## Empty\n\n_(empty sheet)_\n"
    )


def test_resolver_doc_sheet_slides(tree, drive):
    src, _ = tree
    r = GoogleDriveResolver(drive)
    doc = r.resolve(src / "notes.gdoc")
    assert doc.markdown == "# Notes\n\nHello\n" and doc.data is None and doc.fingerprint.startswith("v1@")
    sheet = r.resolve(src / "finance" / "budget.gsheet")
    assert "| Rent | 1000 |" in sheet.markdown and "Food\\|Drinks" in sheet.markdown
    deck = r.resolve(src / "deck.gslides")
    assert deck.markdown is None and deck.data == b"%PDF-fake" and deck.filename == "deck.pdf"
    # metadata is cached per doc within a run
    assert drive.metadata_calls == 3


def test_resolver_rejects_unknown_type(tmp_path, drive):
    drive.add("FORM", "application/vnd.google-apps.form", "Survey", {})
    with pytest.raises(GoogleDriveError, match="unsupported"):
        GoogleDriveResolver(drive).describe(write_stub(tmp_path / "s.gform", "FORM"))


def test_convert_file_on_stub(tree, drive):
    src, _ = tree
    assert convert_file(src / "notes.gdoc", gdrive=GoogleDriveResolver(drive)) == "# Notes\n\nHello\n"


# -- sync integration ----------------------------------------------------------


def test_sync_mixes_local_and_remote(tree, drive):
    src, dst = tree
    conv = FakeConverter()
    syncer = Syncer(src, dst, conv, gdrive=GoogleDriveResolver(drive))
    plan = syncer.plan()
    assert {i.rel_path for i in plan.to_generate} == {"notes.gdoc", "finance/budget.gsheet", "deck.gslides", "scan.pdf"}
    result = syncer.execute(plan)
    assert not result.failed
    assert (dst / "notes.md").read_text() == "# Notes\n\nHello\n"
    assert (dst / "finance" / "budget.md").read_text().startswith("# Budget\n")
    assert (dst / "deck.md").read_text().startswith("# deck.pdf")  # went through the engine
    assert conv.calls == ["scan.pdf", "deck.pdf"] or conv.calls == ["deck.pdf", "scan.pdf"]
    manifest = json.loads((dst / MANIFEST_NAME).read_text())["files"]
    assert manifest["notes.gdoc"]["engine"] == "gdrive"
    assert manifest["deck.gslides"]["engine"] == "fake"
    assert manifest["notes.gdoc"]["fingerprint"].startswith("v1@")


def test_remote_edit_detected_without_stub_change(tree, drive):
    src, dst = tree
    Syncer(src, dst, FakeConverter(), gdrive=GoogleDriveResolver(drive)).sync()
    stub_before = (src / "notes.gdoc").read_bytes()
    drive.bump("DOC1", **{MIME_MD: b"# Notes v2\n"})
    syncer = Syncer(src, dst, FakeConverter(), gdrive=GoogleDriveResolver(drive))
    plan = syncer.plan()
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("notes.gdoc", Reason.CHANGED)]
    assert len(plan.up_to_date) == 3
    syncer.execute(plan)
    assert (dst / "notes.md").read_text() == "# Notes v2\n"
    assert (src / "notes.gdoc").read_bytes() == stub_before  # source untouched


def test_no_export_when_up_to_date(tree, drive):
    src, dst = tree
    Syncer(src, dst, FakeConverter(), gdrive=GoogleDriveResolver(drive)).sync()
    n = len(drive.export_calls)
    Syncer(src, dst, FakeConverter(), gdrive=GoogleDriveResolver(drive)).sync()
    assert len(drive.export_calls) == n  # only metadata calls on second run


def test_engine_change_only_affects_slides(tree, drive):
    src, dst = tree
    Syncer(src, dst, FakeConverter(), gdrive=GoogleDriveResolver(drive)).sync()

    class Other(FakeConverter):
        name = "other"

    plan = Syncer(src, dst, Other(), gdrive=GoogleDriveResolver(drive)).plan()
    assert {i.rel_path for i in plan.to_generate} == {"deck.gslides", "scan.pdf"}


def test_metadata_failure_is_reported_not_fatal(tree, drive):
    src, dst = tree
    write_stub(src / "gone.gdoc", "MISSING")
    syncer = Syncer(src, dst, FakeConverter(), gdrive=GoogleDriveResolver(drive))
    plan = syncer.plan()
    assert set(plan.errors) == {"gone.gdoc"} and "404" in plan.errors["gone.gdoc"]
    assert len(plan.to_generate) == 4
    result = syncer.execute(plan)
    assert set(result.failed) == {"gone.gdoc"} and len(result.generated) == 4


def test_stubs_unsupported_without_resolver(tree):
    src, dst = tree
    plan = Syncer(src, dst, FakeConverter()).plan()
    assert set(plan.unsupported) == {"notes.gdoc", "finance/budget.gsheet", "deck.gslides"}


def test_manifest_v1_is_readable(tmp_path):
    from pdftomd.sync import Manifest

    p = tmp_path / MANIFEST_NAME
    p.write_text(json.dumps({"version": 1, "files": {"a.pdf": {"sha256": "abc", "size": 1, "mtime": 1.0, "engine": "marker", "output": "a.md", "generated_at": 1.0}}}))
    assert Manifest(p).entries["a.pdf"].fingerprint == "abc"
