import ctypes
import json

import pytest

from pdftomd import MANIFEST_NAME, Syncer
from pdftomd import sync
from pdftomd.gdrive import MIME_DOC, MIME_MD, MIME_SHEET, MIME_SLIDES, GoogleDriveResolver
from tests.test_gdrive import FakeDrive, write_stub
from tests.test_sync import FakeConverter
from tests.test_cli import runner
from pdftomd import cli


@pytest.mark.parametrize("suffix,mime,kind", [
    (".gdoc", MIME_DOC, "document"),
    (".gsheet", MIME_SHEET, "spreadsheets"),
    (".gslides", MIME_SLIDES, "presentation"),
])
@pytest.mark.parametrize("key", ["", "resource-key"])
def test_stub_url(tmp_path, suffix, mime, kind, key):
    src = tmp_path / "src"
    path = write_stub(src / ("doc" + suffix), "SHORTCUT", key)
    drive = FakeDrive()
    drive.add("SHORTCUT", mime, "doc", {})
    original = drive.metadata

    def metadata(stub):
        return {**original(stub), "id": "RESOLVED"}

    drive.metadata = metadata
    syncer = Syncer(src, tmp_path / "dst", FakeConverter(), gdrive=GoogleDriveResolver(drive))
    syncer.gdrive.describe(path)
    expected = f"https://docs.google.com/{kind}/d/RESOLVED/edit"
    if key:
        expected += "?resourcekey=resource-key"
    assert syncer._source_url(path) == expected
    assert drive.metadata_calls == 1


def test_stub_url_saved(tmp_path):
    src = tmp_path / "src"
    write_stub(src / "doc.gdoc", "DOC")
    drive = FakeDrive()
    drive.add("DOC", MIME_DOC, "doc", {MIME_MD: b"# A document with enough content to keep"})
    syncer = Syncer(src, tmp_path / "dst", FakeConverter(), gdrive=GoogleDriveResolver(drive))
    syncer.sync()
    assert syncer.manifest.entries["doc.gdoc"].source_url == "https://docs.google.com/document/d/DOC/edit"
    assert drive.metadata_calls == 1


@pytest.mark.parametrize("value,expected", [(b"drive-id", "drive-id"), (b"", None), (b"\xff", None), (None, None)])
def test_xattr_reader(tmp_path, monkeypatch, value, expected):
    def getxattr(path, name):
        assert name == b"com.google.drivefs.item-id#S"
        if value is None:
            raise OSError("no attribute")
        return value

    monkeypatch.setattr(sync.os, "getxattr", getxattr, raising=False)
    assert sync._read_drive_id(tmp_path / "file.pdf") == expected


@pytest.mark.parametrize("failure", [False, True])
def test_macos_xattr_reader(tmp_path, monkeypatch, failure):
    monkeypatch.delattr(sync.os, "getxattr", raising=False)
    monkeypatch.setattr(sync.sys, "platform", "darwin")

    class Getxattr:
        def __call__(self, path, name, buffer, size, position, options):
            assert name == b"com.google.drivefs.item-id#S"
            assert position == options == 0
            if failure:
                return -1
            if buffer is not None:
                ctypes.memmove(buffer, b"mac-id", 6)
            return 6

    class Libc:
        getxattr = Getxattr()

    monkeypatch.setattr(sync.ctypes, "CDLL", lambda *a, **kw: Libc())
    assert sync._read_drive_id(tmp_path / "file.pdf") == (None if failure else "mac-id")


def test_unsupported_xattr(tmp_path, monkeypatch):
    monkeypatch.delattr(sync.os, "getxattr", raising=False)
    monkeypatch.setattr(sync.sys, "platform", "win32")
    assert sync._read_drive_id(tmp_path / "file.pdf") is None


def test_backfill_move_and_manual(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    path = src / "a.pdf"
    path.write_bytes(b"content")
    dst = tmp_path / "dst"
    conv = FakeConverter()
    monkeypatch.setattr(sync, "_read_drive_id", lambda path: None)
    syncer = Syncer(src, dst, conv)
    syncer.sync()
    manifest = dst / MANIFEST_NAME
    old = json.loads(manifest.read_text())
    del old["files"]["a.pdf"]["source_url"]
    manifest.write_text(json.dumps(old))
    monkeypatch.setattr(sync, "_read_drive_id", lambda path: path.stem + "-id")
    syncer = Syncer(src, dst, conv)
    before = manifest.read_bytes()
    assert syncer.plan().to_generate == []
    assert manifest.read_bytes() == before
    syncer.sync()
    assert conv.calls == ["a.pdf"]
    assert json.loads(manifest.read_text())["files"]["a.pdf"]["source_url"] == "https://drive.google.com/file/d/a-id/view"
    monkeypatch.setattr(sync, "_read_drive_id", lambda path: "changed-id")
    syncer.sync()
    assert syncer.manifest.entries["a.pdf"].source_url.endswith("/changed-id/view")
    monkeypatch.setattr(sync, "_read_drive_id", lambda path: path.stem + "-id")
    path.rename(src / "b.pdf")
    (dst / "notes.md").write_text("# Hand written notes\n")
    result = syncer.sync()
    assert result.moved == ["b.pdf"]
    assert conv.calls == ["a.pdf"]
    assert syncer.manifest.entries["b.pdf"].source_url == "https://drive.google.com/file/d/b-id/view"
    assert syncer.manifest.entries["notes.md"].source_url is None


def test_missing_url_summary_and_dry_run(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.pdf").write_bytes(b"content")
    dst = tmp_path / "dst"
    monkeypatch.setattr(cli, "_converter", lambda *a, **kw: FakeConverter())
    monkeypatch.setattr(sync, "_read_drive_id", lambda path: None)
    args = ["sync", str(src), str(dst)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert result.output.count("1 tracked source(s) without a Drive URL") == 1
    assert "[  no Drive URL]" not in result.output
    result = runner.invoke(cli.app, args + ["--verbose"])
    assert "[  no Drive URL] a.pdf" in result.output
    before = (dst / MANIFEST_NAME).read_bytes()
    monkeypatch.setattr(sync, "_read_drive_id", lambda path: "found")
    result = runner.invoke(cli.app, args + ["--dry-run"])
    assert result.exit_code == 0, result.output
    assert (dst / MANIFEST_NAME).read_bytes() == before
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "without a Drive URL" not in result.output
