from __future__ import annotations

import hashlib
import json
import threading
import time
import shutil

from pathlib import Path

import pytest

from pdftomd import MANIFEST_NAME, Reason, Syncer
from pdftomd.converters.base import ConversionError, Converter


class FakeConverter(Converter):
    name = "fake"
    extensions = frozenset({".pdf", ".png"})

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_on = fail_on or set()

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        self.calls.append(filename)
        if filename in self.fail_on:
            raise ConversionError("boom")
        return f"# {filename}\n\n{hashlib.sha256(data).hexdigest()[:8]}\n"


class SlowConverter(FakeConverter):
    def __init__(self, delay: float, fail_on: set[str] | None = None) -> None:
        super().__init__(fail_on)
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return super().convert_bytes(data, filename=filename)
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def widetree(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(8):
        (src / f"f{i}.pdf").write_bytes(f"content-{i}".encode())
    return src, tmp_path / "dst"


@pytest.fixture
def tree(tmp_path: Path):
    src = tmp_path / "src"
    (src / "a" / "b").mkdir(parents=True)
    (src / "root.pdf").write_bytes(b"root")
    (src / "a" / "one.pdf").write_bytes(b"one")
    (src / "a" / "b" / "two.png").write_bytes(b"two")
    (src / "a" / "notes.txt").write_bytes(b"ignored")
    return src, tmp_path / "dst"


@pytest.fixture
def extree(tmp_path: Path):
    src = tmp_path / "src"
    for d in ("old", "sub/old", "older", "old_stuff", "sub", "x.pdf"):
        (src / d).mkdir(parents=True, exist_ok=True)
    (src / "old" / "a.pdf").write_bytes(b"a")
    (src / "sub" / "old" / "b.pdf").write_bytes(b"b")
    (src / "older" / "c.pdf").write_bytes(b"c")
    (src / "older" / "old").write_bytes(b"old-file")
    (src / "old_stuff" / "d.pdf").write_bytes(b"d")
    (src / "sub" / "x.tmp").write_bytes(b"tmp")
    (src / "sub" / "e.pdf").write_bytes(b"e")
    (src / "x.pdf" / "f.pdf").write_bytes(b"f")
    (src / "top.pdf").write_bytes(b"top")
    return src, tmp_path / "dst"


def snapshot(path: Path) -> dict[str, bytes]:
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_initial_sync_mirrors_hierarchy(tree):
    src, dst = tree
    before = snapshot(src)
    conv = FakeConverter()
    syncer = Syncer(src, dst, conv)
    plan = syncer.plan()
    assert {i.rel_path for i in plan.to_generate} == {"root.pdf", "a/one.pdf", "a/b/two.png"}
    assert all(i.reason is Reason.NEW for i in plan.to_generate)
    assert plan.unsupported == ["a/notes.txt"]

    result = syncer.execute(plan)
    assert sorted(result.generated) == ["a/b/two.png", "a/one.pdf", "root.pdf"]
    assert (dst / "root.md").exists()
    assert (dst / "a" / "one.md").exists()
    assert (dst / "a" / "b" / "two.md").read_text().startswith("# two.png")
    assert (dst / MANIFEST_NAME).exists()
    assert snapshot(src) == before  # source untouched


def test_second_run_is_noop(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    assert len(conv.calls) == 3
    plan = Syncer(src, dst, conv).plan()
    assert plan.to_generate == []
    assert len(plan.up_to_date) == 3


def test_changed_content_is_regenerated(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    (src / "a" / "one.pdf").write_bytes(b"one v2")
    plan = Syncer(src, dst, conv).plan()
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("a/one.pdf", Reason.CHANGED)]


def test_touched_but_identical_content_is_not_regenerated(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    import os, time

    os.utime(src / "root.pdf", (time.time() + 100, time.time() + 100))
    assert Syncer(src, dst, conv).plan().to_generate == []


def test_missing_output_is_regenerated(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    (dst / "root.md").unlink()
    plan = Syncer(src, dst, conv).plan()
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("root.pdf", Reason.OUTPUT_MISSING)]


def test_engine_change_is_regenerated(tree):
    src, dst = tree
    Syncer(src, dst, FakeConverter()).sync()

    class Other(FakeConverter):
        name = "other"

    plan = Syncer(src, dst, Other()).plan()
    assert {i.reason for i in plan.to_generate} == {Reason.ENGINE_CHANGED}
    assert len(plan.to_generate) == 3


def test_force_and_select(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()

    plan = Syncer(src, dst, conv).plan(force=True)
    assert len(plan.to_generate) == 3 and all(i.reason is Reason.FORCED for i in plan.to_generate)

    plan = Syncer(src, dst, conv).plan(select=["a/one.pdf"])
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("a/one.pdf", Reason.FORCED)]

    plan = Syncer(src, dst, conv).plan(select=[src / "a" / "b" / "two.png"])  # absolute works too
    assert [i.rel_path for i in plan.to_generate] == ["a/b/two.png"]

    with pytest.raises(FileNotFoundError):
        Syncer(src, dst, conv).plan(select=["nope.pdf"])


def test_failure_does_not_poison_manifest(tree):
    src, dst = tree
    conv = FakeConverter(fail_on={"one.pdf"})
    result = Syncer(src, dst, conv).sync()
    assert set(result.failed) == {"a/one.pdf"}
    assert not (dst / "a" / "one.md").exists()
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    assert "a/one.pdf" not in manifest["files"]
    # Next run retries only the failed one.
    plan = Syncer(src, dst, FakeConverter()).plan()
    assert [i.rel_path for i in plan.to_generate] == ["a/one.pdf"]


def test_orphans_and_prune(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    (src / "root.pdf").unlink()
    plan = Syncer(src, dst, conv).plan()
    assert [p.name for p in plan.orphans] == ["root.md"]
    assert (dst / "root.md").exists()  # plan alone never deletes
    result = Syncer(src, dst, conv).execute(plan, prune=True)
    assert result.pruned == ["root.md"]
    assert not (dst / "root.md").exists()
    assert "root.pdf" not in json.loads((dst / MANIFEST_NAME).read_text())["files"]


def test_moved_source_relocates_markdown_without_converting(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    original = (dst / "a" / "one.md").read_text()
    manifest = json.loads((dst / MANIFEST_NAME).read_text())["files"]
    (src / "moved").mkdir()
    shutil.move(src / "a" / "one.pdf", src / "moved" / "renamed.pdf")

    syncer = Syncer(src, dst, conv)
    plan = syncer.plan()
    assert [(i.rel_path, i.reason, i.moved_from) for i in plan.to_generate] == [
        ("moved/renamed.pdf", Reason.MOVED, "a/one.pdf")
    ]
    assert plan.orphans == []
    assert (dst / "a" / "one.md").exists()  # plan alone never moves

    result = syncer.execute(plan)
    assert result.moved == ["moved/renamed.pdf"] and result.generated == []
    assert len(conv.calls) == 3  # no reconversion
    assert (dst / "moved" / "renamed.md").read_text() == original
    assert not (dst / "a" / "one.md").exists()
    assert (dst / "a" / "b" / "two.md").exists()
    entries = json.loads((dst / MANIFEST_NAME).read_text())["files"]
    assert "a/one.pdf" not in entries
    assert entries["moved/renamed.pdf"]["fingerprint"] == manifest["a/one.pdf"]["fingerprint"]
    assert entries["moved/renamed.pdf"]["output"] == "moved/renamed.md"
    assert Syncer(src, dst, conv).plan().to_generate == []


def test_moved_directory_relocates_all_outputs(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    shutil.move(src / "a", src / "archive")
    result = Syncer(src, dst, conv).sync()
    assert sorted(result.moved) == ["archive/b/two.png", "archive/one.pdf"]
    assert len(conv.calls) == 3
    assert (dst / "archive" / "one.md").exists() and (dst / "archive" / "b" / "two.md").exists()
    assert not (dst / "a").exists()  # emptied directories are dropped


def test_moved_and_modified_source_is_new(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    shutil.move(src / "root.pdf", src / "renamed.pdf")
    (src / "renamed.pdf").write_bytes(b"edited")
    plan = Syncer(src, dst, conv).plan()
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("renamed.pdf", Reason.NEW)]
    assert [p.name for p in plan.orphans] == ["root.md"]


def test_copied_source_is_new_not_moved(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    shutil.copy(src / "root.pdf", src / "copy.pdf")
    plan = Syncer(src, dst, conv).plan()
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("copy.pdf", Reason.NEW)]
    assert plan.orphans == []


def test_move_requires_matching_engine(tree):
    src, dst = tree
    Syncer(src, dst, FakeConverter()).sync()
    shutil.move(src / "root.pdf", src / "renamed.pdf")
    other = FakeConverter()
    other.name = "other"
    plan = Syncer(src, dst, other).plan()
    assert ("renamed.pdf", Reason.NEW) in [(i.rel_path, i.reason) for i in plan.to_generate]
    assert [p.name for p in plan.orphans] == ["root.md"]


def test_move_with_missing_old_output_is_new(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    (dst / "root.md").unlink()
    shutil.move(src / "root.pdf", src / "renamed.pdf")
    plan = Syncer(src, dst, conv).plan()
    assert [(i.rel_path, i.reason) for i in plan.to_generate] == [("renamed.pdf", Reason.NEW)]


def test_duplicate_sources_moved_once_each(tree):
    src, dst = tree
    conv = FakeConverter()
    (src / "dup.pdf").write_bytes(b"root")  # same content as root.pdf
    Syncer(src, dst, conv).sync()
    shutil.move(src / "root.pdf", src / "x.pdf")
    shutil.move(src / "dup.pdf", src / "y.pdf")
    plan = Syncer(src, dst, conv).plan()
    moves = sorted((i.rel_path, i.moved_from) for i in plan.to_generate if i.reason is Reason.MOVED)
    assert [m[0] for m in moves] == ["x.pdf", "y.pdf"]
    assert sorted(m[1] for m in moves) == ["dup.pdf", "root.pdf"]
    assert plan.orphans == []
    Syncer(src, dst, conv).execute(plan)
    assert (dst / "x.md").exists() and (dst / "y.md").exists()
    assert not (dst / "root.md").exists() and not (dst / "dup.md").exists()
    assert len(conv.calls) == 4


def test_prune_removes_empty_directories(tree):
    src, dst = tree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    assert (dst / "a" / "b").is_dir()
    shutil.rmtree(src / "a")
    Syncer(src, dst, conv).sync(prune=True)
    assert not (dst / "a").exists()
    assert (dst / "root.md").exists()
    assert (dst / MANIFEST_NAME).exists()


def test_exclude_name_matches_dirs_and_files(extree):
    src, dst = extree
    plan = Syncer(src, dst, FakeConverter(), exclude_name=["old"]).plan()
    generated = {i.rel_path for i in plan.to_generate}
    assert generated == {"older/c.pdf", "old_stuff/d.pdf", "sub/e.pdf", "top.pdf", "x.pdf/f.pdf"}
    assert plan.excluded == ["old/", "older/old", "sub/old/"]


def test_exclude_name_regex(extree):
    src, dst = extree
    plan = Syncer(src, dst, FakeConverter(), exclude_name=[r".*\.tmp"]).plan()
    assert "sub/x.tmp" not in {i.rel_path for i in plan.to_generate}
    assert "sub/x.tmp" not in plan.unsupported
    assert plan.excluded == ["sub/x.tmp"]


def test_exclude_path(extree):
    src, dst = extree
    plan = Syncer(src, dst, FakeConverter(), exclude_path=[r"^sub/old/"]).plan()
    generated = {i.rel_path for i in plan.to_generate}
    assert "sub/old/b.pdf" not in generated
    assert "old/a.pdf" in generated and "sub/e.pdf" in generated
    assert plan.excluded == ["sub/old/"]

    plan = Syncer(src, dst, FakeConverter(), exclude_path=[r"/$"]).plan()
    assert {i.rel_path for i in plan.to_generate} == {"top.pdf"}
    assert plan.excluded == ["old/", "old_stuff/", "older/", "sub/", "x.pdf/"]


def test_exclude_path_matches_files_but_dirs_still_get_trailing_slash(extree):
    src, dst = extree
    plan = Syncer(src, dst, FakeConverter(), exclude_path=[r"\.pdf$"]).plan()
    assert plan.to_generate == []
    assert "x.pdf/" not in plan.excluded  # dir target is "x.pdf/", still descended
    assert "x.pdf/f.pdf" in plan.excluded
    assert plan.excluded == [
        "top.pdf",
        "old/a.pdf",
        "old_stuff/d.pdf",
        "older/c.pdf",
        "sub/e.pdf",
        "sub/old/b.pdf",
        "x.pdf/f.pdf",
    ]


def test_newly_excluded_sources_become_orphans(extree):
    src, dst = extree
    conv = FakeConverter()
    Syncer(src, dst, conv).sync()
    assert (dst / "old" / "a.md").exists() and (dst / "sub" / "old" / "b.md").exists()

    syncer = Syncer(src, dst, conv, exclude_name=["old"])
    plan = syncer.plan()
    assert sorted(p.relative_to(dst).as_posix() for p in plan.orphans) == ["old/a.md", "sub/old/b.md"]
    assert (dst / "old" / "a.md").exists()  # plan alone never deletes

    result = syncer.execute(plan, prune=True)
    assert sorted(result.pruned) == ["old/a.md", "sub/old/b.md"]
    assert not (dst / "old" / "a.md").exists() and not (dst / "sub" / "old" / "b.md").exists()
    entries = json.loads((dst / MANIFEST_NAME).read_text())["files"]
    assert "old/a.pdf" not in entries and "sub/old/b.pdf" not in entries


def test_select_does_not_override_exclusion(extree):
    src, dst = extree
    with pytest.raises(FileNotFoundError, match="excluded"):
        Syncer(src, dst, FakeConverter(), exclude_name=["old"]).plan(select=["old/a.pdf"])


def test_invalid_exclude_pattern_rejected(extree):
    src, dst = extree
    with pytest.raises(ValueError, match=r"Invalid exclude pattern '\('"):
        Syncer(src, dst, FakeConverter(), exclude_name=["("])


def test_on_done_reports_elapsed_per_file(tree):
    src, dst = tree
    calls = []
    syncer = Syncer(src, dst, FakeConverter())
    syncer.execute(
        syncer.plan(),
        workers=1,
        on_done=lambda item, i, n, elapsed: calls.append((item.rel_path, i, n, elapsed)),
    )
    assert [c[0] for c in calls] == ["root.pdf", "a/one.pdf", "a/b/two.png"]
    assert all(c[2] == 3 for c in calls)
    assert [c[1] for c in calls] == [1, 2, 3]
    assert all(c[3] >= 0 for c in calls)


def test_parallel_execute(widetree):
    src, dst = widetree
    conv = SlowConverter(0.05)
    syncer = Syncer(src, dst, conv)
    started = time.monotonic()
    result = syncer.execute(syncer.plan(), workers=4)
    elapsed = time.monotonic() - started
    assert len(result.generated) == 8
    assert conv.max_active >= 2  # actually parallel
    assert elapsed < 0.3  # 8 * 0.05 sequential would take 0.4s
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    assert len(manifest["files"]) == 8
    for i in range(8):
        assert (dst / f"f{i}.md").read_text().startswith(f"# f{i}.pdf")


def test_non_parallel_safe_converter_uses_one_worker(widetree):
    src, dst = widetree

    class Serial(SlowConverter):
        parallel_safe = False

    conv = Serial(0.01)
    syncer = Syncer(src, dst, conv)
    assert syncer.effective_workers(4) == 1
    result = syncer.execute(syncer.plan(), workers=4)
    assert len(result.generated) == 8
    assert conv.max_active == 1


def test_parallel_failures_do_not_stop_others(widetree):
    src, dst = widetree
    conv = SlowConverter(0.01, fail_on={"f2.pdf", "f5.pdf"})
    syncer = Syncer(src, dst, conv)
    result = syncer.execute(syncer.plan(), workers=4)
    assert set(result.failed) == {"f2.pdf", "f5.pdf"}
    assert len(result.generated) == 6
    assert not (dst / "f2.md").exists() and not (dst / "f5.md").exists()
    manifest = json.loads((dst / MANIFEST_NAME).read_text())
    assert len(manifest["files"]) == 6


def test_on_done_and_on_progress_indices(widetree):
    src, dst = widetree
    done_calls, progress_calls = [], []
    syncer = Syncer(src, dst, FakeConverter())
    syncer.execute(
        syncer.plan(),
        workers=4,
        on_progress=lambda item, i, n: progress_calls.append((item.rel_path, i, n)),
        on_done=lambda item, i, n, elapsed: done_calls.append((item.rel_path, i, n)),
    )
    assert [c[1] for c in done_calls] == list(range(1, 9))
    assert sorted(c[0] for c in done_calls) == [f"f{i}.pdf" for i in range(8)]
    assert [c[1] for c in progress_calls] == list(range(1, 9))
    assert all(c[2] == 8 for c in done_calls + progress_calls)


def test_workers_one_goes_through_pool(widetree):
    src, dst = widetree
    conv = SlowConverter(0.01)
    result = Syncer(src, dst, conv).execute(Syncer(src, dst, conv).plan(), workers=1)
    assert len(result.generated) == 8
    assert conv.max_active == 1


def test_unexpected_worker_exception_propagates(widetree):
    src, dst = widetree

    class Broken(FakeConverter):
        def convert_bytes(self, data: bytes, *, filename: str) -> str:
            raise RuntimeError("not a ConversionError")

    syncer = Syncer(src, dst, Broken())
    with pytest.raises(RuntimeError):
        syncer.execute(syncer.plan(), workers=2)


def test_encrypted_pdf_fails_clearly_and_is_retried(tree):
    import io

    from pypdf import PdfWriter

    from pdftomd.pdfcheck import EncryptedPdfError, ensure_pdf_readable

    class CheckingConverter(FakeConverter):
        def convert_bytes(self, data: bytes, *, filename: str) -> str:
            ensure_pdf_readable(data, filename=filename)
            return super().convert_bytes(data, filename=filename)

    src, dst = tree

    # The fixture's .pdf payloads are fake bytes; replace them with a real PDF
    # so only the encrypted one is rejected by the check.
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    buf = io.BytesIO()
    doc.save(buf)
    real_pdf = buf.getvalue()
    (src / "root.pdf").write_bytes(real_pdf)
    (src / "a" / "one.pdf").write_bytes(real_pdf)

    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.encrypt(user_password="secret", owner_password="owner")
    buf = io.BytesIO()
    w.write(buf)
    (src / "locked.pdf").write_bytes(buf.getvalue())

    syncer = Syncer(src, dst, CheckingConverter())
    result = syncer.execute(syncer.plan())
    assert "locked.pdf" in result.failed
    assert "password-protected" in result.failed["locked.pdf"]
    assert isinstance(EncryptedPdfError("x"), ConversionError)
    assert len(result.generated) == 3
    assert not (dst / "locked.md").exists()

    # Retried on the next run: still reported as new.
    plan = Syncer(src, dst, CheckingConverter()).plan()
    assert [i.rel_path for i in plan.to_generate] == ["locked.pdf"]
    assert plan.to_generate[0].reason is Reason.NEW


def test_dest_inside_source_rejected(tree):
    src, _ = tree
    with pytest.raises(ValueError):
        Syncer(src, src / "out", FakeConverter())


def test_string_level_api_writes_nothing(tmp_path):
    from pdftomd import convert_bytes
    from pdftomd import converters

    conv = FakeConverter()
    converters_get = converters.get_converter
    try:
        converters.get_converter = lambda engine, **kw: conv
        import pdftomd

        pdftomd.get_converter = converters.get_converter
        md = pdftomd.convert_bytes(b"abc", filename="x.pdf", engine="fake")
    finally:
        converters.get_converter = converters_get
        pdftomd.get_converter = converters_get
    assert md.startswith("# x.pdf")
    assert list(tmp_path.iterdir()) == []
