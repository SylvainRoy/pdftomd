from __future__ import annotations

import hashlib
import json
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


@pytest.fixture
def tree(tmp_path: Path):
    src = tmp_path / "src"
    (src / "a" / "b").mkdir(parents=True)
    (src / "root.pdf").write_bytes(b"root")
    (src / "a" / "one.pdf").write_bytes(b"one")
    (src / "a" / "b" / "two.png").write_bytes(b"two")
    (src / "a" / "notes.txt").write_bytes(b"ignored")
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
