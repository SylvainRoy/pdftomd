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
