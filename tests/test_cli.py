from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pdftomd import cli
from pdftomd.converters.fallback import FallbackConverter
from tests.test_sync import FakeConverter, OutputConverter

runner = CliRunner()


@pytest.fixture
def fake_engine(monkeypatch):
    conv = FakeConverter()
    monkeypatch.setattr(cli, "_converter", lambda *a, **k: conv)
    return conv


@pytest.fixture
def tree(tmp_path: Path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.pdf").write_bytes(b"a")
    (src / "sub" / "b.pdf").write_bytes(b"b")
    return src, tmp_path / "dst"


def test_dry_run_lists_without_writing(tree, fake_engine):
    src, dst = tree
    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "a.pdf" in res.output and "sub/b.pdf" in res.output
    assert not dst.exists()
    assert fake_engine.calls == []


def test_sync_then_list_then_select(tree, fake_engine):
    src, dst = tree
    assert runner.invoke(cli.app, ["sync", str(src), str(dst)]).exit_code == 0
    assert (dst / "sub" / "b.md").exists()
    assert len(fake_engine.calls) == 2

    res = runner.invoke(cli.app, ["list", str(src), str(dst)])
    assert res.exit_code == 0
    assert "0 to generate" in res.output

    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--select", "sub/b.pdf"])
    assert res.exit_code == 0, res.output
    assert fake_engine.calls[-1] == "b.pdf" and len(fake_engine.calls) == 3

    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--select", "missing.pdf"])
    assert res.exit_code == 2


def test_list_adoption_flags_and_overwrite_edited_option(tree, fake_engine):
    src, dst = tree
    dst.mkdir()
    (dst / "notes.md").write_text("# Notes\n", "utf-8")

    res = runner.invoke(cli.app, ["list", str(src), str(dst)])
    assert res.exit_code == 0, res.output
    assert "[       adopted] notes.md" in res.output

    res = runner.invoke(cli.app, ["list", str(src), str(dst), "--no-adopt"])
    assert res.exit_code == 0, res.output
    assert "[        orphan]" in res.output

    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--overwrite-edited", "--dry-run"])
    assert res.exit_code == 0, res.output


def test_moved_file_is_reported_and_relocated(tree, fake_engine):
    src, dst = tree
    assert runner.invoke(cli.app, ["sync", str(src), str(dst)]).exit_code == 0
    (src / "sub" / "b.pdf").rename(src / "c.pdf")

    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "[         moved] sub/b.pdf -> c.pdf" in res.output
    assert "orphan" not in res.output.split("--")[0]
    assert "0 to generate, 1 to move" in res.output

    res = runner.invoke(cli.app, ["sync", str(src), str(dst)])
    assert res.exit_code == 0, res.output
    assert "generated 0, moved 1" in res.output
    assert (dst / "c.md").exists() and not (dst / "sub").exists()
    assert len(fake_engine.calls) == 2


def test_exclude_flags(tree, fake_engine):
    src, dst = tree
    (src / "old").mkdir()
    (src / "old" / "c.pdf").write_bytes(b"c")
    (src / "sub" / "x.pdf").write_bytes(b"x")
    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--dry-run", "-v", "-x", "old", "-X", "^sub/x"])
    assert res.exit_code == 0, res.output
    assert "[      excluded] old/" in res.output
    assert "[      excluded] sub/x.pdf" in res.output
    assert "excluded" in res.output


def test_invalid_exclude_pattern_exits_2(tree, fake_engine):
    src, dst = tree
    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "-x", "("])
    assert res.exit_code == 2
    assert "Invalid exclude pattern" in res.output


def test_workers_option(tree, fake_engine):
    src, dst = tree
    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "--workers", "3"])
    assert res.exit_code == 0, res.output
    assert (dst / "a.md").exists() and (dst / "sub" / "b.md").exists()
    assert "converting with 3 workers" in res.output


def test_workers_min_one(tree, fake_engine):
    src, dst = tree
    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "-j", "0"])
    assert res.exit_code == 2


def test_non_parallel_safe_engine_warns(tree, fake_engine, monkeypatch):
    src, dst = tree
    monkeypatch.setattr(fake_engine, "parallel_safe", False)
    res = runner.invoke(cli.app, ["sync", str(src), str(dst), "-j", "4"])
    assert res.exit_code == 0, res.output
    assert "using 1 worker" in res.output


def test_convert_single_file_to_stdout(tree, fake_engine):
    src, _ = tree
    res = runner.invoke(cli.app, ["convert", str(src / "a.pdf")])
    assert res.exit_code == 0
    assert "# a.pdf" in res.output  # stderr (timing line) is mixed into output by CliRunner


def test_convert_rejects_tiny_engine_output_and_min_chars_can_disable(tree, monkeypatch, tmp_path):
    src, _ = tree
    conv = OutputConverter({"a.pdf": "x"})
    monkeypatch.setattr(cli, "_converter", lambda *a, **k: conv)
    output = tmp_path / "tiny.md"

    result = runner.invoke(cli.app, ["convert", str(src / "a.pdf"), "--output", str(output)])
    assert result.exit_code == 1
    assert "non-whitespace" in result.output
    assert not output.exists()

    result = runner.invoke(
        cli.app,
        ["convert", str(src / "a.pdf"), "--output", str(output), "--min-chars", "0"],
    )
    assert result.exit_code == 0, result.output
    assert output.read_text() == "x"


def test_sync_reports_tiny_engine_output_as_failed(tree, monkeypatch):
    src, dst = tree
    conv = OutputConverter({"a.pdf": "x"})
    monkeypatch.setattr(cli, "_converter", lambda *a, **k: conv)

    result = runner.invoke(cli.app, ["sync", str(src), str(dst)])
    assert result.exit_code == 1
    assert "FAILED a.pdf" in result.output
    assert "non-whitespace" in result.output


def test_gemini_options_reach_converter(tree, monkeypatch):
    src, dst = tree
    captured = {}

    def capture_converter(*args, **kwargs):
        captured.update(kwargs)
        return FakeConverter()

    monkeypatch.setattr(cli, "_converter", capture_converter)
    result = runner.invoke(
        cli.app,
        [
            "sync",
            str(src),
            str(dst),
            "--dry-run",
            "--gemini-fallback-model",
            "gemini-4-argon",
            "--gemini-temperature",
            "0.8",
            "--gemini-chunk-pages",
            "3",
            "--fallback-engine",
            "marker",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["fallback_model"] == "gemini-4-argon"
    assert captured["temperature"] == 0.8
    assert captured["chunk_pages"] == 3
    assert captured["fallback_engine"] == "marker"


def test_fallback_engine_marker_builds_wrapper(monkeypatch):
    created = []

    def fake_get_converter(name, **kwargs):
        converter = FakeConverter()
        converter.name = name
        converter.options = kwargs
        created.append(converter)
        return converter

    monkeypatch.setattr(cli, "get_converter", fake_get_converter)
    converter = cli._converter(
        "gemini",
        True,
        True,
        ["eng"],
        cli.DEFAULT_MODEL,
        None,
        fallback_engine="marker",
    )

    assert isinstance(converter, FallbackConverter)
    assert converter.name == "gemini"
    assert created[0].options["fail_fast_on_recitation"] is True
    assert created[1].options == {"force_ocr": True, "use_llm": True, "languages": ["eng"]}


def test_fallback_engine_foo_is_rejected(tree):
    src, dst = tree
    result = runner.invoke(
        cli.app,
        ["sync", str(src), str(dst), "--engine", "gemini", "--fallback-engine", "foo", "--dry-run"],
    )

    assert result.exit_code == 2
    assert "only 'marker' is supported" in result.output


def test_fallback_engine_is_rejected_for_marker_engine():
    with pytest.raises(cli.typer.BadParameter, match="only valid with --engine gemini"):
        cli._converter(
            "marker",
            False,
            False,
            None,
            cli.DEFAULT_MODEL,
            None,
            fallback_engine="marker",
        )
