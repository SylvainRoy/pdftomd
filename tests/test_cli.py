from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pdftomd import cli
from tests.test_sync import FakeConverter

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


def test_convert_single_file_to_stdout(tree, fake_engine):
    src, _ = tree
    res = runner.invoke(cli.app, ["convert", str(src / "a.pdf")])
    assert res.exit_code == 0
    assert res.output.startswith("# a.pdf")
