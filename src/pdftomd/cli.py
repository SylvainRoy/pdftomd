from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .converters import ENGINES, ConversionError, get_converter
from .converters.gemini import DEFAULT_MODEL
from .sync import PlannedItem, Syncer, SyncPlan

app = typer.Typer(
    help="Convert documents (PDF, scans, ...) to Markdown. One-way, incremental, source is never modified.",
    no_args_is_help=True,
    add_completion=False,
)

Engine = typer.Option("marker", "--engine", "-e", help=f"Conversion backend: {', '.join(ENGINES)}.", case_sensitive=False)
ForceOcr = typer.Option(False, "--force-ocr", help="(marker) Re-OCR every page (best for poor scans).")
UseLlm = typer.Option(False, "--use-llm", help="(marker) Hybrid LLM mode for better tables/forms (needs GOOGLE_API_KEY).")
Langs = typer.Option(None, "--lang", help="(marker) OCR language(s), e.g. --lang en --lang fr.")
GeminiModel = typer.Option(DEFAULT_MODEL, "--gemini-model", help="(gemini) Model name.")
GeminiKey = typer.Option(None, "--gemini-api-key", envvar="GEMINI_API_KEY", help="(gemini) API key.", show_default=False)


def _converter(engine: str, force_ocr: bool, use_llm: bool, langs: Optional[list[str]], model: str, api_key: Optional[str]):
    engine = engine.lower()
    if engine not in ENGINES:
        raise typer.BadParameter(f"engine must be one of {ENGINES}")
    if engine == "marker":
        return get_converter("marker", force_ocr=force_ocr, use_llm=use_llm, languages=langs or None)
    return get_converter("gemini", model=model, api_key=api_key)


def _print_plan(plan: SyncPlan, *, verbose: bool) -> None:
    for item in plan.to_generate:
        typer.echo(f"[{item.reason.value:>14}] {item.rel_path}")
    if verbose:
        for rel in plan.up_to_date:
            typer.echo(f"[    up-to-date] {rel}")
        for rel in plan.unsupported:
            typer.echo(f"[   unsupported] {rel}")
    for orphan in plan.orphans:
        typer.echo(f"[        orphan] {orphan}")
    typer.echo(
        f"-- {len(plan.to_generate)} to generate, {len(plan.up_to_date)} up to date, "
        f"{len(plan.unsupported)} unsupported, {len(plan.orphans)} orphan output(s)",
        err=True,
    )


def _run(syncer: Syncer, plan: SyncPlan, prune: bool) -> int:
    def progress(item: PlannedItem, i: int, n: int) -> None:
        typer.echo(f"[{i}/{n}] {item.rel_path} ({item.reason.value})", err=True)

    def error(item: PlannedItem, exc: Exception) -> None:
        typer.secho(f"  FAILED {item.rel_path}: {exc}", fg=typer.colors.RED, err=True)

    result = syncer.execute(plan, prune=prune, on_progress=progress, on_error=error)
    typer.echo(f"-- generated {len(result.generated)}, failed {len(result.failed)}, pruned {len(result.pruned)}", err=True)
    return 1 if result.failed else 0


@app.callback()
def _main(version: bool = typer.Option(False, "--version", is_eager=True)) -> None:
    if version:
        typer.echo(f"pdftomd {__version__}")
        raise typer.Exit()


@app.command()
def sync(
    source: Path = typer.Argument(..., exists=True, file_okay=False, readable=True, help="Source directory (read-only)."),
    dest: Path = typer.Argument(..., help="Destination directory for .md files."),
    engine: str = Engine,
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="List what would be (re)generated and exit."),
    force: bool = typer.Option(False, "--force", "-f", help="Regenerate everything even if up to date."),
    select: Optional[list[str]] = typer.Option(None, "--select", "-s", help="Regenerate only these source files (relative to SOURCE); implies force for them. Repeatable."),
    prune: bool = typer.Option(False, "--prune", help="Delete destination .md files whose source no longer exists."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also list up-to-date and unsupported files."),
    force_ocr: bool = ForceOcr,
    use_llm: bool = UseLlm,
    lang: Optional[list[str]] = Langs,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
) -> None:
    """Synchronise SOURCE into DEST, regenerating only stale Markdown files."""
    conv = _converter(engine, force_ocr, use_llm, lang, gemini_model, gemini_api_key)
    syncer = Syncer(source, dest, conv)
    try:
        plan = syncer.plan(force=force, select=select or None)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    _print_plan(plan, verbose=verbose or dry_run)
    if dry_run:
        raise typer.Exit()
    raise typer.Exit(_run(syncer, plan, prune))


@app.command("list")
def list_cmd(
    source: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    dest: Path = typer.Argument(...),
    engine: str = Engine,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also list up-to-date and unsupported files."),
) -> None:
    """List the documents that would be (re)generated, without converting anything."""
    conv = _converter(engine, False, False, None, DEFAULT_MODEL, None)
    _print_plan(Syncer(source, dest, conv).plan(), verbose=verbose)


@app.command()
def convert(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Document to convert."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write Markdown here instead of stdout."),
    engine: str = Engine,
    force_ocr: bool = ForceOcr,
    use_llm: bool = UseLlm,
    lang: Optional[list[str]] = Langs,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
) -> None:
    """Convert a single document and print the Markdown to stdout."""
    conv = _converter(engine, force_ocr, use_llm, lang, gemini_model, gemini_api_key)
    try:
        markdown = conv.convert_file(file)
    except ConversionError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, "utf-8")
    else:
        sys.stdout.write(markdown)


@app.command()
def watch(
    source: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    dest: Path = typer.Argument(...),
    engine: str = Engine,
    interval: float = typer.Option(10.0, "--interval", "-i", help="Seconds between scans of SOURCE."),
    prune: bool = typer.Option(False, "--prune", help="Delete destination .md files whose source disappeared."),
    force_ocr: bool = ForceOcr,
    use_llm: bool = UseLlm,
    lang: Optional[list[str]] = Langs,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
) -> None:
    """Keep DEST in sync with SOURCE, re-scanning periodically until interrupted."""
    conv = _converter(engine, force_ocr, use_llm, lang, gemini_model, gemini_api_key)
    syncer = Syncer(source, dest, conv)
    typer.echo(f"watching {syncer.source_dir} every {interval:g}s (Ctrl-C to stop)", err=True)
    try:
        while True:
            plan = syncer.plan()
            if plan.to_generate or (prune and plan.orphans):
                _run(syncer, plan, prune)
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("stopped", err=True)


if __name__ == "__main__":  # pragma: no cover
    app()
