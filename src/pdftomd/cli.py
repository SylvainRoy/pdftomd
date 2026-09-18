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
    help="Convert documents (PDF, scans, Google Docs, ...) to Markdown. One-way, incremental, source is never modified.",
    no_args_is_help=True,
    add_completion=False,
)
gdrive_app = typer.Typer(help="Google Drive account setup (for .gdoc / .gsheet / .gslides stubs).", no_args_is_help=True)
app.add_typer(gdrive_app, name="gdrive")

Engine = typer.Option("marker", "--engine", "-e", help=f"Conversion backend: {', '.join(ENGINES)}.", case_sensitive=False)
ForceOcr = typer.Option(False, "--force-ocr", help="(marker) Re-OCR every page (best for poor scans).")
UseLlm = typer.Option(False, "--use-llm", help="(marker) Hybrid LLM mode for better tables/forms (needs GOOGLE_API_KEY).")
Langs = typer.Option(None, "--lang", help="(marker) OCR language(s), e.g. --lang en --lang fr.")
GeminiModel = typer.Option(DEFAULT_MODEL, "--gemini-model", help="(gemini) Model name.")
GeminiKey = typer.Option(None, "--gemini-api-key", envvar="GEMINI_API_KEY", help="(gemini) API key.", show_default=False)
ExcludeName = typer.Option(None, "--exclude-name", "-x", help="Skip files/directories whose name fully matches this regex (any depth). Repeatable.")
ExcludePath = typer.Option(None, "--exclude-path", "-X", help="Skip files/directories whose source-relative path matches this regex (directories end with '/'). Repeatable.")
GDrive = typer.Option(
    None,
    "--gdrive/--no-gdrive",
    help="Fetch .gdoc/.gsheet/.gslides stubs from Google Drive. Default: on when logged in (`pdftomd gdrive login`).",
    show_default=False,
)


def _converter(engine: str, force_ocr: bool, use_llm: bool, langs: Optional[list[str]], model: str, api_key: Optional[str]):
    engine = engine.lower()
    if engine not in ENGINES:
        raise typer.BadParameter(f"engine must be one of {ENGINES}")
    if engine == "marker":
        return get_converter("marker", force_ocr=force_ocr, use_llm=use_llm, languages=langs or None)
    return get_converter("gemini", model=model, api_key=api_key)


def _resolver(enabled: Optional[bool]):
    from . import gdrive

    if enabled is None:
        enabled = gdrive.has_token()
    if not enabled:
        return None
    if not gdrive.has_token():
        typer.secho("Google Drive not configured; run `pdftomd gdrive login`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    return gdrive.GoogleDriveResolver()


def _syncer(source: Path, dest: Path, conv, gdrive, exclude_name: Optional[list[str]], exclude_path: Optional[list[str]]) -> Syncer:
    try:
        return Syncer(source, dest, conv, gdrive=gdrive, exclude_name=exclude_name, exclude_path=exclude_path)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _print_plan(plan: SyncPlan, *, verbose: bool) -> None:
    for item in plan.to_generate:
        typer.echo(f"[{item.reason.value:>14}] {item.rel_path}")
    if verbose:
        for rel in plan.up_to_date:
            typer.echo(f"[    up-to-date] {rel}")
        for rel in plan.unsupported:
            typer.echo(f"[   unsupported] {rel}")
        for rel in plan.excluded:
            typer.echo(f"[      excluded] {rel}")
    for orphan in plan.orphans:
        typer.echo(f"[        orphan] {orphan}")
    for rel, msg in plan.errors.items():
        typer.secho(f"[         error] {rel}: {msg}", fg=typer.colors.RED)
    typer.echo(
        f"-- {len(plan.to_generate)} to generate, {len(plan.up_to_date)} up to date, "
        f"{len(plan.unsupported)} unsupported, {len(plan.excluded)} excluded, "
        f"{len(plan.orphans)} orphan output(s), {len(plan.errors)} error(s)",
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
    exclude_name: Optional[list[str]] = ExcludeName,
    exclude_path: Optional[list[str]] = ExcludePath,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also list up-to-date and unsupported files."),
    use_gdrive: Optional[bool] = GDrive,
    force_ocr: bool = ForceOcr,
    use_llm: bool = UseLlm,
    lang: Optional[list[str]] = Langs,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
) -> None:
    """Synchronise SOURCE into DEST, regenerating only stale Markdown files."""
    conv = _converter(engine, force_ocr, use_llm, lang, gemini_model, gemini_api_key)
    syncer = _syncer(source, dest, conv, _resolver(use_gdrive), exclude_name, exclude_path)
    try:
        plan = syncer.plan(force=force, select=select or None)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    _print_plan(plan, verbose=verbose or dry_run)
    if dry_run:
        raise typer.Exit(1 if plan.errors else 0)
    raise typer.Exit(_run(syncer, plan, prune))


@app.command("list")
def list_cmd(
    source: Path = typer.Argument(..., exists=True, file_okay=False, readable=True),
    dest: Path = typer.Argument(...),
    engine: str = Engine,
    exclude_name: Optional[list[str]] = ExcludeName,
    exclude_path: Optional[list[str]] = ExcludePath,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also list up-to-date and unsupported files."),
    use_gdrive: Optional[bool] = GDrive,
) -> None:
    """List the documents that would be (re)generated, without converting anything."""
    conv = _converter(engine, False, False, None, DEFAULT_MODEL, None)
    plan = _syncer(source, dest, conv, _resolver(use_gdrive), exclude_name, exclude_path).plan()
    _print_plan(plan, verbose=verbose)
    raise typer.Exit(1 if plan.errors else 0)


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
    """Convert a single document (or Google Drive stub) and print the Markdown to stdout."""
    from .gdrive import GoogleDriveResolver, is_stub

    conv = _converter(engine, force_ocr, use_llm, lang, gemini_model, gemini_api_key)
    try:
        if is_stub(file):
            doc = GoogleDriveResolver().resolve(file)
            markdown = doc.markdown if doc.markdown is not None else conv.convert_bytes(doc.data, filename=doc.filename)
        else:
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
    exclude_name: Optional[list[str]] = ExcludeName,
    exclude_path: Optional[list[str]] = ExcludePath,
    use_gdrive: Optional[bool] = GDrive,
    force_ocr: bool = ForceOcr,
    use_llm: bool = UseLlm,
    lang: Optional[list[str]] = Langs,
    gemini_model: str = GeminiModel,
    gemini_api_key: Optional[str] = GeminiKey,
) -> None:
    """Keep DEST in sync with SOURCE, re-scanning periodically until interrupted."""
    conv = _converter(engine, force_ocr, use_llm, lang, gemini_model, gemini_api_key)
    syncer = _syncer(source, dest, conv, _resolver(use_gdrive), exclude_name, exclude_path)
    typer.echo(f"watching {syncer.source_dir} every {interval:g}s (Ctrl-C to stop)", err=True)
    try:
        while True:
            plan = syncer.plan()
            if plan.to_generate or plan.errors or (prune and plan.orphans):
                _run(syncer, plan, prune)
            time.sleep(interval)
    except KeyboardInterrupt:
        typer.echo("stopped", err=True)


@gdrive_app.command("login")
def gdrive_login(
    client_secret: Optional[Path] = typer.Option(None, "--client-secret", help="OAuth 'Desktop app' client JSON downloaded from Google Cloud console."),
) -> None:
    """Authorise read-only access to your Google Drive (opens a browser once)."""
    from . import gdrive

    try:
        path = gdrive.login(client_secret)
    except ConversionError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f"Logged in. Token saved to {path}")


@gdrive_app.command("status")
def gdrive_status() -> None:
    """Show whether Google Drive access is configured and working."""
    from . import gdrive

    typer.echo(f"config dir:    {gdrive.config_dir()}")
    typer.echo(f"client secret: {'present' if gdrive.client_secret_path().exists() else 'missing'} ({gdrive.client_secret_path()})")
    typer.echo(f"token:         {'present' if gdrive.has_token() else 'missing'} ({gdrive.token_path()})")
    if gdrive.has_token():
        try:
            gdrive.load_credentials()
            typer.echo("credentials:   valid")
        except ConversionError as exc:
            typer.secho(f"credentials:   {exc}", fg=typer.colors.RED)
            raise typer.Exit(1)


@gdrive_app.command("logout")
def gdrive_logout() -> None:
    """Forget the cached Google Drive token."""
    from . import gdrive

    gdrive.token_path().unlink(missing_ok=True)
    typer.echo("Token removed.")


if __name__ == "__main__":  # pragma: no cover
    app()
