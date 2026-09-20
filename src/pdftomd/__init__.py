"""pdftomd: convert documents to Markdown, one file or a whole directory tree.

Library usage::

    from pdftomd import convert_file, convert_bytes, Syncer, get_converter

    md = convert_file("report.pdf", engine="marker", force_ocr=True)
    md = convert_bytes(pdf_bytes, filename="report.pdf", engine="gemini")

    syncer = Syncer("docs/", "docs-md/", get_converter("marker"))
    plan = syncer.plan()             # dry run
    syncer.execute(plan)             # write only what is stale
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .converters import ENGINES, ConversionError, Converter, get_converter
from .sync import (
    MANIFEST_NAME,
    Manifest,
    PlannedItem,
    Reason,
    Syncer,
    SyncPlan,
    SyncResult,
    markdown_path_for,
)

__version__ = "1.0.0"


def convert_bytes(data: bytes, *, filename: str, engine: str = "marker", **options: Any) -> str:
    """String-level API: document bytes in, Markdown string out. Nothing is written to disk."""
    return get_converter(engine, **options).convert_bytes(data, filename=filename)


def convert_file(path: str | Path, *, engine: str = "marker", gdrive: Any = None, **options: Any) -> str:
    """File-level API: read a document from disk and return its Markdown.

    ``.gdoc`` / ``.gsheet`` / ``.gslides`` stubs are fetched from Google Drive
    (``gdrive`` may be a configured ``GoogleDriveResolver``; a default one is
    created otherwise). Slides go through ``engine`` after export to PDF.
    """
    from .gdrive import GoogleDriveResolver, is_stub

    if is_stub(path):
        doc = (gdrive or GoogleDriveResolver()).resolve(path)
        if doc.markdown is not None:
            return doc.markdown
        return get_converter(engine, **options).convert_bytes(doc.data, filename=doc.filename)
    return get_converter(engine, **options).convert_file(path)


__all__ = [
    "ENGINES",
    "MANIFEST_NAME",
    "ConversionError",
    "Converter",
    "Manifest",
    "PlannedItem",
    "Reason",
    "SyncPlan",
    "SyncResult",
    "Syncer",
    "__version__",
    "convert_bytes",
    "convert_file",
    "get_converter",
    "markdown_path_for",
]
