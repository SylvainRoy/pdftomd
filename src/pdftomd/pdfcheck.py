"""Cheap pre-flight checks on PDF bytes, so engines get a clear error instead of an opaque API failure."""

from __future__ import annotations

from pathlib import Path

from .converters.base import ConversionError


class EncryptedPdfError(ConversionError):
    """The PDF requires a password to open."""


def ensure_pdf_readable(data: bytes, *, filename: str) -> None:
    """Raise EncryptedPdfError if ``data`` is a password-protected PDF. No-op for non-PDF files
    or when pypdfium2 is unavailable."""
    if Path(filename).suffix.lower() != ".pdf":
        return
    try:
        import pypdfium2 as pdfium
    except ImportError:
        return
    try:
        doc = pdfium.PdfDocument(data)
        close = getattr(doc, "close", None)
        if close is not None:
            close()
    except pdfium.PdfiumError as exc:
        if "password" in str(exc).lower():
            raise EncryptedPdfError(
                f"{filename}: PDF is password-protected and cannot be converted (no password available)"
            ) from exc
        # Other parse problems: let the engine try, it may cope better than pdfium.
