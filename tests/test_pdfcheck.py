from __future__ import annotations

import io

import pytest

from pdftomd.converters.base import ConversionError
from pdftomd.pdfcheck import EncryptedPdfError, ensure_pdf_readable


def _plain_pdf() -> bytes:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(200, 200)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _encrypted_pdf(user_password: str, owner_password: str) -> bytes:
    from pypdf import PdfWriter

    w = PdfWriter()
    w.add_blank_page(200, 200)
    w.encrypt(user_password=user_password, owner_password=owner_password)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_unencrypted_pdf_passes():
    ensure_pdf_readable(_plain_pdf(), filename="doc.pdf")


def test_user_password_pdf_is_flagged():
    data = _encrypted_pdf("secret", "owner")
    with pytest.raises(EncryptedPdfError, match="password-protected"):
        ensure_pdf_readable(data, filename="locked.pdf")
    assert issubclass(EncryptedPdfError, ConversionError)


def test_owner_password_only_pdf_passes():
    ensure_pdf_readable(_encrypted_pdf("", "owner"), filename="locked.pdf")


def test_non_pdf_filename_is_skipped():
    ensure_pdf_readable(b"not a pdf at all", filename="doc.txt")


def test_garbage_pdf_is_left_to_the_engine():
    # Only encryption is diagnosed here; other parse problems fall through to the engine.
    ensure_pdf_readable(b"this is definitely not a pdf", filename="bad.pdf")


def test_gemini_checks_before_touching_the_network(monkeypatch):
    from pdftomd.converters.gemini import GeminiConverter

    conv = GeminiConverter(api_key="dummy")
    monkeypatch.setattr(conv, "_get_client", lambda: (_ for _ in ()).throw(AssertionError("network touched")))
    with pytest.raises(EncryptedPdfError, match="password-protected"):
        conv.convert_bytes(_encrypted_pdf("secret", "owner"), filename="a.pdf")
