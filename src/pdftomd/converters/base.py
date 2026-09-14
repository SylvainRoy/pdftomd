from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path


class ConversionError(Exception):
    """Raised when a backend fails to produce Markdown for a document."""


class Converter(ABC):
    """A backend able to turn a document (bytes) into a Markdown string.

    Implementations must be pure with respect to the destination: they never
    write anything except transient temp files they own.
    """

    #: Short identifier used in the CLI and stored in the sync manifest.
    name: str = "base"
    #: Lower-case file extensions (with leading dot) this backend accepts.
    extensions: frozenset[str] = frozenset()

    def supports(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.extensions

    @abstractmethod
    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        """Convert in-memory document bytes to Markdown.

        ``filename`` is only used to infer the document type; nothing is read
        from disk.
        """

    def convert_file(self, path: str | Path) -> str:
        """Convert a document on disk to a Markdown string (nothing written)."""
        path = Path(path)
        return self.convert_bytes(path.read_bytes(), filename=path.name)


def guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"
