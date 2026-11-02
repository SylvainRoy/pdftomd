from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from .base import ConversionError, Converter
from .gemini import GeminiNoTextError


class FallbackConverter(Converter):
    def __init__(self, primary: Converter, fallback: Converter) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.extensions = primary.extensions
        self.parallel_safe = primary.parallel_safe
        self._fallback_lock = threading.Lock()

    def _run_fallback(self, primary_error: GeminiNoTextError, run: Callable[[], str]) -> str:
        logging.getLogger(__name__).warning(
            f"{self.name}: {primary_error}; converting with {self.fallback.name}"
        )
        try:
            if self.fallback.parallel_safe:
                return run()
            with self._fallback_lock:
                return run()
        except ConversionError as fallback_error:
            raise ConversionError(
                f"{primary_error}; fallback {self.fallback.name} failed: {fallback_error}"
            ) from fallback_error

    def convert_file(self, path: str | Path) -> str:
        try:
            return self.primary.convert_file(path)
        except GeminiNoTextError as exc:
            if not self.fallback.supports(path):
                raise
            return self._run_fallback(exc, lambda: self.fallback.convert_file(path))

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        try:
            return self.primary.convert_bytes(data, filename=filename)
        except GeminiNoTextError as exc:
            if not self.fallback.supports(filename):
                raise
            return self._run_fallback(exc, lambda: self.fallback.convert_bytes(data, filename=filename))

    def describe_image(self, data: bytes, mime: str) -> str:
        return self.primary.describe_image(data, mime)
