from __future__ import annotations

from typing import Any

from .base import ConversionError, Converter

ENGINES = ("marker", "gemini")


def get_converter(engine: str, **options: Any) -> Converter:
    """Instantiate a converter by engine name (``"marker"`` or ``"gemini"``)."""
    if engine == "marker":
        from .marker import MarkerConverter

        return MarkerConverter(**options)
    if engine == "gemini":
        from .gemini import GeminiConverter

        return GeminiConverter(**options)
    raise ValueError(f"Unknown engine {engine!r}; expected one of {ENGINES}")


__all__ = ["ConversionError", "Converter", "ENGINES", "get_converter"]
