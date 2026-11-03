from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pdftomd import MANIFEST_NAME, Syncer
from pdftomd.converters.base import ConversionError, Converter
from pdftomd.converters.fallback import FallbackConverter
from pdftomd.converters.gemini import GeminiNoTextError, GeminiUnavailableError


class StubConverter(Converter):
    def __init__(
        self,
        name: str,
        *,
        output: str = "# converted document\n",
        error: ConversionError | None = None,
        extensions: frozenset[str] = frozenset({".pdf"}),
        parallel_safe: bool = True,
    ) -> None:
        self.name = name
        self.extensions = extensions
        self.parallel_safe = parallel_safe
        self.output = output
        self.error = error
        self.calls: list[str] = []
        self.description: str | None = None

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        self.calls.append(filename)
        if self.error:
            raise self.error
        return self.output

    def describe_image(self, data: bytes, mime: str) -> str:
        return self.description or f"{mime}:{data.decode()}"


def test_fallback_returns_output_and_preserves_primary_name():
    primary = StubConverter("gemini", error=GeminiNoTextError("no text", recitation=True))
    fallback = StubConverter("marker", output="# marker output\n")
    converter = FallbackConverter(primary, fallback)

    assert converter.convert_bytes(b"pdf", filename="document.pdf") == "# marker output\n"
    assert converter.name == "gemini"
    assert converter.extensions == primary.extensions
    assert converter.parallel_safe is primary.parallel_safe


def test_unavailable_error_triggers_marker_fallback():
    primary = StubConverter("gemini", error=GeminiUnavailableError("server deadline", deadline=True))
    fallback = StubConverter("marker", output="# marker output\n")
    converter = FallbackConverter(primary, fallback)

    assert converter.convert_bytes(b"pdf", filename="document.pdf") == "# marker output\n"
    assert fallback.calls == ["document.pdf"]


def test_primary_conversion_error_does_not_trigger_fallback():
    primary_error = ConversionError("503")
    primary = StubConverter("gemini", error=primary_error)
    fallback = StubConverter("marker")
    converter = FallbackConverter(primary, fallback)

    with pytest.raises(ConversionError) as exc_info:
        converter.convert_bytes(b"pdf", filename="document.pdf")

    assert exc_info.value is primary_error
    assert fallback.calls == []


def test_unsupported_extension_reraises_primary_error():
    primary_error = GeminiNoTextError("no text", recitation=True)
    primary = StubConverter("gemini", error=primary_error)
    fallback = StubConverter("marker", extensions=frozenset({".png"}))
    converter = FallbackConverter(primary, fallback)

    with pytest.raises(GeminiNoTextError) as exc_info:
        converter.convert_bytes(b"doc", filename="document.docx")

    assert exc_info.value is primary_error
    assert fallback.calls == []


def test_fallback_conversion_error_combines_both_failures():
    primary = StubConverter("gemini", error=GeminiNoTextError("no text"))
    fallback = StubConverter("marker", error=ConversionError("marker failed"))
    converter = FallbackConverter(primary, fallback)

    with pytest.raises(ConversionError) as exc_info:
        converter.convert_bytes(b"pdf", filename="document.pdf")

    assert "no text" in str(exc_info.value)
    assert "fallback marker failed: marker failed" in str(exc_info.value)


def test_non_parallel_safe_fallback_calls_are_serialized():
    class SlowFallback(StubConverter):
        def __init__(self):
            super().__init__("marker", parallel_safe=False)
            self.active = 0
            self.max_active = 0
            self.activity_lock = threading.Lock()

        def convert_bytes(self, data: bytes, *, filename: str) -> str:
            with self.activity_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.02)
                return super().convert_bytes(data, filename=filename)
            finally:
                with self.activity_lock:
                    self.active -= 1

    primary = StubConverter("gemini", error=GeminiNoTextError("no text"))
    fallback = SlowFallback()
    converter = FallbackConverter(primary, fallback)
    barrier = threading.Barrier(3)
    outputs = []

    def convert(filename: str) -> str:
        barrier.wait()
        outputs.append(converter.convert_bytes(b"pdf", filename=filename))
        return outputs[-1]

    threads = [
        threading.Thread(target=convert, args=(f"doc{index}.pdf",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert fallback.max_active == 1
    assert len(fallback.calls) == 2
    assert outputs == ["# converted document\n"] * 2


def test_sync_manifest_keeps_primary_engine_and_second_plan_is_current(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "document.pdf").write_bytes(b"pdf")
    destination = tmp_path / "destination"
    primary = StubConverter("gemini", error=GeminiNoTextError("no text"))
    fallback = StubConverter("marker")
    syncer = Syncer(source, destination, FallbackConverter(primary, fallback))

    result = syncer.sync()

    assert result.generated == ["document.pdf"]
    manifest = json.loads((destination / MANIFEST_NAME).read_text())
    assert manifest["files"]["document.pdf"]["engine"] == "gemini"
    assert syncer.plan().to_generate == []


def test_describe_image_delegates_to_primary():
    primary = StubConverter("gemini")
    primary.description = "primary description"
    converter = FallbackConverter(primary, StubConverter("marker"))

    assert converter.describe_image(b"image", "image/png") == "primary description"
