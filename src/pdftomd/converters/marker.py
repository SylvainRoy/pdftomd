from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from .base import ConversionError, Converter


class MarkerConverter(Converter):
    """Local conversion with `marker-pdf` (https://github.com/datalab-to/marker).

    Models are loaded lazily on first use and cached on the instance, so keep a
    single converter alive when processing many documents.
    """

    name = "marker"
    extensions = frozenset(
        {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
         ".docx", ".pptx", ".xlsx", ".html", ".epub"}
    )

    def __init__(
        self,
        *,
        force_ocr: bool = False,
        use_llm: bool = False,
        languages: list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> None:
        self.config: dict[str, Any] = {"output_format": "markdown"}
        if force_ocr:
            # Re-OCR every page: best choice for poor scans with bad embedded text.
            self.config["force_ocr"] = True
        if use_llm:
            # Hybrid mode: LLM post-processing improves complex/multi-page tables.
            self.config["use_llm"] = True
        if languages:
            self.config["languages"] = ",".join(languages)
        if extra_config:
            self.config.update(extra_config)
        self._converter = None

    def _get_converter(self):
        if self._converter is None:
            try:
                from marker.config.parser import ConfigParser
                from marker.converters.pdf import PdfConverter
                from marker.models import create_model_dict
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ConversionError(
                    "marker-pdf is not installed. Install with `pip install 'pdftomd[marker]'`."
                ) from exc
            parser = ConfigParser(self.config)
            self._converter = PdfConverter(
                config=parser.generate_config_dict(),
                artifact_dict=create_model_dict(),
                processor_list=parser.get_processors(),
                renderer=parser.get_renderer(),
                llm_service=parser.get_llm_service(),
            )
        return self._converter

    def convert_file(self, path: str | Path) -> str:
        from marker.output import text_from_rendered

        path = Path(path)
        try:
            rendered = self._get_converter()(str(path))
            text, _, _images = text_from_rendered(rendered)
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(f"marker failed on {path.name}: {exc}") from exc
        return text

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        # marker only reads from a path; use a private temp file that is
        # removed immediately after conversion. Nothing else touches disk.
        suffix = Path(filename).suffix
        with tempfile.TemporaryDirectory(prefix="pdftomd-") as tmp:
            tmp_path = Path(tmp) / f"document{suffix}"
            tmp_path.write_bytes(data)
            return self.convert_file(tmp_path)
