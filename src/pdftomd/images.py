"""Replace base64-embedded images in Markdown by a textual description.

Google Docs exported as ``text/markdown`` inline every picture as a data URI
(``![][image1]`` + ``[image1]: <data:image/png;base64,...>``). Such blobs are
useless to a text index and can weigh hundreds of kilobytes each, so they are
sent to a vision model and swapped for the text it reads/describes.
"""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Protocol

_DATA_URI = r"data:(image/[\w.+-]+);base64,([A-Za-z0-9+/=\s]+)"
# Reference definition on its own line: `[image1]: <data:...>` (angle brackets optional).
_DEFINITION_RE = re.compile(rf"^[ \t]*\[([^\]]+)\]:[ \t]*<?{_DATA_URI}>?[ \t]*$", re.MULTILINE)
# Inline image: `![alt](data:...)`.
_INLINE_RE = re.compile(rf"[ \t]*!\[([^\]]*)\]\(<?{_DATA_URI}>?\)[ \t]*")
_DATA_IMAGE_RE = re.compile(r"data:image/[\w.+-]+;base64,")


class ImageDescriber(Protocol):
    def describe_image(self, data: bytes, mime: str) -> str:
        """Return Markdown text standing in for the image (transcription and/or description)."""
        ...


def has_data_images(markdown: str) -> bool:
    return _DATA_IMAGE_RE.search(markdown) is not None


def describe_images(markdown: str, describer: ImageDescriber) -> str:
    """Return ``markdown`` with every base64 image replaced by ``describer``'s text.

    Identical images are described once. Descriptions are inserted as their
    own paragraph where the image was referenced.
    """
    if not has_data_images(markdown):
        return markdown
    cache: dict[str, str] = {}

    def text_for(mime: str, payload: str) -> str:
        key = hashlib.sha256(payload.encode("ascii", "ignore")).hexdigest()
        if key not in cache:
            try:
                data = base64.b64decode(re.sub(r"\s+", "", payload), validate=True)
            except (ValueError, TypeError):
                cache[key] = "*[Image: unreadable]*"
            else:
                cache[key] = describer.describe_image(data, mime).strip() or "*[Image]*"
        return cache[key]

    definitions: dict[str, str] = {}

    def take_definition(m: re.Match) -> str:
        definitions[m.group(1)] = text_for(m.group(2), m.group(3))
        return ""

    markdown = _DEFINITION_RE.sub(take_definition, markdown)
    if definitions:
        labels = "|".join(re.escape(label) for label in definitions)
        markdown = re.sub(rf"[ \t]*!\[[^\]]*\]\[({labels})\][ \t]*", lambda m: f"\n\n{definitions[m.group(1)]}\n\n", markdown)
    markdown = _INLINE_RE.sub(lambda m: f"\n\n{text_for(m.group(2), m.group(3))}\n\n", markdown)
    return re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
