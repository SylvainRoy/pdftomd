from __future__ import annotations

import base64

import pytest

from pdftomd.images import describe_images, has_data_images

PIXEL = base64.b64encode(b"\x89PNG fake").decode()
OTHER = base64.b64encode(b"GIF89a fake").decode()


class Describer:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    def describe_image(self, data: bytes, mime: str) -> str:
        self.calls.append((data, mime))
        return f"*[Image: {mime} #{len(self.calls)}]*"


def test_untouched_without_images():
    md = "# Title\n\n![logo](https://example.com/logo.png)\n"
    assert not has_data_images(md)
    assert describe_images(md, Describer()) is md


def test_reference_definitions_replaced_and_deduplicated():
    md = (
        "# Doc\n\nIntro ![][image1] inline.\n\n![][image2]\n\n![][image1]\n\nEnd\n\n"
        f"[image1]: <data:image/png;base64,{PIXEL}>\n[image2]: data:image/gif;base64,{OTHER}\n"
    )
    d = Describer()
    out = describe_images(md, d)
    assert out == "# Doc\n\nIntro\n\n*[Image: image/png #1]*\n\ninline.\n\n*[Image: image/gif #2]*\n\n*[Image: image/png #1]*\n\nEnd\n"
    assert d.calls == [(b"\x89PNG fake", "image/png"), (b"GIF89a fake", "image/gif")]
    assert not has_data_images(out)


def test_inline_images_and_wrapped_base64():
    wrapped = PIXEL[:4] + "\n" + PIXEL[4:]
    md = f"Before ![alt](data:image/png;base64,{wrapped}) after\n"
    d = Describer()
    assert describe_images(md, d) == "Before\n\n*[Image: image/png #1]*\n\nafter\n"
    assert d.calls[0][0] == b"\x89PNG fake"


def test_invalid_base64_is_marked_unreadable():
    md = "![](data:image/png;base64,QUJDRA)\n"
    d = Describer()
    assert describe_images(md, d) == "*[Image: unreadable]*\n"
    assert d.calls == []


def test_describer_errors_propagate():
    class Boom:
        def describe_image(self, data, mime):
            raise RuntimeError("vision down")

    with pytest.raises(RuntimeError, match="vision down"):
        describe_images(f"![](data:image/png;base64,{PIXEL})\n", Boom())
