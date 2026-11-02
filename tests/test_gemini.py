from __future__ import annotations

import io
from types import SimpleNamespace

import pypdfium2 as pdfium
import pytest
from google.genai import errors, types

from pdftomd.converters.base import ConversionError
from pdftomd.converters import gemini as gemini_module
from pdftomd.converters.gemini import (
    ALT_USER_PROMPT,
    DEFAULT_MODEL,
    GeminiConverter,
    GeminiNoTextError,
)


def _response(
    text: str | None = None,
    *,
    finish_reason: str | None = "STOP",
    candidates: bool = True,
    block_reason: str | None = None,
):
    response_candidates = (
        [SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason), finish_message=None)]
        if candidates
        else []
    )
    feedback = None
    if block_reason is not None:
        feedback = SimpleNamespace(block_reason=SimpleNamespace(name=block_reason), block_reason_message=None)
    return SimpleNamespace(
        text=text,
        candidates=response_candidates,
        prompt_feedback=feedback,
        usage_metadata=SimpleNamespace(candidates_token_count=0, thoughts_token_count=0),
    )


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        result = self.script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def _converter(monkeypatch, script, *, sleep_delays=None, **kwargs):
    monkeypatch.setattr(
        "pdftomd.converters.gemini.time.sleep",
        lambda delay: sleep_delays.append(delay) if sleep_delays is not None else None,
    )
    options = {"retries": 1, **kwargs}
    conv = GeminiConverter(**options)
    conv._client = FakeClient(script)
    return conv


def _pdf_bytes(page_count: int) -> bytes:
    document = pdfium.PdfDocument.new()
    try:
        for _ in range(page_count):
            page = document.new_page(200, 200)
            page.close()
        output = io.BytesIO()
        document.save(output)
        return output.getvalue()
    finally:
        document.close()


def _user_text(call) -> str:
    return call[1][0].parts[-1].text


def test_stop_response_uses_default_model_and_temperature(monkeypatch):
    conv = _converter(monkeypatch, [_response("transcribed text")])

    assert conv.convert_bytes(b"not-a-pdf", filename="scan.png") == "transcribed text\n"
    assert len(conv._client.models.calls) == 1
    model, _, config = conv._client.models.calls[0]
    assert model == "gemini-3.8-flash" == DEFAULT_MODEL
    assert config.temperature == 1.0


def test_recitation_retries_with_alternate_prompt(monkeypatch):
    conv = _converter(
        monkeypatch,
        [_response("partial", finish_reason="RECITATION"), _response("complete text")],
    )

    assert conv.convert_bytes(b"image", filename="scan.png") == "complete text\n"
    assert len(conv._client.models.calls) == 2
    assert _user_text(conv._client.models.calls[1]) == ALT_USER_PROMPT


def test_fallback_model_runs_after_both_primary_prompts_fail(monkeypatch):
    conv = _converter(
        monkeypatch,
        [
            _response("blocked", finish_reason="RECITATION"),
            _response("blocked", finish_reason="RECITATION"),
            _response("fallback success"),
        ],
        fallback_model="gemini-4-argon",
    )

    assert conv.convert_bytes(b"image", filename="scan.png") == "fallback success\n"
    assert [call[0] for call in conv._client.models.calls] == [
        "gemini-3.8-flash",
        "gemini-3.8-flash",
        "gemini-4-argon",
    ]


def test_long_pdf_uses_ordered_page_chunks(monkeypatch):
    conv = _converter(
        monkeypatch,
        [
            _response("partial", finish_reason="RECITATION"),
            _response("partial", finish_reason="RECITATION"),
            _response("first chunk"),
            _response("second chunk"),
            _response("third chunk"),
        ],
    )

    result = conv.convert_bytes(_pdf_bytes(12), filename="long.pdf")

    assert result == "first chunk\n\nsecond chunk\n\nthird chunk\n"
    calls = conv._client.models.calls
    assert len(calls) == 5
    page_counts = []
    for _, contents, _ in calls[2:]:
        part = contents[0].parts[0]
        assert part.inline_data.mime_type == "application/pdf"
        chunk = pdfium.PdfDocument(part.inline_data.data)
        try:
            page_counts.append(len(chunk))
        finally:
            chunk.close()
    assert page_counts == [5, 5, 2]


def test_unparseable_pdf_skips_chunk_attempt(monkeypatch):
    blocked = _response("partial", finish_reason="RECITATION")
    conv = _converter(monkeypatch, [blocked, blocked])

    with pytest.raises(GeminiNoTextError, match="Gemini returned no usable text after 2 attempt\\(s\\)"):
        conv.convert_bytes(b"not a pdf", filename="bad.pdf")

    assert len(conv._client.models.calls) == 2


def test_all_attempts_fail_with_labels_and_diagnostics(monkeypatch):
    blocked = _response("partial", finish_reason="RECITATION")
    conv = _converter(
        monkeypatch,
        [blocked, blocked, blocked, blocked],
        fallback_model="gemini-4-argon",
        chunk_pages=0,
    )

    with pytest.raises(GeminiNoTextError) as exc_info:
        conv.convert_bytes(b"image", filename="scan.png")

    message = str(exc_info.value)
    assert "scan.png: Gemini returned no usable text after 4 attempt(s)" in message
    for label in (
        "gemini-3.8-flash/full",
        "gemini-3.8-flash/full-alt",
        "gemini-4-argon/full",
        "gemini-4-argon/full-alt",
    ):
        assert label in message
    assert message.count("RECITATION") == 4


def test_no_candidates_reports_blocked_prompt(monkeypatch):
    conv = _converter(
        monkeypatch,
        [_response(candidates=False, block_reason="PROHIBITED_CONTENT"), _response(candidates=False)],
        chunk_pages=0,
    )

    with pytest.raises(GeminiNoTextError) as exc_info:
        conv.convert_bytes(b"image", filename="scan.png")

    assert "prompt blocked" in str(exc_info.value)
    assert "PROHIBITED_CONTENT" in str(exc_info.value)


def test_empty_stop_response_advances_to_alternate_prompt(monkeypatch):
    conv = _converter(monkeypatch, [_response(" \n"), _response("now transcribed")])

    assert conv.convert_bytes(b"image", filename="scan.png") == "now transcribed\n"
    assert len(conv._client.models.calls) == 2


def test_api_failure_propagates_without_attempt_ladder(monkeypatch):
    conv = _converter(monkeypatch, [RuntimeError("service unavailable")], retries=1)

    with pytest.raises(ConversionError, match="Gemini request failed after 1 attempts"):
        conv.convert_bytes(b"image", filename="scan.png")

    assert len(conv._client.models.calls) == 1


def test_max_tokens_continuation_concatenates_text(monkeypatch):
    conv = _converter(
        monkeypatch,
        [_response("first ", finish_reason="MAX_TOKENS"), _response("second")],
    )

    assert conv.convert_bytes(b"image", filename="scan.png") == "first second\n"
    assert len(conv._client.models.calls) == 2


def test_non_pdf_does_not_use_chunk_attempt(monkeypatch):
    conv = _converter(
        monkeypatch,
        [_response("partial", finish_reason="RECITATION"), _response("partial", finish_reason="RECITATION")],
    )

    with pytest.raises(GeminiNoTextError):
        conv.convert_bytes(b"image", filename="scan.png")

    assert len(conv._client.models.calls) == 2


def test_chunk_alternate_failure_reports_page_range(monkeypatch):
    blocked = _response("partial", finish_reason="RECITATION")
    conv = _converter(
        monkeypatch,
        [blocked, blocked, blocked, blocked],
        chunk_pages=5,
    )

    with pytest.raises(GeminiNoTextError) as exc_info:
        conv.convert_bytes(_pdf_bytes(6), filename="long.pdf")

    assert "pages 1-5" in str(exc_info.value)
    assert "RECITATION" in str(exc_info.value)
    assert len(conv._client.models.calls) == 4


def test_describe_image_uses_configured_temperature(monkeypatch):
    conv = _converter(monkeypatch, [_response("description")])

    assert conv.describe_image(b"image", "image/png") == "description"
    assert conv._client.models.calls[0][2].temperature == 1.0


def test_generate_retries_503_with_backoff(monkeypatch):
    unavailable = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}},
    )
    delays = []
    conv = _converter(
        monkeypatch,
        [unavailable, unavailable, _response("recovered")],
        retries=6,
        sleep_delays=delays,
    )

    assert conv.convert_bytes(b"image", filename="scan.png") == "recovered\n"
    assert len(conv._client.models.calls) == 3
    assert len(delays) == 2


def test_generate_retries_429(monkeypatch):
    rate_limited = errors.ClientError(
        429,
        {"error": {"code": 429, "message": "rate limited", "status": "RESOURCE_EXHAUSTED"}},
    )
    delays = []
    conv = _converter(monkeypatch, [rate_limited, _response("recovered")], retries=6, sleep_delays=delays)

    assert conv.convert_bytes(b"image", filename="scan.png") == "recovered\n"
    assert len(conv._client.models.calls) == 2
    assert len(delays) == 1


def test_generate_fails_fast_on_client_error(monkeypatch):
    invalid_request = errors.ClientError(
        400,
        {"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}},
    )
    delays = []
    conv = _converter(monkeypatch, [invalid_request], retries=6, sleep_delays=delays)

    with pytest.raises(ConversionError, match="Gemini request failed:"):
        conv.convert_bytes(b"image", filename="scan.png")

    assert len(conv._client.models.calls) == 1
    assert delays == []


def test_generate_exhausts_retries_without_sleeping_after_last_attempt(monkeypatch):
    unavailable = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}},
    )
    delays = []
    conv = _converter(monkeypatch, [unavailable] * 6, retries=6, sleep_delays=delays)

    with pytest.raises(ConversionError, match="Gemini request failed after 6 attempts"):
        conv.convert_bytes(b"image", filename="scan.png")

    assert len(conv._client.models.calls) == 6
    assert len(delays) == 5


def test_backoff_delay_doubles_and_caps(monkeypatch):
    monkeypatch.setattr(gemini_module.random, "uniform", lambda *_: 0)

    assert [gemini_module._backoff_delay(attempt) for attempt in range(7)] == [2, 4, 8, 16, 32, 60, 60]


def test_upload_retries_503_with_backoff(monkeypatch):
    unavailable = errors.ServerError(
        503,
        {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}},
    )
    delays = []
    conv = _converter(monkeypatch, [], retries=6, sleep_delays=delays)

    class FakeFiles:
        def __init__(self):
            self.results = [unavailable, SimpleNamespace(
                name="files/1",
                uri="https://example.test/files/1",
                state=SimpleNamespace(name="ACTIVE"),
            )]
            self.calls = 0

        def upload(self, *, file, config):
            self.calls += 1
            result = self.results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    conv._client.files = FakeFiles()

    uploaded = conv._upload(conv._client, b"document", "application/pdf")

    assert uploaded.name == "files/1"
    assert conv._client.files.calls == 2
    assert len(delays) == 1
