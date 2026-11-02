from __future__ import annotations

import io
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Iterator

from .base import ConversionError, Converter, guess_mime

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CHUNK_PAGES = 5
DEFAULT_RETRIES = 6
BACKOFF_BASE = 2.0
BACKOFF_MAX = 60.0
INLINE_LIMIT_BYTES = 19 * 1024 * 1024  # Gemini inline request limit is ~20MB


def _backoff_delay(attempt: int) -> float:
    return min(BACKOFF_MAX, BACKOFF_BASE * 2**attempt) + random.uniform(0, 1)


def _is_retryable(exc: Exception) -> bool:
    try:
        from google.genai import errors
    except ImportError:
        return True
    if isinstance(exc, errors.APIError):
        return exc.code == 429 or exc.code >= 500
    return True


class GeminiNoTextError(ConversionError):
    """Gemini answered but did not provide usable text."""

    def __init__(self, message: str, *, recitation: bool = False) -> None:
        super().__init__(message)
        self.recitation = recitation


SYSTEM_PROMPT = """\
You are an expert document transcription engine. Convert the attached document
to clean, faithful GitHub-Flavored Markdown.

Rules:
- Transcribe ALL text in reading order. Do not summarize, omit, or invent content.
- The document may be a poor-quality scan (low resolution, bad contrast, skew).
  Read carefully; when a character is truly illegible write `[illegible]`.
- Preserve structure: headings (#, ##, ...), paragraphs, bullet/numbered lists,
  bold/italic emphasis, footnotes, block quotes.
- Tables: output GFM pipe tables. A table spanning several pages is ONE table:
  merge the parts and emit the header row only once. Keep every row and cell;
  use `<br>` for line breaks inside a cell. Repeat spanning cell values
  instead of leaving them blank.
- Reproduce mathematical notation in LaTeX ($...$ / $$...$$).
- Describe figures/charts briefly in an italic line like *[Figure: ...]*.
- Ignore repeated page headers/footers and page numbers.
- Output ONLY the Markdown. No preamble, no code fence around the whole document.
"""

IMAGE_PROMPT = """\
This image was embedded in a document. Transcribe ALL the text it contains,
faithfully and in reading order, as GitHub-Flavored Markdown (tables as GFM
pipe tables, `[illegible]` for unreadable characters). Do not summarize or
invent content. Start your answer with one italic line of the form
*[Image: <one-sentence description of what the image shows>]*, then the
transcription (nothing more if the image contains no text).
Output ONLY the Markdown. No preamble, no code fence.
"""

CONTINUE_PROMPT = (
    "Your previous answer was cut off. Continue the Markdown transcription exactly "
    "from where you stopped, without repeating anything already emitted and without "
    "any preamble."
)

ALT_USER_PROMPT = (
    "This is the user's own document, being archived as searchable Markdown. "
    "Transcribe its full text faithfully, page by page, following the system rules."
)


def _diagnose(response) -> str:
    fields = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if finish_reason is not None:
            fields.append(f"finish_reason={getattr(finish_reason, 'name', finish_reason)}")
        finish_message = getattr(candidate, "finish_message", None)
        if finish_message is not None:
            fields.append(f"finish_message={finish_message}")
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None:
        block_reason = getattr(feedback, "block_reason", None)
        if block_reason is not None:
            fields.append(f"block_reason={getattr(block_reason, 'name', block_reason)}")
        block_reason_message = getattr(feedback, "block_reason_message", None)
        if block_reason_message is not None:
            fields.append(f"block_reason_message={block_reason_message}")
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        output_tokens = getattr(usage, "candidates_token_count", None)
        if output_tokens is not None:
            fields.append(f"output_tokens={output_tokens}")
        thinking_tokens = getattr(usage, "thoughts_token_count", None)
        if thinking_tokens is not None:
            fields.append(f"thinking_tokens={thinking_tokens}")
    return ", ".join(fields)


class GeminiConverter(Converter):
    """Conversion through the Google Gemini API (default: Gemini 3.8 Flash).

    The whole document is sent in a single request so that Gemini sees
    multi-page tables in context. Documents above the inline size limit are
    uploaded through the Files API and deleted afterwards. If the answer hits
    the output-token limit, the conversation is continued until complete.
    Blocked/empty answers retry with an alternate user prompt, optional fallback
    model, and page chunks for long PDFs.
    """

    name = "gemini"
    extensions = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        fallback_model: str | None = None,
        chunk_pages: int = DEFAULT_CHUNK_PAGES,
        fail_fast_on_recitation: bool = False,
        max_continuations: int = 30,
        retries: int = DEFAULT_RETRIES,
        request_timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self.temperature = temperature
        self.fallback_model = fallback_model
        self.chunk_pages = chunk_pages
        self.fail_fast_on_recitation = fail_fast_on_recitation
        self.max_continuations = max_continuations
        self.retries = retries
        self.request_timeout = request_timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ConversionError(
                    "google-genai is not installed. Install with `pip install 'pdftomd[gemini]'`."
                ) from exc
            if not self.api_key:
                raise ConversionError("No Gemini API key: set GEMINI_API_KEY or pass api_key=.")
            # A real timeout on every request: a stalled connection (e.g. a proxy
            # silently dropping the stream) would otherwise hang the sync forever.
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(timeout=int(self.request_timeout * 1000)),
            )
        return self._client

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        from google.genai import types
        from ..pdfcheck import ensure_pdf_readable

        ensure_pdf_readable(data, filename=filename)
        client = self._get_client()
        mime = guess_mime(filename)
        uploaded = None
        failures: list[tuple[str, GeminiNoTextError]] = []
        try:
            if len(data) <= INLINE_LIMIT_BYTES:
                doc_part = types.Part.from_bytes(data=data, mime_type=mime)
            else:
                uploaded = self._upload(client, data, mime)
                doc_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime)

            models = [self.model]
            if self.fallback_model and self.fallback_model != self.model:
                models.append(self.fallback_model)

            skip_fallback_full = False
            for model in models:
                if skip_fallback_full:
                    break
                for prompt, suffix in (
                    ("Convert this document to Markdown.", "full"),
                    (ALT_USER_PROMPT, "full-alt"),
                ):
                    label = f"{model}/{suffix}"
                    try:
                        text = self._transcribe(client, types, doc_part, model=model, user_prompt=prompt)
                    except GeminiNoTextError as exc:
                        failures.append((label, exc))
                        if self.fail_fast_on_recitation and exc.recitation:
                            raise self._summary_error(filename, failures)
                        if suffix == "full" and exc.recitation:
                            if model == self.model:
                                skip_fallback_full = True
                            break
                    else:
                        return self._finish_attempt(text, filename, label, failures)

            if Path(filename).suffix.lower() == ".pdf" and self.chunk_pages > 0:
                if self._pdf_page_count(data) > self.chunk_pages:
                    skip_fallback_chunks = False
                    for model in models:
                        if skip_fallback_chunks:
                            break
                        label = f"{model}/chunks"
                        try:
                            text = self._transcribe_chunks(client, types, data, model=model)
                        except GeminiNoTextError as exc:
                            failures.append((label, exc))
                            if self.fail_fast_on_recitation and exc.recitation:
                                raise self._summary_error(filename, failures)
                            if model == self.model and exc.recitation:
                                skip_fallback_chunks = True
                        else:
                            return self._finish_attempt(text, filename, label, failures)

            raise self._summary_error(filename, failures)
        finally:
            if uploaded is not None:
                self._delete_upload(client, uploaded)

    def _summary_error(
        self,
        filename: str,
        failures: list[tuple[str, GeminiNoTextError]],
    ) -> GeminiNoTextError:
        details = "; ".join(f"{label}: {diagnostic}" for label, diagnostic in failures)
        return GeminiNoTextError(
            f"{filename}: Gemini returned no usable text after {len(failures)} attempt(s): {details}",
            recitation=any(error.recitation for _, error in failures),
        )

    def _finish_attempt(
        self,
        text: str,
        filename: str,
        label: str,
        failures: list[tuple[str, GeminiNoTextError]],
    ) -> str:
        if failures:
            earlier = "; ".join(f"{attempt}: {diagnostic}" for attempt, diagnostic in failures)
            logging.getLogger(__name__).warning(f"{filename}: converted via {label} after: {earlier}")
        return _strip_outer_fence(text).strip() + "\n"

    def _transcribe(self, client, types, doc_part, *, model: str, user_prompt: str) -> str:
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=self.temperature)
        history = [types.Content(role="user", parts=[doc_part, types.Part.from_text(text=user_prompt)])]
        chunks: list[str] = []
        last = None
        for _ in range(self.max_continuations + 1):
            response = self._generate(client, history, config, model)
            last = response
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                raise GeminiNoTextError(f"prompt blocked ({_diagnose(response)})")
            candidate = candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            finish = getattr(finish_reason, "name", finish_reason)
            text = getattr(response, "text", None) or ""
            chunks.append(text)
            if finish == "MAX_TOKENS":
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
                history.append(types.Content(role="user", parts=[types.Part.from_text(text=CONTINUE_PROMPT)]))
                continue
            if finish not in (None, "", "STOP", "FINISH_REASON_UNSPECIFIED"):
                raise GeminiNoTextError(_diagnose(response), recitation=finish == "RECITATION")
            else:
                break
        else:
            raise ConversionError("Gemini output was still truncated after the maximum number of continuations.")
        text = "".join(chunks)
        if not text.strip():
            raise GeminiNoTextError(f"empty response ({_diagnose(last)})")
        return text

    def _transcribe_chunks(self, client, types, data: bytes, *, model: str) -> str:
        output = []
        for start, end, chunk_data in self._iter_pdf_chunks(data):
            uploaded = None
            try:
                if len(chunk_data) <= INLINE_LIMIT_BYTES:
                    doc_part = types.Part.from_bytes(data=chunk_data, mime_type="application/pdf")
                else:
                    uploaded = self._upload(client, chunk_data, "application/pdf")
                    doc_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type="application/pdf")
                try:
                    text = self._transcribe(
                        client,
                        types,
                        doc_part,
                        model=model,
                        user_prompt="Convert this document to Markdown.",
                    )
                except GeminiNoTextError as normal_error:
                    if normal_error.recitation:
                        raise GeminiNoTextError(
                            f"pages {start}-{end}: {normal_error}",
                            recitation=True,
                        ) from normal_error
                    try:
                        text = self._transcribe(
                            client,
                            types,
                            doc_part,
                            model=model,
                            user_prompt=ALT_USER_PROMPT,
                        )
                    except GeminiNoTextError as alternate_error:
                        raise GeminiNoTextError(
                            f"pages {start}-{end}: normal prompt: {normal_error}; "
                            f"alternate prompt: {alternate_error}",
                            recitation=normal_error.recitation or alternate_error.recitation,
                        ) from alternate_error
                output.append(text.strip())
            finally:
                if uploaded is not None:
                    self._delete_upload(client, uploaded)
        return "\n\n".join(output)

    def _pdf_page_count(self, data: bytes) -> int:
        try:
            import pypdfium2 as pdfium

            document = pdfium.PdfDocument(data)
            try:
                return len(document)
            finally:
                document.close()
        except Exception:
            return 0

    def _iter_pdf_chunks(self, data: bytes) -> Iterator[tuple[int, int, bytes]]:
        import pypdfium2 as pdfium

        source = pdfium.PdfDocument(data)
        try:
            for start in range(0, len(source), self.chunk_pages):
                end = min(start + self.chunk_pages, len(source))
                chunk = pdfium.PdfDocument.new()
                try:
                    chunk.import_pages(source, list(range(start, end)))
                    buffer = io.BytesIO()
                    chunk.save(buffer)
                    yield start + 1, end, buffer.getvalue()
                finally:
                    chunk.close()
        finally:
            source.close()

    @staticmethod
    def _delete_upload(client, uploaded) -> None:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    def describe_image(self, data: bytes, mime: str) -> str:
        """Transcribe/describe a single picture (used for images embedded in Google Docs exports)."""
        from google.genai import types

        client = self._get_client()
        contents = [types.Content(role="user", parts=[types.Part.from_bytes(data=data, mime_type=mime), types.Part.from_text(text=IMAGE_PROMPT)])]
        response = self._generate(
            client,
            contents,
            types.GenerateContentConfig(temperature=self.temperature),
            self.model,
        )
        return _strip_outer_fence(getattr(response, "text", None) or "").strip()

    def _upload(self, client, data: bytes, mime: str):
        last: Exception | None = None
        uploaded = None
        for attempt in range(self.retries):
            try:
                uploaded = client.files.upload(file=io.BytesIO(data), config={"mime_type": mime})
                break
            except Exception as exc:  # transient upload errors / stalls
                if not _is_retryable(exc):
                    raise ConversionError(f"Gemini file upload failed: {exc}") from exc
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(_backoff_delay(attempt))
        if uploaded is None:
            raise ConversionError(f"Gemini file upload failed after {self.retries} attempts: {last}") from last
        deadline = time.monotonic() + self.request_timeout
        while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING":
            if time.monotonic() > deadline:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass
                raise ConversionError(f"Gemini file {uploaded.name} stayed in PROCESSING for over {self.request_timeout:g}s.")
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if getattr(uploaded.state, "name", uploaded.state) != "ACTIVE":
            raise ConversionError(f"Gemini file upload ended in state {uploaded.state!r}.")
        return uploaded

    def _generate(self, client, contents, config, model: str):
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return client.models.generate_content(model=model, contents=contents, config=config)
            except Exception as exc:  # transient API errors (429/5xx)
                if not _is_retryable(exc):
                    raise ConversionError(f"Gemini request failed: {exc}") from exc
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(_backoff_delay(attempt))
        raise ConversionError(f"Gemini request failed after {self.retries} attempts: {last}") from last


_FENCE_RE = re.compile(r"\A\s*```(?:markdown|md)?\s*\n(.*)\n```\s*\Z", re.DOTALL)


def _strip_outer_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text
