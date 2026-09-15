from __future__ import annotations

import io
import os
import re
import time

from .base import ConversionError, Converter, guess_mime

DEFAULT_MODEL = "gemini-3.6-flash"
INLINE_LIMIT_BYTES = 19 * 1024 * 1024  # Gemini inline request limit is ~20MB

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

CONTINUE_PROMPT = (
    "Your previous answer was cut off. Continue the Markdown transcription exactly "
    "from where you stopped, without repeating anything already emitted and without "
    "any preamble."
)


class GeminiConverter(Converter):
    """Conversion through the Google Gemini API (default: Gemini 3.6 Flash).

    The whole document is sent in a single request so that Gemini sees
    multi-page tables in context. Documents above the inline size limit are
    uploaded through the Files API and deleted afterwards. If the answer hits
    the output-token limit, the conversation is continued until complete.
    """

    name = "gemini"
    extensions = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_continuations: int = 30,
        retries: int = 3,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model = model
        self.max_continuations = max_continuations
        self.retries = retries
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - depends on env
                raise ConversionError(
                    "google-genai is not installed. Install with `pip install 'pdftomd[gemini]'`."
                ) from exc
            if not self.api_key:
                raise ConversionError("No Gemini API key: set GEMINI_API_KEY or pass api_key=.")
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def convert_bytes(self, data: bytes, *, filename: str) -> str:
        from google.genai import types

        client = self._get_client()
        mime = guess_mime(filename)
        uploaded = None
        try:
            if len(data) <= INLINE_LIMIT_BYTES:
                doc_part = types.Part.from_bytes(data=data, mime_type=mime)
            else:
                uploaded = client.files.upload(file=io.BytesIO(data), config={"mime_type": mime})
                while getattr(uploaded.state, "name", uploaded.state) == "PROCESSING":
                    time.sleep(2)
                    uploaded = client.files.get(name=uploaded.name)
                doc_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime)
            return self._transcribe(client, types, doc_part)
        finally:
            if uploaded is not None:
                try:
                    client.files.delete(name=uploaded.name)
                except Exception:
                    pass

    def _transcribe(self, client, types, doc_part) -> str:
        config = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.0)
        history = [types.Content(role="user", parts=[doc_part, types.Part.from_text(text="Convert this document to Markdown.")])]
        chunks: list[str] = []
        for _ in range(self.max_continuations + 1):
            response = self._generate(client, history, config)
            text = response.text or ""
            chunks.append(text)
            finish = None
            if response.candidates:
                finish = getattr(response.candidates[0].finish_reason, "name", str(response.candidates[0].finish_reason))
            if finish != "MAX_TOKENS":
                break
            history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
            history.append(types.Content(role="user", parts=[types.Part.from_text(text=CONTINUE_PROMPT)]))
        else:
            raise ConversionError("Gemini output was still truncated after the maximum number of continuations.")
        return _strip_outer_fence("".join(chunks)).strip() + "\n"

    def _generate(self, client, contents, config):
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                return client.models.generate_content(model=self.model, contents=contents, config=config)
            except Exception as exc:  # transient API errors (429/5xx)
                last = exc
                time.sleep(2 ** attempt)
        raise ConversionError(f"Gemini request failed after {self.retries} attempts: {last}") from last


_FENCE_RE = re.compile(r"\A\s*```(?:markdown|md)?\s*\n(.*)\n```\s*\Z", re.DOTALL)


def _strip_outer_fence(text: str) -> str:
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text
