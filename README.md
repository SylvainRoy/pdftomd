# pdftomd

Convert a directory tree of documents (PDFs, scans, images, ...) into Markdown
files, incrementally. The source tree is **never modified**; the destination
mirrors its hierarchy with one `.md` per document.

Two conversion engines are available, selectable with `--engine`:

| engine   | how it works                                                                | good for                                               |
|----------|-----------------------------------------------------------------------------|--------------------------------------------------------|
| `marker` | local, [marker-pdf](https://github.com/datalab-to/marker) (layout + OCR)    | offline, private documents; `--force-ocr` for bad scans |
| `gemini` | Google Gemini API (default model `gemini-2.5-flash`, configurable)          | poor scans, multi-page tables seen in full context      |

## Install

```bash
uv sync --group dev                 # core + tests
uv sync --extra marker              # + marker-pdf (downloads models on first run)
uv sync --extra gemini              # + google-genai (needs GEMINI_API_KEY)
uv sync --extra all
```

Requires Python 3.10-3.13 (marker's PyTorch dependency does not support 3.14 yet).

## CLI

```bash
# What would be (re)generated? Nothing is written.
pdftomd list  ./docs ./docs-md
pdftomd sync  ./docs ./docs-md --dry-run

# Incremental sync: only new / changed / missing outputs are regenerated.
pdftomd sync ./docs ./docs-md --engine marker --force-ocr
pdftomd sync ./docs ./docs-md --engine gemini --gemini-model gemini-2.5-pro

# Force specific files (relative to the source dir), even if up to date.
pdftomd sync ./docs ./docs-md --select reports/2024/q3.pdf --select scans/invoice.png

# Regenerate everything; remove .md files whose source disappeared.
pdftomd sync ./docs ./docs-md --force --prune

# Single document to stdout / file (string-level path, nothing else written).
pdftomd convert scan.pdf --engine gemini > scan.md

# Poll the source folder and keep the destination in sync.
pdftomd watch ./docs ./docs-md --interval 30
```

### Change detection

State is kept in `<dest>/.pdftomd-manifest.json` (SHA-256, size, mtime and
engine per source file). A document is regenerated when it is new, its content
hash changed, the engine changed, its `.md` is missing, or it was selected /
forced. A touched-but-identical file is *not* regenerated. Destination files
whose source vanished are reported as orphans and only deleted with `--prune`.

## Library

```python
from pdftomd import convert_bytes, convert_file, get_converter, Syncer

# String level: bytes in, Markdown out, nothing written to disk.
md = convert_bytes(pdf_bytes, filename="report.pdf", engine="gemini")

# File level: read a document, return Markdown (still nothing written).
md = convert_file("report.pdf", engine="marker", force_ocr=True)

# Directory level.
syncer = Syncer("docs", "docs-md", get_converter("marker", force_ocr=True))
plan = syncer.plan()                          # dry run, pure
for item in plan.to_generate:
    print(item.reason.value, item.rel_path)
result = syncer.execute(plan, prune=False)    # writes .md files + manifest
result = syncer.sync(select=["a/b.pdf"])      # force selected files
```

Custom backends: subclass `pdftomd.Converter`, set `name` / `extensions`, and
implement `convert_bytes(data, *, filename) -> str`.

## Accuracy notes

* **marker**: `--force-ocr` re-OCRs every page (use on scans with broken or
  missing text layers); `--use-llm` enables marker's hybrid mode which merges
  tables across pages and fixes forms (uses Gemini under the hood).
* **gemini**: the whole document is sent in one request with a strict
  transcription prompt (merge multi-page tables, mark `[illegible]` rather than
  guess, temperature 0). Files over ~20 MB go through the Files API; output
  truncated at the token limit is continued automatically.

## Troubleshooting

**`CERTIFICATE_VERIFY_FAILED ... self-signed certificate in certificate chain`**
when marker downloads its models: you are behind a TLS-inspecting proxy
(Netskope, Zscaler, ...). Python does not use the macOS keychain, so export the
system roots and point Python at them:

```bash
mkdir -p ~/.config/pdftomd
{ security find-certificate -a -p /Library/Keychains/System.keychain
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain
} > ~/.config/pdftomd/ca-bundle.pem
export SSL_CERT_FILE=~/.config/pdftomd/ca-bundle.pem
export REQUESTS_CA_BUNDLE=~/.config/pdftomd/ca-bundle.pem
```

Add the two `export` lines to your shell profile to make it permanent. The
first marker run downloads ~2.6 GB of models into `~/Library/Caches/datalab`
and can look idle for several minutes; later runs start immediately.

## Development

```bash
uv run pytest
```
