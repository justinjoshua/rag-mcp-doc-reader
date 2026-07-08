"""OCR for scanned / image-only PDFs via Gemini vision.

pypdfium2 renders each page to an image (no system dependency); Gemini
transcribes the text. Used as a fallback in ingest.parse_file when a PDF has no
extractable text layer. Requires a Gemini API key.
"""
from __future__ import annotations

from io import BytesIO

from .config import CONFIG

_BATCH = 4  # pages per Gemini call — bounds output size and cost
_PROMPT = (
    "Transcribe ALL text from these document/slide page images verbatim, in "
    "reading order. Include titles, headings, bullet points, and any table text. "
    "Separate pages with a blank line. Output only the transcribed text — no "
    "commentary, no markdown fences."
)


def _page_jpeg(page) -> bytes:
    pil = page.render(scale=2.0).to_pil().convert("RGB")
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def ocr_pdf(path) -> str:
    """Return text transcribed from a PDF's page images. Empty string on failure."""
    if not CONFIG.gemini_api_key:
        return ""
    import pypdfium2 as pdfium
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=CONFIG.gemini_api_key)
    pdf = pdfium.PdfDocument(str(path))
    n_pages = min(len(pdf), CONFIG.ocr_max_pages)

    out: list[str] = []
    try:
        for start in range(0, n_pages, _BATCH):
            parts = [types.Part.from_bytes(data=_page_jpeg(pdf[i]), mime_type="image/jpeg")
                     for i in range(start, min(start + _BATCH, n_pages))]
            try:
                resp = client.models.generate_content(
                    model=CONFIG.gemini_model, contents=[_PROMPT, *parts]
                )
                if resp.text:
                    out.append(resp.text.strip())
            except Exception:  # noqa: BLE001 - skip a bad batch, keep the rest
                continue
    finally:
        pdf.close()
    return "\n\n".join(out)
