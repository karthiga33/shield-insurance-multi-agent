"""Document analysis for claim intake.

Handles three input kinds:
  * PDF (text-based)   -> extract text with pypdf
  * PDF (scanned)      -> render pages to images, then vision + OCR
  * Image (jpg/png...) -> OpenAI vision + OCR

The analyzer produces:
  1. raw_text      - best-effort plain text of the document
  2. extracted     - a structured dict of claim fields pulled out by the LLM

The LLM extraction is schema-guided so downstream agents get predictable keys.
If no OpenAI key is configured, the analyzer still returns raw_text (from
pypdf / OCR) and an empty structured extraction, so the pipeline degrades
gracefully instead of crashing.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentcore_client
import config
import llm
import policy_rules

_PDFIUM_LOCK = threading.Lock()

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXTS = {".pdf"}

# Fields we try to pull out of any claim document.
EXTRACTION_SCHEMA = {
    "claimant_name": "Full name of the person filing the claim",
    "policy_number": "Insurance policy number if present",
    "incident_type": "Normalized incident type, one of: collision, theft, vandalism, "
    "fire, flood, water_damage, storm, wear_and_tear, hospitalization, surgery, "
    "emergency, cosmetic, death, terminal_illness. Pick the closest match.",
    "incident_date": "Date of the incident in YYYY-MM-DD format",
    "claimed_amount": "Total amount being claimed, as a number (no currency symbol)",
    "description": "Short description of what happened",
    "document_kind": "What kind of document this is (e.g. claim form, repair invoice, "
    "police report, medical bill, photo of damage)",
    "line_items": "List of itemized costs if the document is an invoice/bill, else empty list",
}


def classify_document(path: str) -> str:
    """Return 'pdf', 'image', or 'unknown' based on the file extension."""
    ext = Path(path).suffix.lower()
    if ext in PDF_EXTS:
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    return "unknown"


# --- Text extraction -------------------------------------------------------
def _extract_pdf_text(path: str) -> str:
    """Pull embedded text from a text-based PDF."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        parts = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(parts).strip()
    except Exception as exc:  # pragma: no cover - defensive
        return f"[pdf text extraction failed: {exc}]"


def _ocr_image(path_or_image) -> str:
    """Run Tesseract OCR on an image path or PIL image. Empty string on failure."""
    try:
        import pytesseract
        from PIL import Image

        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

        img = path_or_image
        if isinstance(path_or_image, (str, Path)):
            img = Image.open(path_or_image)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        # Tesseract binary may not be installed; that's fine, vision handles it.
        return ""


def _pdf_pages_to_images(path: str, max_pages: int = 6) -> List[Any]:
    """Render PDF pages to PIL images.

    Uses pypdfium2 (pure pip install, no external binary). Falls back to
    pdf2image if Poppler happens to be installed.
    """
    try:
        import pypdfium2 as pdfium

        # pdfium is not thread-safe, and documents can be analysed in parallel.
        with _PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(path)
            try:
                pages = min(len(pdf), max_pages)
                # scale 2.8 is about 200 DPI, good for OCR and vision.
                return [pdf[i].render(scale=2.8).to_pil() for i in range(pages)]
            finally:
                pdf.close()
    except Exception:
        pass
    try:
        from pdf2image import convert_from_path

        return convert_from_path(path, dpi=200, last_page=max_pages)
    except Exception:
        return []


def _looks_scanned(text: str) -> bool:
    """A text-based PDF yields plenty of characters; a scan yields almost none."""
    return len(text.strip()) < 30


# --- Vision extraction -----------------------------------------------------
_MODEL_IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _encode_pil(img) -> str:
    """PNG data URL for an in-memory PIL image."""
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_image_file(path: str) -> str:
    """Return a base64 data URL in a format vision models accept.

    Bedrock takes PNG/JPEG/GIF/WebP only, so BMP/TIFF uploads are converted
    to PNG first.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    if mime in _MODEL_IMAGE_MIMES:
        with open(path, "rb") as fh:
            raw = fh.read()
    else:
        import io
        from PIL import Image

        buf = io.BytesIO()
        Image.open(path).convert("RGB").save(buf, format="PNG")
        raw, mime = buf.getvalue(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('utf-8')}"


def _vision_read(image_data_urls: List[str], ocr_hint: str) -> str:
    """Ask the vision model to transcribe/describe the document images."""
    if not llm.is_available():
        return ocr_hint
    prompt = (
        "Transcribe all readable text from this insurance claim document "
        "and briefly describe any images of damage. "
        f"OCR hint (may be noisy): {ocr_hint[:1500]}"
    )
    try:
        return llm.complete(prompt, images=image_data_urls, max_tokens=2000) or ocr_hint
    except Exception:
        # Model unavailable (credentials, access, outage): keep the OCR text
        # so the claim still goes through the pipeline.
        return ocr_hint


def _structure_fields(raw_text: str, schema: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Use the LLM to pull structured fields out of the raw text.

    `schema` maps field name -> description; defaults to the claim schema.
    """
    if not llm.is_available() or not raw_text.strip():
        return {}
    schema = schema or EXTRACTION_SCHEMA
    schema_desc = "\n".join(f"- {k}: {v}" for k, v in schema.items())
    prompt = (
        "You extract structured data from an insurance document.\n"
        "Return ONLY a JSON object with these keys (use null if unknown):\n"
        f"{schema_desc}\n\n"
        "Document text:\n"
        f"{raw_text[:6000]}"
    )
    try:
        fields = json.loads(llm.complete(prompt, json_mode=True, max_tokens=1200))
    except Exception:
        # Includes model errors and invalid JSON: treat as "nothing extracted".
        return {}
    if not isinstance(fields, dict):
        return {}
    # Keep incident_type only if it is one of our known types. A label like
    # "repair" (from a repair invoice) is not an incident and would otherwise
    # trigger a false "incident type mismatch" fraud flag.
    if "incident_type" in schema:
        incident = str(fields.get("incident_type") or "").strip().lower()
        fields["incident_type"] = incident if incident in policy_rules.INCIDENT_RULES else None
    return fields


# --- Public entry point ----------------------------------------------------
def analyze_document(path: str, schema: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Analyze a document and return {document_type, raw_text, extracted}.

    `schema` selects which fields to extract (claim fields by default).
    """
    if agentcore_client.enabled():
        try:
            return agentcore_client.analyze_document(path, schema)
        except Exception as exc:
            # Keep the claim moving: read the document locally instead.
            print(f"[agentcore] analyze_document failed, reading locally: {exc}")
    kind = classify_document(path)
    raw_text = ""
    doc_type = kind

    if kind == "pdf":
        raw_text = _extract_pdf_text(path)
        if _looks_scanned(raw_text):
            doc_type = "scanned"
            images = _pdf_pages_to_images(path)
            ocr_text = "\n".join(_ocr_image(img) for img in images) if images else ""
            # Encode rendered pages in memory (no shared temp files, so parallel
            # analyses cannot overwrite each other's pages).
            data_urls = [_encode_pil(img) for img in images[:4]]  # cap pages sent to vision
            raw_text = _vision_read(data_urls, ocr_text) if data_urls else ocr_text
    elif kind == "image":
        doc_type = "image"
        ocr_text = _ocr_image(path)
        raw_text = _vision_read([_encode_image_file(path)], ocr_text)
    else:
        raw_text = "[unsupported document type]"

    extracted = _structure_fields(raw_text, schema)
    return {
        "document_type": doc_type,
        "raw_text": raw_text,
        "extracted": extracted,
    }
