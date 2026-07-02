"""The RAG engine: extract text -> chunk -> embed -> search by similarity.

This is the heart of the project. No AI answering here yet — just the
"find the relevant pieces of your notes" machinery.
"""

import json
import numpy as np
import fitz  # PyMuPDF — stronger text extraction + can render pages to images for OCR
from io import BytesIO
from sentence_transformers import SentenceTransformer

import config

# Load the embedding model ONCE (downloaded on first run, then cached on disk).
# This is what turns text into a "meaning vector".
_model = SentenceTransformer(config.EMBEDDING_MODEL)

# EasyOCR reader is heavy to build, so create it lazily only if a scanned page
# actually needs OCR (keeps normal text PDFs fast, and startup light).
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # English only, CPU. First call downloads a small model (~64 MB), then cached.
        # verbose=False silences the progress bar, whose block glyph crashes the
        # Windows cp1252 console during the one-time model download.
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


def _ocr_page(page) -> str:
    """Render a PDF page to an image and OCR it — for scanned/image-only pages."""
    pix = page.get_pixmap(dpi=200)          # crisp enough for OCR, not too slow
    img_bytes = pix.tobytes("png")
    result = _get_ocr_reader().readtext(img_bytes, detail=0, paragraph=True)
    return "\n".join(result)


def extract_pages(file_bytes: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] — one entry per page (1-indexed).

    Uses PyMuPDF's text layer first. If a page has (almost) no extractable text
    — i.e. it's a scan/image — we fall back to OCR so image PDFs still work.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if len(text) < 20:                  # looks like a scanned/image page
            try:
                text = _ocr_page(page).strip()
            except Exception:
                pass                        # OCR failed — keep whatever we had
        pages.append((i, text))
    doc.close()
    return pages


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Join every page's text (kept for callers that don't need page numbers)."""
    return "\n".join(text for _, text in extract_pages(file_bytes))


def extract_pages_any(filename: str, file_bytes: bytes) -> list[tuple[int, str]]:
    """Extract [(page, text)] from PDF / DOCX / TXT based on the file extension.

    Only PDFs have real page numbers; DOCX/TXT return a single (None, text) block.
    """
    fn = filename.lower()
    if fn.endswith(".pdf"):
        return extract_pages(file_bytes)
    if fn.endswith(".docx"):
        import docx
        doc = docx.Document(BytesIO(file_bytes))
        return [(None, "\n".join(p.text for p in doc.paragraphs))]
    if fn.endswith(".txt"):
        return [(None, file_bytes.decode("utf-8", errors="ignore"))]
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


def chunk_text(text: str) -> list[str]:
    """Split a big block of text into overlapping ~200-word chunks.

    Overlap means a sentence cut in half still appears whole in a neighbour,
    so we never lose context at the boundaries.
    """
    words = text.split()
    chunks = []
    step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
    for start in range(0, len(words), step):
        piece = words[start:start + config.CHUNK_SIZE]
        if piece:
            chunks.append(" ".join(piece))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into a list of vectors (lists of floats)."""
    vectors = _model.encode(texts)
    return [v.tolist() for v in vectors]   # .tolist() so it's JSON-serialisable


def embed_one(text: str) -> np.ndarray:
    """Embed a single string (e.g. the user's question) as a numpy vector."""
    return _model.encode([text])[0]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How aligned are two vectors? 1.0 = identical meaning, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def search(question: str, chunks: list) -> list:
    """Return the TOP_K chunks whose meaning is closest to the question.

    `chunks` is a list of SQLAlchemy Chunk rows (each has .content and .embedding).
    """
    q_vec = embed_one(question)
    scored = []
    for chunk in chunks:
        chunk_vec = np.array(json.loads(chunk.embedding))   # JSON string -> vector
        score = _cosine_similarity(q_vec, chunk_vec)
        scored.append((score, chunk))

    # highest similarity first, keep the best TOP_K
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for score, chunk in scored[:config.TOP_K]]


def label_chunks(chunks: list):
    """Split retrieved chunks into (labeled_context, sources).

    `labeled_context` tags each chunk with its source doc + page so the model
    can cite and compare across documents. `sources` is the structured payload
    the API returns so the UI can show citations.
    """
    labeled, sources = [], []
    for c in chunks:
        fname = c.document.filename if c.document else "unknown"
        tag = f"[{fname}" + (f", p.{c.page}" if c.page else "") + "]"
        labeled.append(f"{tag}\n{c.content}")
        sources.append({"filename": fname, "page": c.page, "content": c.content})
    return labeled, sources
