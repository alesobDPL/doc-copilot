# backend/core/pdf.py
import pdfplumber
import tiktoken

def _get_encoder(model_hint: str = "gpt-4o-mini"):
    # Algunos modelos nuevos no están en tiktoken; usa cl100k_base como fallback
    try:
        return tiktoken.encoding_for_model(model_hint)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

def extract_pages(path: str):
    """
    Devuelve (page_number, text). Si el PDF es escaneado, puede venir vacío.
    """
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            yield i, text

def chunk_text(text: str, max_tokens=700, overlap=220, model: str = "gpt-4o-mini"):
    enc = _get_encoder(model)
    toks = enc.encode(text or "")
    chunks, start = [], 0
    n = len(toks)
    if n == 0:
        return []
    while start < n:
        end = min(start + max_tokens, n)
        chunks.append(enc.decode(toks[start:end]))
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks
