# backend/core/rag.py
import re
from backend.core import vectorstore
from .llm import chat as llm_chat

SYS_ANSWER = (
    "Eres un asistente que responde usando SOLO el contexto proporcionado.\n"
    "Si el contexto es muy breve, haz el mejor resumen posible con lo disponible "
    "y explícitalo de forma transparente.\n"
    "Cita siempre así: (Doc:{doc_title}, pág {page})."
)

# ---------- utilidades léxicas ----------
def _keyword_terms(q: str):
    """Palabras significativas (>=4 letras, incluye tildes)."""
    terms = [t.lower() for t in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{4,}", q or "")]
    return list(dict.fromkeys(terms))  # únicos preservando orden

def _lex_contains_all(txt: str, terms):
    lt = (txt or "").lower()
    return all(t in lt for t in terms)

def _lex_contains_any(txt: str, terms):
    lt = (txt or "").lower()
    return any(t in lt for t in terms)

# ---------- recuperación híbrida ----------
def retrieve(collection: str, q: str, embed_fn, k: int = 8):
    """
    1) Recupera por embeddings (Chroma/mini).
    2) Normaliza score: si hay 'distances' (cosine 0..2), convierte a similitud en [-1,1].
    3) Si es ENTIDAD o top flojo, aplica boost léxico:
       - Sube el score de los hits que contengan TODOS los términos (0.95) o ALGUNO (0.70).
       - Además, inyecta chunks adicionales de la colección que cumplan lo mismo.
    """
    # --- 1) embeddings ---
    res = vectorstore.query(collection, [q], embed_fn, k=k)

    docs   = res.get("documents", [[]])[0] if res.get("documents") else []
    metas  = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
    ids    = res.get("ids", [[]])[0]       if res.get("ids")       else []

    # Preferimos distances si viene (más estable entre drivers); si no, usa scores crudos
    dists  = res.get("distances", [[None]])[0] if res.get("distances") else [None]*len(docs)
    rawsc  = res.get("scores",    [[None]])[0] if res.get("scores")    else [None]*len(docs)

    out = []
    for i, (txt, meta, id_) in enumerate(zip(docs, metas, ids)):
        dist = dists[i] if i < len(dists) else None
        if dist is not None:
            # Chroma cosine: distance ∈ [0,2]; similitud = 1 - distance ∈ [-1,1]
            score = 1.0 - float(dist)
        else:
            # Fallback: usa lo que vino como "score" (puede ser negativo en tu driver)
            sc = rawsc[i]
            score = float(sc) if sc is not None else 0.0

        out.append({"text": txt, "meta": meta, "id": id_, "score": score})

    # --- 2) heurística: entidad o mean_top bajo → boost léxico (sobre hits y extras) ---
    looks_entity = bool(re.search(r"\b(quien|quién|quienes|quiénes)\b", (q or "").lower())) \
                   or any(tok.istitle() for tok in (q or "").split())
    top3 = out[:3]
    mean_top = (sum(c["score"] for c in top3) / max(1, len(top3))) if top3 else 0.0

    if looks_entity or mean_top < 0.22:
        terms = _keyword_terms(q)

        # 2.a) BOOST sobre los hits ya recuperados (clave para tu caso)
        boosted = []
        for c in out:
            s = c["score"]
            t = c["text"] or ""
            if terms and _lex_contains_all(t, terms):
                s = max(s, 0.95)
            elif terms and _lex_contains_any(t, terms):
                s = max(s, 0.70)
            boosted.append({**c, "score": s})
        out = boosted

        # 2.b) Inyectar EXTRAS de toda la colección (si no estaban)
        if terms:
            docs_all, metas_all, ids_all = vectorstore.load_collection(collection)
            already = set(c["id"] for c in out)
            extras = []
            for txt, meta, cid in zip(docs_all, metas_all, ids_all):
                if cid in already:
                    continue
                if _lex_contains_all(txt, terms):
                    extras.append({"text": txt, "meta": meta, "id": cid, "score": 0.95})
                elif _lex_contains_any(txt, terms):
                    extras.append({"text": txt, "meta": meta, "id": cid, "score": 0.70})
            # mezcla y recorta
            out.extend(extras)
            out.sort(key=lambda x: -x["score"])
            out = out[:max(k, 8)]

    return out

# ---------- prompting ----------
def _context_from_docs(docs):
    ctx = []
    for d in docs:
        m = d["meta"] or {}
        cite = f"(Doc:{m.get('doc_title','?')}, pág {m.get('page','?')})"
        txt = (d["text"] or "").strip()
        if txt:
            ctx.append(f"{cite}\n{txt}")
    return "\n---\n".join(ctx)

def augment(user_q, docs):
    context = _context_from_docs(docs)
    return (
        f"Pregunta: {user_q}\n\n"
        f"Contexto (extractos con cita):\n{context}\n\n"
        "Instrucción: Responde con citas y sé explícito si el contexto es parcial."
    )

def answer(prompt):
    return llm_chat([
        {"role": "system", "content": SYS_ANSWER},
        {"role": "user",   "content": prompt}
    ])
