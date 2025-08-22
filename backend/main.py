import os, re, unicodedata, uuid, itertools, hashlib
from io import BytesIO

# 1) Cargar .env ANTES de importar nada de backend.core.*
from dotenv import load_dotenv

load_dotenv()

# 2) Recién ahora importa FastAPI y tus módulos que leen env
from fastapi import FastAPI, UploadFile, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.core.pdf import chunk_text
from backend.core.llm import embed, chat as llm_chat
from backend.core.vectorstore import upsert_chunks, load_collection, clear_collection
from backend.core.rag import retrieve, augment, answer
from backend.core.orchestrator import run_chat

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict, Counter
from backend.core import vectorstore as vstore


APP_VERSION = os.getenv("APP_VERSION", "0.1.0")


app = FastAPI()


# --- helper: coseno seguro ---
def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float((a @ b) / (na * nb))


def _sha1_hex(b: bytes, n: int = 16) -> str:
    """Hash SHA1 (hex) truncado a n caracteres."""
    return hashlib.sha1(b).hexdigest()[:n]


def _chunk_id(doc_hash: str, page_idx: int, c_idx: int, chunk_text_str: str) -> str:
    """
    ID único y estable por chunk:
    - doc_hash: hash del PDF completo (binario)
    - page_idx/c_idx: trazabilidad
    - chunk_hash: hash del texto del chunk (estabilidad por contenido)
    """
    chunk_hash = _sha1_hex(chunk_text_str.encode("utf-8"), n=12)
    return f"{doc_hash}|p{page_idx:04d}|c{c_idx:04d}|{chunk_hash}"


def _auto_max_per_doc(n_chunks: int) -> int:
    """
    Heurística simple para elegir cuántos chunks muestrear por documento.
    Ajusta a gusto si tus PDFs son muy largos/cortos.
    """
    if n_chunks <= 6:
        return n_chunks
    if n_chunks <= 20:
        return 12
    if n_chunks <= 50:
        return 18
    return 24

def _sample_cover(chunks: list[str], k: int) -> list[str]:
    """
    Muestreo 'estratificado' sencillo para cubrir el documento.
    """
    if not chunks or k <= 0:
        return []
    if len(chunks) <= k:
        return chunks
    step = max(1, len(chunks) // k)
    return [chunks[i] for i in range(0, len(chunks), step)][:k]

def _synthesize_single_doc_summary(doc_title: str, chosen_chunks: list[str]) -> str:
    """
    Llama al LLM para resumir un documento con los chunks seleccionados.
    Respeta FAKE_LLM para tests/local sin clave real.
    """
    text_blob = "\n\n---\n\n".join(chosen_chunks).strip()
    if not text_blob:
        return "No hay contenido disponible para resumir."

    system = (
        "Eres un analista que resume documentos con precisión y SIN inventar.\n"
        "Usa exclusivamente el texto entregado. Redacta en español, conciso y claro.\n"
        "Si el contenido parece parcial, indícalo al final como 'Cobertura parcial'."
    )
    user = (
        f"Documento: {doc_title}\n\n"
        f"Texto (fragmentos seleccionados):\n{text_blob}\n\n"
        "Tarea: Devuelve un resumen en 5-8 viñetas, breves y específicas."
    )

    # Soporte para entorno de pruebas sin OpenAI real
    if os.getenv("FAKE_LLM", "0") == "1":
        bullets = [
            "• Resumen simulado (FAKE_LLM) basado en fragmentos entregados.",
            "• Este es un placeholder para pruebas locales.",
            "• En producción, se usa un modelo LLM real (OpenAI).",
        ]
        return "\n".join(bullets)

    try:
        out = llm_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return (out or "").strip() or "No se pudo generar el resumen."
    except Exception as e:
        # Nunca explotes: devuelve texto útil para el front
        return f"No se pudo generar el resumen (error LLM): {e}"

def _synthesize_overview(per_doc_summaries: list[dict]) -> str:
    """
    Genera un overview ejecutivo de la colección a partir de los resúmenes por documento.
    """
    if not per_doc_summaries:
        return ""
    joined = []
    for item in per_doc_summaries:
        t = item.get("doc_title", "Documento")
        s = (item.get("summary") or "").strip()
        if s:
            joined.append(f"# {t}\n{s}")
    corpus = "\n\n====\n\n".join(joined)

    system = (
        "Eres un analista senior. Lee resúmenes de varios documentos y redacta un "
        "overview ejecutivo en 5-8 viñetas, claro y accionable, sin repetir textualmente."
    )
    user = f"Resúmenes por documento:\n\n{corpus}\n\nTarea: overview ejecutivo en viñetas."

    if os.getenv("FAKE_LLM", "0") == "1":
        return (
            "• Overview simulado (FAKE_LLM): la colección cubre temas diversos.\n"
            "• Los documentos comparten estructura formal.\n"
            "• Hay diferencias en propósito y audiencia según el tipo de carta.\n"
        )
    try:
        out = llm_chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        return (out or "").strip()
    except Exception as e:
        return f"No se pudo generar el overview (error LLM): {e}"

@app.on_event("startup")
def _startup_log():
    print(f"[startup] VSTORE_BACKEND={vstore.BACKEND}")


@app.get("/healthz")
def healthz():
    return {"ok": True, "version": APP_VERSION}


@app.get("/version")
def version():
    return {"version": APP_VERSION}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_title(name: str) -> str:
    base = os.path.basename(name or "")
    base = unicodedata.normalize("NFKC", base).strip()
    base = re.sub(r"\s+", " ", base)  # colapsa espacios
    return base


@app.post("/ingest")
async def ingest(collection: str = Form(...), files: list[UploadFile] = []):
    if not collection:
        raise HTTPException(
            status_code=400, detail="El campo 'collection' es obligatorio."
        )
    if not files:
        raise HTTPException(status_code=400, detail="Debes subir al menos 1 PDF.")
    files = files[:5]

    chunks, metas, ids = [], [], []
    try:
        for f in files:
            name = normalize_title(f.filename or f"doc-{uuid.uuid4()}.pdf")
            if not name.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"'{name}' no es un PDF.")

            # leer el archivo COMPLETO en memoria
            data = await f.read()
            if not data:
                raise HTTPException(status_code=400, detail=f"'{name}' está vacío.")

            # === NUEVO: hash estable del PDF (namespace del doc) ===
            doc_hash = _sha1_hex(data, n=16)

            # abrir en memoria (sin usar /tmp)
            buf = BytesIO(data)

            # --- extracción y chunking por página ---
            import pdfplumber

            total_before = len(chunks)

            with pdfplumber.open(buf) as pdf:
                for page_idx, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    # Puedes limpiar texto aquí para estabilidad del hash (opcional):
                    # text = re.sub(r"-\n", "", text)  # juntar guiones de corte de línea
                    for c_idx, ch in enumerate(chunk_text(text)):
                        chunks.append(ch)
                        metas.append(
                            {
                                "doc_title": name,
                                "page": page_idx,
                                "chunk_index": c_idx,
                                "doc_id": doc_hash,  # <-- NUEVO: útil para filtros
                            }
                        )
                        # === NUEVO: ID que no colisiona al re-ingestar ===
                        ids.append(_chunk_id(doc_hash, page_idx, c_idx, ch))

            if len(chunks) == total_before:
                # No salió texto: probablemente escaneado
                raise HTTPException(
                    status_code=400,
                    detail=f"'{name}' no contiene texto extraíble (¿PDF escaneado?).",
                )

        # Embeddings + upsert (tu driver Chroma ya hace upsert o delete+add)
        upsert_chunks(collection, chunks, metas, ids, embed_fn=lambda xs: embed(xs))
        titles = sorted({m["doc_title"] for m in metas})
        return {"ok": True, "chunks": len(chunks), "docs": titles}

    except HTTPException:
        raise
    except Exception as e:
        print("[/ingest] ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Falló la indexación. Revisa la consola del backend para el error exacto.",
        )


@app.post("/chat")
async def chat_rag(payload: dict = Body(...)):
    try:
        q = payload.get("question", "").strip()
        collection = payload.get("collection", "").strip()
        if not collection:
            raise HTTPException(
                status_code=400, detail="El campo 'collection' es obligatorio."
            )
        if not q:
            raise HTTPException(
                status_code=400, detail="El campo 'question' es obligatorio."
            )

        def _retrieve(coll, query, k=8):
            return retrieve(coll, query, embed_fn=lambda xs: embed(xs), k=k)

        def _complete(system, user):
            return llm_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )

        out = run_chat(q, collection, _retrieve, _complete)

        # Tu heurística de reintento si no cita nada (opcional, ya lo hace el orquestador):
        if "(Doc:" not in out["answer"]:
            out = run_chat(q, collection, _retrieve, _complete)

        return out

    except HTTPException:
        raise
    except Exception as e:
        print("[/chat] ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail="Falló el chat. Revisa la consola del backend para el error exacto.",
        )


@app.post("/debug/retrieval")
async def debug_retrieval(payload: dict = Body(...)):
    q = (payload.get("question") or "").strip()
    collection = (payload.get("collection") or "").strip()
    k = int(payload.get("k", 8))
    if not collection or not q:
        raise HTTPException(
            status_code=400, detail="Campos requeridos: collection, question."
        )
    docs = retrieve(collection, q, embed_fn=lambda xs: embed(xs), k=k)
    return {
        "hits": [
            {
                "score": d.get("score"),
                "meta": d["meta"],
                "excerpt": (d["text"] or "")[:400],
            }
            for d in docs
        ]
    }


@app.get("/collections/{collection}/docs")
def list_docs(collection: str):
    try:
        docs, metas, ids = load_collection(collection)
        titles = sorted({(m.get("doc_title") or "").strip() for m in metas})
        return {"collection": collection, "docs": titles, "count": len(titles)}
    except Exception as e:
        # Evita 500 y entrega mensaje útil al frontend
        return {"collection": collection, "docs": [], "count": 0, "error": str(e)}


@app.delete("/collections/{collection}")
def clear(collection: str):
    clear_collection(collection)
    return {"ok": True, "cleared": collection}


@app.post("/debug/retrieval")
async def debug_retrieval(payload: dict = Body(...)):
    q = payload.get("question", "").strip()
    collection = payload.get("collection", "").strip()
    k = int(payload.get("k", 8))
    if not collection or not q:
        raise HTTPException(
            status_code=400, detail="Campos requeridos: collection, question."
        )

    docs = retrieve(collection, q, embed_fn=lambda xs: embed(xs), k=k)
    # Devolvemos texto y metadatos para que veas si llegó algo útil
    return {
        "hits": [{"meta": d["meta"], "excerpt": (d["text"] or "")[:400]} for d in docs]
    }


@app.post("/summary")
def summary(payload: dict = Body(...)):
    """
    Resumen de un documento o de toda la colección.
    Request:
      {
        "collection": "default",
        "doc_title": "opcional; si no viene, resume colección",
        "max_per_doc": 12,       # opcional; si no viene, se elige automáticamente por doc
        "include_overview": true # opcional; aplica cuando NO viene doc_title
      }
    """
    try:
        collection = (payload.get("collection") or "").strip()
        if not collection:
            raise HTTPException(status_code=400, detail="collection es obligatorio")

        target_doc = payload.get("doc_title")  # opcional
        include_overview = bool(payload.get("include_overview", True))
        manual_max = payload.get("max_per_doc")

        docs, metas, ids = load_collection(collection)

        # Agrupar por doc_title, preservando orden de aparición
        per_doc = defaultdict(list)  # title -> list[str]
        for txt, meta in zip(docs, metas):
            title = (meta or {}).get("doc_title", "¿?")
            per_doc[title].append(txt or "")

        # --- Caso: resumen de UN documento ---
        if target_doc:
            chunks_doc = per_doc.get(target_doc, [])
            if not chunks_doc:
                raise HTTPException(status_code=404, detail=f"No existe '{target_doc}' en {collection}")

            use_k = int(manual_max) if manual_max else _auto_max_per_doc(len(chunks_doc))
            chosen = _sample_cover(chunks_doc, use_k)

            summary_text = _synthesize_single_doc_summary(target_doc, chosen)
            return {
                "mode": "single",
                "doc_title": target_doc,
                "summary": summary_text,
                "chunks_used": len(chosen),
            }

        # --- Caso: resumen de COLECCIÓN ---
        per_doc_summaries = []
        for title, chunks_doc in per_doc.items():
            use_k = int(manual_max) if manual_max else _auto_max_per_doc(len(chunks_doc))
            chosen = _sample_cover(chunks_doc, use_k)
            summary_text = _synthesize_single_doc_summary(title, chosen)
            per_doc_summaries.append({
                "doc_title": title,
                "summary": summary_text,
                "chunks_used": len(chosen),
            })

        overview = _synthesize_overview(per_doc_summaries) if include_overview else ""
        return {
            "mode": "collection_per_doc",
            "overview": overview,
            "per_doc": per_doc_summaries,
        }

    except HTTPException:
        # Deja que FastAPI maneje los 4xx
        raise
    except Exception as e:
        # En 500, devuelve JSON explícito
        print("[/summary] ERROR:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Falló el resumen. Revisa logs del backend. ({e})"
        )

@app.post("/compare")
async def compare_docs(payload: dict = Body(...)):
    collection = payload.get("collection", "").strip()
    doc_a = payload.get("doc_a")
    doc_b = payload.get("doc_b")
    if not collection or not doc_a or not doc_b:
        raise HTTPException(
            status_code=400, detail="Campos requeridos: collection, doc_a, doc_b."
        )

    docs, metas, ids = load_collection(collection)
    A = [d for d, m in zip(docs, metas) if m.get("doc_title") == doc_a][:15]
    B = [d for d, m in zip(docs, metas) if m.get("doc_title") == doc_b][:15]
    if not A or not B:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron chunks para alguno de los documentos.",
        )

    context = (
        f"=== {doc_a} ===\n" + "\n".join(A) + 
        f"\n\n=== {doc_b} ===\n" + "\n".join(B)
    )
    prompt = (
        "Compara los dos documentos. Responde con:\n"
        "1) Resumen de cada documento (3 bullets c/u)\n"
        "2) Similitudes (3–5 bullets)\n"
        "3) Diferencias clave (5 bullets)\n"
        "4) Riesgos/lagunas en cada uno (2–3 bullets)\n"
        "Sé concreto y cita fragmentos si es útil."
    )
    out = llm_chat(
        [
            {"role": "system", "content": "Eres un analista técnico y claro."},
            {"role": "user", "content": f"{prompt}\n\n{context}"},
        ]
    )
    return {"comparison": out}


@app.post("/compare/global")
async def compare_global(payload: dict = Body(...)):
    """
    Comparación global AUTOMÁTICA:
      - Sin parámetros del usuario.
      - Centroide por documento (promedio de hasta 20 chunks representativos).
      - Matriz de similitud coseno doc-doc.
      - Umbral auto = P75 de similitudes off-diagonal (o top_k fallback).
      - 'top_pairs': los pares más similares con resumen compacto (similitudes/diferencias).
      - 'most_similar_by_doc': pareja más parecida por documento.
      - 'groups': clusters de documentos afines (componentes conexas sobre threshold).
    """
    collection = payload.get("collection", "").strip()
    if not collection:
        raise HTTPException(
            status_code=400, detail="El campo 'collection' es obligatorio."
        )

    # Hiperparámetros automáticos
    CENTROID_MAX_CHUNKS = 20  # por doc para centroide
    COMPARE_TOPK = 6  # extractos por doc para resumen por par
    TOP_PAIRS_CAP = 10  # máximo de pares que “redactamos”
    PERCENTILE = 0.75  # umbral auto = percentil 75

    # Carga colección
    docs, metas, ids = load_collection(collection)
    if not docs:
        raise HTTPException(status_code=404, detail="Colección vacía o inexistente.")

    # Agrupar índices por documento
    by_doc_idx = {}
    for i, m in enumerate(metas):
        title = (m.get("doc_title") or "").strip()
        by_doc_idx.setdefault(title, []).append(i)

    doc_titles = sorted(by_doc_idx.keys())
    n = len(doc_titles)
    if n < 2:
        raise HTTPException(
            status_code=400, detail="Se requieren al menos 2 documentos para comparar."
        )

    # Muestra representativa por doc (espaciado uniforme)
    doc_samples = {}
    for t in doc_titles:
        idxs = by_doc_idx[t]
        if len(idxs) <= CENTROID_MAX_CHUNKS:
            pick = idxs
        else:
            step = max(1, len(idxs) // CENTROID_MAX_CHUNKS)
            pick = idxs[::step][:CENTROID_MAX_CHUNKS]
        doc_samples[t] = pick

    # Embeddings de todas las muestras (batch único)
    flat_texts, flat_map = [], []  # (title, idx_global)
    for t in doc_titles:
        for gi in doc_samples[t]:
            flat_texts.append(docs[gi])
            flat_map.append((t, gi))

    try:
        flat_embs = np.array(embed(flat_texts), dtype=np.float32)
    except Exception as e:
        print("[/compare/global] embed ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Fallo generando embeddings.")

    # Reagrupar embeddings por doc
    per_doc_embs = {t: [] for t in doc_titles}
    for (t, gi), emb in zip(flat_map, flat_embs):
        per_doc_embs[t].append(emb)

    # Centroide por documento
    centroids = {}
    for t in doc_titles:
        X = np.stack(per_doc_embs[t], axis=0)
        centroids[t] = X.mean(axis=0)

    # Matriz de similitud
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(doc_titles):
        for j, b in enumerate(doc_titles):
            if j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
            else:
                sim_matrix[i, j] = _cosine_sim(centroids[a], centroids[b])

    # Distribución de similitudes off-diagonal
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(float(sim_matrix[i, j]))

    # Umbral auto: percentil 75 (o fallback a top-K si colección chica)
    if len(off_diag) == 0:
        auto_threshold = 0.0
    else:
        auto_threshold = float(np.quantile(off_diag, PERCENTILE))

    # Pares ordenados por similitud desc
    pairs_all = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs_all.append((doc_titles[i], doc_titles[j], float(sim_matrix[i, j])))
    pairs_all.sort(key=lambda x: -x[2])

    # Fallback: si casi no hay pares > umbral, toma top K
    selected_pairs = [p for p in pairs_all if p[2] >= auto_threshold]
    if len(selected_pairs) == 0 and len(pairs_all) > 0:
        selected_pairs = pairs_all[: min(TOP_PAIRS_CAP, len(pairs_all))]

    # “Pareja más similar” por documento
    most_similar_by_doc = {}
    for i, a in enumerate(doc_titles):
        best = None
        best_sim = -1.0
        for j, b in enumerate(doc_titles):
            if i == j:
                continue
            sim = float(sim_matrix[i, j])
            if sim > best_sim:
                best_sim = sim
                best = b
        most_similar_by_doc[a] = {"doc": best, "sim": round(best_sim, 4)}

    # Grupos de afinidad (grafo con edges >= threshold; componentes conexas)
    adj = {t: set() for t in doc_titles}
    for a, b, s in pairs_all:
        if s >= auto_threshold:
            adj[a].add(b)
            adj[b].add(a)

    visited, groups = set(), []
    for t in doc_titles:
        if t in visited:
            continue
        comp, stack = [], [t]
        visited.add(t)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
        groups.append(sorted(comp))
    groups.sort(key=lambda g: (-len(g), g))

    # Para los pares seleccionados: redactar comparación breve con extractos “centrales”
    def top_central_snips(title, topk):
        idxs = doc_samples[title]
        E = np.stack(per_doc_embs[title], axis=0)
        c = centroids[title]
        scores = (E @ c) / (
            (np.linalg.norm(E, axis=1) + 1e-9) * (np.linalg.norm(c) + 1e-9)
        )
        order = np.argsort(-scores)[:topk]
        return [docs[idxs[i]] for i in order]

    pairs_out = []
    for a, b, sim in selected_pairs[:TOP_PAIRS_CAP]:
        a_snips = top_central_snips(a, COMPARE_TOPK)
        b_snips = top_central_snips(b, COMPARE_TOPK)
        prompt = (
            "Compara brevemente los dos documentos a partir de extractos representativos.\n"
            f"Documento A: {a}\nDocumento B: {b}\n\n"
            "Responde únicamente en dos secciones de bullets en Markdown:\n"
            "## Similitudes\n- ...\n- ...\n"
            "## Diferencias\n- ...\n- ...\n"
            "No inventes contenido que no esté respaldado por los extractos.\n\n"
            "=== Extractos A ===\n"
            + "\n---\n".join(a_snips)
            + "\n\n=== Extractos B ===\n"
            + "\n---\n".join(b_snips)
        )
        comp = llm_chat(
            [
                {
                    "role": "system",
                    "content": "Eres analítico y conciso; escribe bullets claros.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        pairs_out.append({"a": a, "b": b, "sim": round(sim, 4), "comparison": comp})

    explain = (
        f"Se calculó la similitud coseno entre documentos. "
        f"El umbral se fijó en el percentil {int(PERCENTILE*100)} de las similitudes off-diagonal "
        f"(auto_threshold={auto_threshold:.2f}). Se muestran hasta {TOP_PAIRS_CAP} pares."
    )

    return {
        "docs": doc_titles,
        "similarity_matrix": sim_matrix.tolist(),
        "auto_threshold": round(auto_threshold, 4),
        "explain_threshold": explain,
        "most_similar_by_doc": most_similar_by_doc,
        "groups": groups,
        "top_pairs": pairs_out,
    }


@app.post("/classify")
async def classify_topics(payload: dict = Body(...)):
    """
    Clasifica chunks en tópicos (clusters) con KMeans y k automático (Opción B).
    Devuelve mezcla de tópicos a nivel de documento (proporciones).
    """
    collection = payload.get("collection", "").strip()
    if not collection:
        raise HTTPException(
            status_code=400, detail="El campo 'collection' es obligatorio."
        )

    # límites de eficiencia
    max_chunks_global = int(
        payload.get("max_chunks_global", 1000)
    )  # cap duro para colecciones muy grandes
    max_examples_per_topic = int(
        payload.get("max_examples_per_topic", 5)
    )  # para nombrar con LLM
    random_state = int(payload.get("random_state", 42))

    # carga
    docs, metas, ids = load_collection(collection)
    if not docs:
        raise HTTPException(status_code=404, detail="Colección vacía.")

    N = len(docs)
    # muestrear si hay demasiados chunks (mantener distribución por doc lo mejor posible)
    if N > max_chunks_global:
        # muestrear estratificado por documento
        by_doc = defaultdict(list)
        for i, (d, m) in enumerate(zip(docs, metas)):
            by_doc[(m.get("doc_title") or "").strip()].append(i)
        # cuántos por doc
        docs_titles = list(by_doc.keys())
        share = max(1, max_chunks_global // max(1, len(docs_titles)))
        picked_idx = []
        for t in docs_titles:
            idxs = by_doc[t]
            step = max(1, len(idxs) // share)
            picked_idx.extend(idxs[::step][:share])
        picked_idx = picked_idx[:max_chunks_global]
        picked_idx.sort()
    else:
        picked_idx = list(range(N))

    docs_slice = [docs[i] for i in picked_idx]
    metas_slice = [metas[i] for i in picked_idx]

    # embeddings de chunks
    try:
        embs = np.array(embed(docs_slice), dtype=np.float32)
    except Exception as e:
        print("[/classify] embed ERROR:", repr(e))
        raise HTTPException(status_code=500, detail="Fallo generando embeddings.")

    if embs.ndim != 2 or embs.shape[0] < 2:
        raise HTTPException(
            status_code=400, detail="No hay suficientes datos para clusterizar."
        )

    n_chunks = embs.shape[0]

    # nº de documentos en esta colección (observados en los chunks tomados)
    doc_titles_in_slice = [(m.get("doc_title") or "").strip() for m in metas_slice]
    uniq_docs = sorted(set(doc_titles_in_slice))
    n_docs = max(1, len(uniq_docs))

    # rango para k: 2..min(8, n_docs+3, n_chunks-1)
    k_min = 2
    k_max = max(k_min, min(8, n_docs + 3, n_chunks - 1))

    best_k = None
    best_score = -1.0
    scores = []

    # evalúa silhouette por k
    for k in range(k_min, k_max + 1):
        try:
            km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
            labels = km.fit_predict(embs)
            # silhouette puede fallar si 1 clúster o clusters muy pequeños → manejar
            if len(set(labels)) < 2:
                score = -1.0
            else:
                score = silhouette_score(embs, labels, metric="cosine")
        except Exception:
            score = -1.0
        scores.append({"k": k, "silhouette": float(score)})
        if score > best_score:
            best_score = score
            best_k = k

    # sesgo suave hacia k ≈ nº docs si hay empate o casi-empate (delta <= 0.01)
    if n_docs >= k_min and n_docs <= k_max:
        # busca el score de k cercano a n_docs
        def get_score_for(k):
            for s in scores:
                if s["k"] == k:
                    return s["silhouette"]
            return -1.0

        target_k = n_docs
        target_score = get_score_for(target_k)
        if target_score >= 0:
            # si la diferencia es pequeña, favorece target_k
            if (best_score - target_score) <= 0.01:
                best_k = target_k

    k = best_k if best_k is not None else min(max(3, n_docs), 8)

    # clustering final con k elegido
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(embs)

    # agrupar por cluster
    grouped = defaultdict(list)
    for idx, lbl in enumerate(labels):
        grouped[int(lbl)].append(idx)

    # nombra clusters con LLM (toma hasta max_examples_per_topic ejemplos por cluster)
    topics = []
    topic_name_map = {}
    for lbl in sorted(grouped.keys()):
        idxs = grouped[lbl]
        # ejemplos para el naming
        ex_texts = [docs_slice[i] for i in idxs[:max_examples_per_topic]]
        sample = "\n---\n".join(ex_texts)
        prompt = (
            "Nombra este tópico en 1–3 etiquetas cortas (coma separadas) "
            "y da 2 bullets que lo describan. No inventes datos fuera del texto.\n\n"
            f"{sample}"
        )
        try:
            name = llm_chat(
                [
                    {
                        "role": "system",
                        "content": "Eres un taxonomista de temas, breve y preciso.",
                    },
                    {"role": "user", "content": prompt},
                ]
            ).strip()
        except Exception:
            name = f"Tópico {lbl}"

        topic_name_map[int(lbl)] = name

        # ejemplos de metadatos (3) para UI
        meta_examples = [metas_slice[i] for i in idxs[:3]]

        topics.append(
            {
                "cluster": int(lbl),
                "label": name,
                "count": len(idxs),
                "examples": meta_examples,
            }
        )

    # mezcla de tópicos por documento (proporciones)
    # cuenta chunks por (doc_title, cluster)
    doc_cluster_counts = defaultdict(lambda: Counter())
    for idx, lbl in enumerate(labels):
        title = doc_titles_in_slice[idx]
        doc_cluster_counts[title][int(lbl)] += 1

    # convierte a porcentajes por doc
    doc_topic_distribution = {}
    for title, counter in doc_cluster_counts.items():
        total = sum(counter.values())
        dist = {}
        for lbl, cnt in counter.items():
            dist[topic_name_map.get(lbl, f"Tópico {lbl}")] = round(cnt / total, 4)
        # ordena por mayor proporción
        doc_topic_distribution[title] = dict(sorted(dist.items(), key=lambda x: -x[1]))

    # top documentos por tópico (útil para UI)
    topic_doc_counts = defaultdict(lambda: Counter())
    for idx, lbl in enumerate(labels):
        title = doc_titles_in_slice[idx]
        topic_doc_counts[int(lbl)][title] += 1

    top_docs_per_topic = {}
    for lbl, counter in topic_doc_counts.items():
        label_name = topic_name_map.get(lbl, f"Tópico {lbl}")
        top_docs = counter.most_common()
        # convierto a porcentajes sobre total del cluster
        total_lbl = sum(counter.values())
        top_docs = [(doc, round(cnt / total_lbl, 4)) for doc, cnt in top_docs]
        top_docs_per_topic[label_name] = top_docs

    # ordena topics por tamaño
    topics.sort(key=lambda x: -x["count"])

    return {
        "k": int(k),
        "n_docs": n_docs,
        "n_chunks_used": n_chunks,
        "auto_k_scores": scores,  # para depuración/visualización si quieres
        "topics": topics,  # [{cluster, label, count, examples}]
        "doc_topic_distribution": doc_topic_distribution,  # mezcla por documento
        "top_docs_per_topic": top_docs_per_topic,  # ranking de docs por tópico
    }
