# backend/core/vstore_chroma.py
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.join(".", "data", "chroma")
os.makedirs(BASE_DIR, exist_ok=True)

# Un ÚNICO cliente persistente para todo el proyecto
_client = chromadb.PersistentClient(
    path=BASE_DIR, settings=Settings(allow_reset=True)  # opcional, útil en dev
)

COLLECTION_NAME = "docs"  # usamos 1 collection por “colección lógica” de tu app


def _get_or_create_collection(collection_name: str):
    """
    Mantenemos un 'namespace' por colección de tu app usando
    un nombre compuesto: p.ej., docs__<collection_name>
    """
    name = f"{COLLECTION_NAME}__{collection_name}"
    # Intenta obtener; si no existe, crea con metadatos y espacio coseno
    try:
        return _client.get_collection(name=name)
    except Exception:
        return _client.create_collection(name=name, metadata={"space": "cosine"})


def _chunked(iterable, size: int):
    for i in range(0, len(iterable), size):
        yield i, iterable[i : i + size]


def upsert_chunks(
    collection_name: str,
    chunks: List[str],
    metas: List[Dict[str, Any]],
    ids: List[str],
    embed_fn,
):
    col = _get_or_create_collection(collection_name)
    # Obtén embeddings en lote
    vecs = embed_fn(chunks)

    # Algunas versiones de Chroma ya traen 'upsert'; si no, hacemos delete+add
    has_upsert = hasattr(col, "upsert")

    BATCH = 256
    for i, docs_batch in _chunked(chunks, BATCH):
        ids_batch = ids[i : i + len(docs_batch)]
        metas_batch = metas[i : i + len(docs_batch)]
        embs_batch = vecs[i : i + len(docs_batch)]

        if has_upsert:
            col.upsert(
                ids=ids_batch,
                documents=docs_batch,
                metadatas=metas_batch,
                embeddings=embs_batch,
            )
        else:
            # Borrado “best-effort” (si no existían, no falla) y luego add
            try:
                col.delete(ids=ids_batch)
            except Exception:
                pass
            col.add(
                ids=ids_batch,
                documents=docs_batch,
                metadatas=metas_batch,
                embeddings=embs_batch,
            )


def query(
    collection_name: str,
    query_texts: List[str],
    embed_fn,
    k: int = 6,
    where: Optional[Dict[str, Any]] = None,
):
    col = _get_or_create_collection(collection_name)
    q_embs = embed_fn(query_texts)


    res = col.query(
    query_embeddings=q_embs,
    n_results=k,
    where=where,
    include=["documents", "metadatas", "distances"],  # <-- sin "ids"
)
    documents = res.get("documents", [[]])
    metadatas = res.get("metadatas", [[]])
    ids = res.get("ids", [[]])  # <-- Chroma los trae de todas formas
    distances = res.get("distances", [[]])
    scores = [[1.0 - d for d in row] for row in distances] if distances else None
    return {"documents": documents, "metadatas": metadatas, "ids": ids, "scores": scores}


def load_collection(collection_name: str):
    col = _get_or_create_collection(collection_name)
    # 'ids' no se pasa en include (no es válido). Igual viene en la respuesta.
    got = col.get(include=["documents", "metadatas"])
    docs = got.get("documents", []) or []
    metas = got.get("metadatas", []) or []
    ids = got.get("ids", []) or []  # viene aunque no lo pidas en include
    return docs, metas, ids


def clear_collection(collection_name: str):
    name = f"{COLLECTION_NAME}__{collection_name}"
    try:
        _client.delete_collection(name)
    except Exception:
        # Si no existe o ya está borrada, ignoramos
        pass
