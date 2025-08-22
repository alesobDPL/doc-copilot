# backend/core/vstore_mini.py
from io import BytesIO
import os, json, tempfile
import numpy as np

DATA_DIR = os.path.join(".", "data", "mini_store")
os.makedirs(DATA_DIR, exist_ok=True)

def _paths(name):
    base = os.path.join(DATA_DIR, name)
    return (base + "_vecs.npy", base + "_meta.json", base + "_docs.json", base + "_ids.json")

def _atomic_write(path, data_bytes: bytes):
    # escribe a tmp y renombra (mejora contra corrupción por concurrencia)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".swap_", suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data_bytes)
    os.replace(tmp, path)

def upsert_chunks(collection_name, chunks, metas, ids, embed_fn):
    vecs_path, meta_path, docs_path, ids_path = _paths(collection_name)
    new_embs = np.asarray(embed_fn(chunks), dtype=np.float32)

    if os.path.exists(vecs_path):
        old = np.load(vecs_path, mmap_mode="r")
        embs = np.concatenate([np.asarray(old), new_embs], axis=0)
        with open(meta_path) as f: old_meta = json.load(f)
        with open(docs_path) as f: old_docs = json.load(f)
        with open(ids_path) as f:  old_ids  = json.load(f)
        metas = old_meta + metas
        chunks = old_docs + chunks
        ids    = old_ids + ids
    else:
        embs = new_embs

    # np.save no es atómico → usamos buffer en memoria + _atomic_write
    buf = BytesIO()
    np.save(buf, embs)
    _atomic_write(vecs_path, buf.getvalue())
    _atomic_write(meta_path, json.dumps(metas, ensure_ascii=False).encode("utf-8"))
    _atomic_write(docs_path, json.dumps(chunks, ensure_ascii=False).encode("utf-8"))
    _atomic_write(ids_path,  json.dumps(ids, ensure_ascii=False).encode("utf-8"))

def query(collection_name, query_texts, embed_fn, k=6, where=None):
    vecs_path, meta_path, docs_path, ids_path = _paths(collection_name)
    if not os.path.exists(vecs_path):
        return {"documents":[[]], "metadatas":[[]], "ids":[[]]}

    embs = np.load(vecs_path, mmap_mode="r")  # (N,D) sin cargar todo a RAM
    with open(meta_path) as f: metas = json.load(f)
    with open(docs_path) as f: docs  = json.load(f)
    with open(ids_path)  as f: ids   = json.load(f)

    # filtro 'where' simple (p.ej. {"doc_title": "X"})
    mask = np.ones(len(docs), dtype=bool)
    if where and "doc_title" in where:
        target = where["doc_title"]
        mask = np.array([(m.get("doc_title")==target) for m in metas], dtype=bool)
    idx_all = np.where(mask)[0]
    if idx_all.size == 0:
        return {"documents":[[]], "metadatas":[[]], "ids":[[]]}

    A = np.asarray(embed_fn(query_texts), dtype=np.float32)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = np.asarray(embs[idx_all])
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    sims = A @ B.T
    topk_local = np.argsort(-sims, axis=1)[:, :k]

    out_docs, out_meta, out_ids = [], [], []
    for r in range(topk_local.shape[0]):
        loc = topk_local[r]
        sel = idx_all[loc]
        out_docs.append([docs[i] for i in sel])
        out_meta.append([metas[i] for i in sel])
        out_ids.append([ids[i] for i in sel])
    return {"documents": out_docs, "metadatas": out_meta, "ids": out_ids}

def load_collection(collection_name):
    vecs_path, meta_path, docs_path, ids_path = _paths(collection_name)
    if not os.path.exists(vecs_path):
        return [], [], []
    with open(meta_path) as f: metas = json.load(f)
    with open(docs_path) as f: docs  = json.load(f)
    with open(ids_path)  as f: ids   = json.load(f)
    return docs, metas, ids

def clear_collection(collection_name):
    for p in _paths(collection_name):
        try:
            if os.path.exists(p): os.remove(p)
        except: pass
