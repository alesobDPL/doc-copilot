# tests/test_rag_keyword_boost.py
from backend.core.rag import retrieve
from backend.core import vectorstore as vs

def fake_embed(x):  # no se usa en esta prueba porque devolvemos distances directas
    return [[0.1]*5 for _ in x]

def test_retrieve_keyword_boost(monkeypatch):
    # Colección simulada: 2 chunks
    docs = ["... apoyo a Isabel Mercado ...", "otro chunk sin el nombre"]
    metas = [
        {"doc_title":"Doc.pdf","page":1,"chunk_index":0,"doc_id":"A"},
        {"doc_title":"Otro.pdf","page":1,"chunk_index":0,"doc_id":"B"},
    ]
    ids = ["A|p0001|c0000|xxx", "B|p0001|c0000|yyy"]

    def fake_load_collection(coll):
        return docs, metas, ids

    # La query por embeddings devuelve solo el primer chunk, con distancia ≈1.06 (sim ≈ -0.06)
    def fake_query(coll, q, embed_fn, k=8, where=None):
        return {
            "documents": [[docs[0]]],
            "metadatas": [[metas[0]]],
            "ids": [[ids[0]]],
            "distances": [[1.06]],  # Chroma cosine distance
        }

    monkeypatch.setattr(vs, "load_collection", fake_load_collection)
    monkeypatch.setattr(vs, "query", fake_query)

    hits = retrieve("default", "¿Quién es Isabel Mercado?", fake_embed, k=4)
    assert len(hits) >= 1
    # por boost léxico, debe subir el score
    assert hits[0]["score"] >= 0.7
    assert "Isabel" in hits[0]["text"]
