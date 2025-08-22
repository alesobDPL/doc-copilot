# tests/conftest.py
import sys, pathlib, os, pytest

# 1) Path fix
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2) Vars de entorno por defecto (ajústalas a tu gusto)
@pytest.fixture(autouse=True)
def _env_defaults(monkeypatch):
    monkeypatch.setenv("FAKE_LLM", "1")
    monkeypatch.setenv("VSTORE_BACKEND", "chroma")
    monkeypatch.setenv("FORCE_CITATIONS", "1")
    monkeypatch.setenv("MIN_SCORE", "0.15")
    monkeypatch.setenv("MIN_MEAN_TOP", "0.25")
    monkeypatch.setenv("MIN_COVERAGE", "2")
    monkeypatch.setenv("MIN_DIVERSITY", "2")
    monkeypatch.setenv("RAG_TOPK", "12")
    monkeypatch.setenv("RERANK_TOPK", "6")
    monkeypatch.setenv("MAX_CTX_CHARS", "9000")
    yield


@pytest.fixture
def retrieve_ok():
    def _fn(coll, q, k=12):
        # mezcla de docs, scores razonables (cumple coverage y mean_top)
        return [
            {"text":"A", "meta":{"doc_title":"Doc A","page":1}, "score":0.36},
            {"text":"B", "meta":{"doc_title":"Doc B","page":5}, "score":0.31},
            {"text":"C", "meta":{"doc_title":"Doc A","page":2}, "score":0.26},
            {"text":"D", "meta":{"doc_title":"Doc C","page":1}, "score":0.18},
        ]
    return _fn

@pytest.fixture
def retrieve_bad():
    def _fn(coll, q, k=12):
        # scores bajos, cobertura insuficiente
        return [
            {"text":"X", "meta":{"doc_title":"Doc A","page":1}, "score":0.06},
            {"text":"Y", "meta":{"doc_title":"Doc A","page":2}, "score":0.08},
        ]
    return _fn

@pytest.fixture
def llm_always_cites():
    def _fn(system, user):
        return "Respuesta con cita (Doc: Doc A, pág:1)."
    return _fn