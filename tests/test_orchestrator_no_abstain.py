# tests/test_orchestrator_no_abstain.py
from backend.core.orchestrator import run_chat

def test_run_chat_no_abstain():
    def fake_retrieve(coll, q, k):
        return [{
            "text": "... Isabel Mercado ...",
            "meta": {"doc_title":"Doc.pdf","page":1},
            "id": "X",
            "score": 0.7
        }]

    def fake_complete(system, user):
        # simula modelo que a veces no cita; el orquestador reintenta
        return "Isabel Mercado es recomendada. (Doc:Doc.pdf, pág 1)"

    out = run_chat("¿Quién es Isabel Mercado?", "default", fake_retrieve, fake_complete)
    assert out["abstained"] is False
    assert "(Doc:" in out["answer"]
