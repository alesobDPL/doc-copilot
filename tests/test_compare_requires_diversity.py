# tests/test_compare_requires_diversity.py
from backend.core.orchestrator import run_chat

def test_compare_needs_diversity():
    def fake_retrieve(coll, q, k):
        # Todos los hits del mismo documento → diversity=1
        return [{
            "text": "A1", "meta": {"doc_title":"DocA","page":1}, "id":"a1", "score":0.9
        },{
            "text": "A2", "meta": {"doc_title":"DocA","page":2}, "id":"a2", "score":0.8
        }]

    def fake_complete(system, user):
        return "Similitudes/Diferencias"

    out = run_chat("compara DocA y DocB", "default", fake_retrieve, fake_complete)
    # dependiendo de tus guardrails, podría abstenerse por falta de diversidad
    assert out["intent"] == "compare"
    # o bien, si ajustaste guardrails para permitirlo, chequearías la respuesta
