# tests/test_chat_endpoint_happy.py
from fastapi.testclient import TestClient
from backend.main import app
import backend.core.orchestrator as orch

client = TestClient(app)

def test_chat_endpoint_happy(monkeypatch):
    def fake_run_chat(q, collection, retrieve_fn, llm_complete):
        return {
            "answer": "Hola (Doc:Doc.pdf, pág 1)",
            "abstained": False,
            "intent": "answer",
            "chunks_used": [{"doc":"Doc.pdf","page":1,"score":0.8}],
            "metrics": {}
        }
    monkeypatch.setattr(orch, "run_chat", fake_run_chat)
    r = client.post("/chat", json={"collection":"default", "question":"hola"})
    assert r.status_code == 200
    data = r.json()
    assert "(Doc:" in data["answer"]
    assert data["abstained"] is False
