# tests/test_chat_endpoint_happy.py
from backend.core.orchestrator import run_chat
import backend.core.orchestrator as orch

def test_guardrails_abstain(monkeypatch, retrieve_bad, llm_always_cites):
    # Plan estrictísimo para forzar abstención
    def strict_plan(intent, q):
        return {
 "k": 12,
        "rerank_top_k": 4,
        "min_score": 0.99,
        "min_mean_top": 0.99,
        "min_coverage": 2,
        "min_diversity": 2,
        "force_citations": True,
        "max_ctx_chars": 9000,
        "allow_weak_fallback": False, 
        }
    monkeypatch.setattr(orch, "build_retrieval_plan", strict_plan)

    out = run_chat("¿Qué dice?", "default", retrieve_bad, llm_always_cites)
    assert out["abstained"] is True, f"metrics={out.get('metrics')}"
