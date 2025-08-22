# backend/core/orchestrator.py
import os, time, numpy as np
from typing import List, Dict, Any, Callable, Literal

Intent = Literal["answer", "summarize", "compare", "classify"]


def detect_intent(q: str) -> Intent:
    ql = (q or "").lower()
    if any(k in ql for k in ["resumen", "summary", "sintetiza"]):
        return "summarize"
    if any(k in ql for k in ["compara", "versus", "diferenc"]):
        return "compare"
    if any(k in ql for k in ["tema", "tópico", "clasific"]):
        return "classify"
    return "answer"


def _envfloat(k: str, d: float) -> float:
    try:
        return float(os.getenv(k, d))
    except:
        return d


def _envi(k: str, d: int) -> int:
    try:
        return int(os.getenv(k, d))
    except:
        return d


def build_retrieval_plan(intent: Intent, q: str) -> Dict[str, Any]:
    """
    Defaults deliberadamente permisivos. Puedes endurecerlos por .env:
      MIN_SCORE, MIN_MEAN_TOP, MIN_COVERAGE, MIN_DIVERSITY, RAG_TOPK, RERANK_TOPK, MAX_CTX_CHARS, FORCE_CITATIONS
    """
    return {
        "k": _envi("RAG_TOPK", 12),
        "rerank_top_k": _envi("RERANK_TOPK", 6),
        "min_score": _envfloat("MIN_SCORE", -1.0),
        "min_mean_top": _envfloat("MIN_MEAN_TOP", -1.0),
        "min_coverage": _envi("MIN_COVERAGE", 1),
        "min_diversity": _envi("MIN_DIVERSITY", 2),
        "force_citations": os.getenv("FORCE_CITATIONS", "1") == "1",
        "max_ctx_chars": _envi("MAX_CTX_CHARS", 9000),
        "allow_weak_fallback": os.getenv("ALLOW_WEAK_FALLBACK", "1") == "1",
    }
    


def rerank(cands: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    return sorted(cands, key=lambda x: x.get("score", 0.0), reverse=True)[
        : max(1, top_k)
    ]


def _confidence(scores: List[float], min_score: float) -> Dict[str, Any]:
    if not scores:
        return {"mean_top": 0.0, "coverage": 0, "gap": 0.0}
    s = sorted(scores, reverse=True)
    mean_top = np.mean(s[:3]) if len(s) >= 3 else np.mean(s)
    gap = (s[0] - s[3]) if len(s) >= 4 else (s[0] - s[-1])
    coverage = sum(1 for x in s if x >= min_score)
    return {"mean_top": float(mean_top), "coverage": int(coverage), "gap": float(gap)}


def guardrails(
    q: str, intent: Intent, top: List[Dict[str, Any]], plan: Dict[str, Any]
) -> Dict[str, Any]:
    scores = [c.get("score", 0.0) for c in top]
    conf = _confidence(scores, plan["min_score"])  # ← usa plan, NO env
    diversity = len(set((c.get("meta", {}) or {}).get("doc_title") for c in top))
    need_diversity = (
        ("compara" in q.lower()) or ("versus" in q.lower()) or intent == "compare"
    )
    ok = (
        conf["mean_top"] >= plan["min_mean_top"]
        and conf["coverage"] >= plan["min_coverage"]
        and (diversity >= plan["min_diversity"] if need_diversity else True)
    )
    return {
        "ok": bool(ok),
        "confidence": conf,
        "diversity": diversity,
        "need_diversity": need_diversity,
    }


def _format_evidence(ctxs: List[Dict[str, Any]], max_chars: int) -> str:
    parts, used = [], 0
    for c in ctxs:
        meta = c.get("meta", {}) or {}
        doc = meta.get("doc_title")
        page = meta.get("page")
        txt = (c.get("text") or "").strip().replace("\n", " ")
        snippet = txt[:220]
        line = f"- (Doc:{doc}, pág:{page}) {snippet}..."
        if used + len(line) > max_chars:
            break
        parts.append(line)
        used += len(line)
    return "\n".join(parts)


def synthesize(
    intent: Intent,
    q: str,
    ctxs: List[Dict[str, Any]],
    llm_complete: Callable[[str, str], str],
    plan: Dict[str, Any],
) -> str:
    system = (
        "Eres un asistente sobre PDFs. Responde con precisión y SIN inventar.\n"
        "Siempre cita fragmentos como (Doc:{doc_title}, pág {page}). "
        "Si el contexto es breve, responde con lo disponible y dilo explícitamente."
    )
    evidence = _format_evidence(ctxs, plan["max_ctx_chars"])
    user = f"[INTENT:{intent}]\nPregunta: {q}\n\nEVIDENCIA:\n{evidence}\n\nResponde claro y con citas."
    out = (llm_complete(system, user) or "").strip()
    if plan["force_citations"] and "(Doc:" not in out:
        user2 = (
            user
            + "\n\nIMPORTANTE: Incluye al menos una cita (Doc:{doc_title}, pág {page})."
        )
        out = (llm_complete(system, user2) or "").strip()
    return out


def run_chat(
    q: str,
    collection: str,
    retrieve_fn: Callable[[str, str, int], List[Dict[str, Any]]],
    llm_complete: Callable[[str, str], str],
) -> Dict[str, Any]:
    t0 = time.time()
    intent = detect_intent(q)
    plan = build_retrieval_plan(intent, q)

    # 1) retrieve
    t1 = time.time()
    cands = retrieve_fn(collection, q, plan["k"]) or []
    t2 = time.time()

    # 2) rerank
    top = rerank(cands, plan["rerank_top_k"])
    t3 = time.time()

    # 3) guardrails (no bloquea si NO es compare y hay ≥1 evidencia)
    gr = guardrails(q, intent, top, plan)
    if not gr["ok"]:
        # Fallback: si hay CUALQUIER evidencia, responde igual
        if plan["allow_weak_fallback"] and len(top) >= 1:
            ans = synthesize(intent, q, top, llm_complete, plan)
            t4 = time.time()
            return {
                "answer": ans,
                "intent": intent,
                "abstained": False,
                "chunks_used": [
                    {
                        "doc": (c.get("meta", {}) or {}).get("doc_title"),
                        "page": (c.get("meta", {}) or {}).get("page"),
                        "score": round(float(c.get("score", 0.0)), 4),
                    }
                    for c in top
                ],
                "metrics": {
                    "t_retrieve_ms": int((t2 - t1) * 1000),
                    "t_rerank_ms": int((t3 - t2) * 1000),
                    "t_llm_ms": int((t4 - t3) * 1000),
                    "t_total_ms": int((t4 - t0) * 1000),
                    **gr,
                },
            }
            
        # Real abstención: no hay nada
        return {
            "answer": "No encuentro evidencia suficiente en los PDFs para responder con certeza. "
            "¿Puedes precisar la pregunta o subir documentos más relevantes?",
            "intent": intent,
            "abstained": True,
            "chunks_used": [],
            "metrics": {
                "t_retrieve_ms": int((t2 - t1) * 1000),
                "t_rerank_ms": int((t3 - t2) * 1000),
                "t_total_ms": int((time.time() - t0) * 1000),
                **gr,
                "plan": {
                    k: plan[k]
                    for k in [
                        "min_score",
                        "min_mean_top",
                        "min_coverage",
                        "min_diversity",
                    ]
                },
            },
        }

    # 4) síntesis normal
    ans = synthesize(intent, q, top, llm_complete, plan)
    t4 = time.time()
    return {
        "answer": ans,
        "intent": intent,
        "abstained": False,
        "chunks_used": [
            {
                "doc": (c.get("meta", {}) or {}).get("doc_title"),
                "page": (c.get("meta", {}) or {}).get("page"),
                "score": round(float(c.get("score", 0.0)), 4),
            }
            for c in top
        ],
        "metrics": {
            "t_retrieve_ms": int((t2 - t1) * 1000),
            "t_rerank_ms": int((t3 - t2) * 1000),
            "t_llm_ms": int((t4 - t3) * 1000),
            "t_total_ms": int((t4 - t0) * 1000),
            **gr,
        },
    }
