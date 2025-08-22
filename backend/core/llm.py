# backend/core/llm.py
import os
import hashlib
import random
from typing import List

from dotenv import load_dotenv

load_dotenv()

USE_FAKE = os.getenv("FAKE_LLM", "0") == "1"

# ────────────────────────────────────────────────────────────────────────────────
# MODO FAKE (para tests/CI): no llama a servicios externos
# ────────────────────────────────────────────────────────────────────────────────
if USE_FAKE:

    def _vector_from_text(t: str, dim: int = 1536) -> List[float]:
        """
        Genera un embedding determinista a partir de un hash del texto.
        No representa semántica real, pero sirve para tests reproducibles.
        """
        h = hashlib.sha1((t or "").encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big", signed=False)
        rng = random.Random(seed)
        # valores en [-1, 1] normalizados un poco para estabilidad
        vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        # normalizamos para evitar magnitudes raras
        norm = (sum(v * v for v in vec) ** 0.5) or 1.0
        return [v / norm for v in vec]

    def embed(texts, model: str = "fake-embedding-1536"):
        if isinstance(texts, str):
            texts = [texts]
        return [_vector_from_text(t) for t in texts]

    def chat(messages, model: str = "fake-chat", temperature: float = 0.0) -> str:
        """
        Devuelve una respuesta fake, pero:
        - Si el prompt trae evidencia con "(Doc:" intentamos incluir al menos una cita.
        - Útil para pasar tests de citas/guardrails sin depender del proveedor.
        """
        # reconstruimos el "user" para buscar patrones
        user = ""
        for m in messages or []:
            if m.get("role") == "user":
                user += (m.get("content") or "") + "\n"

        cite = "(Doc:Doc A, pág:1)"
        # Si en el prompt vienen citas reales, preferimos una de ellas
        # para que los tests de “citas forzadas” las detecten.
        import re
        m = re.search(r"\(Doc:[^)]+pág[: ]\s*\d+\)", user)
        if m:
            cite = m.group(0)

        return f"Respuesta (FAKE) con cita {cite}."

# ────────────────────────────────────────────────────────────────────────────────
# MODO REAL: usa OpenAI
# ────────────────────────────────────────────────────────────────────────────────
else:
    from openai import OpenAI

    # Si no hay API key, este módulo no debe romper al importar.
    # Dejamos que falle en tiempo de llamada con un error claro.
    _api_key = os.getenv("OPENAI_API_KEY", "")
    client = OpenAI(api_key=_api_key or None)

    def embed(texts, model: str = "text-embedding-3-small"):
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def chat(messages, model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        ).choices[0].message.content
