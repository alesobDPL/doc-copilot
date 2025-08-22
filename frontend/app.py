# frontend/app.py
import streamlit as st, requests, json
import os

st.set_page_config(page_title="Copiloto sobre PDFs", layout="centered")
st.title("Copiloto Conversacional sobre PDFs")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Estado de la conversación
if "chat" not in st.session_state:
    st.session_state.chat = []  # lista de tuplas (role, msg)

# Panel de colección
collection = st.text_input(
    "Nombre de colección",
    value="default",
    help="Usa el mismo nombre en /ingest y /chat",
)

# Uploader e indexación
st.subheader("Subir PDFs")

with st.form("ingest_form", clear_on_submit=False):
    files = st.file_uploader(
        "Sube hasta 5 PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Selecciona hasta 5 archivos; se enviarán todos juntos al presionar 'Indexar'.",
    )
    submitted = st.form_submit_button("Subir")

if submitted:
    if not collection.strip():
        st.warning("Debes escribir un nombre de colección.")
    elif not files:
        st.warning("Debes subir al menos un PDF.")
    else:
        payload = [
            ("files", (f.name, f.getvalue(), "application/pdf")) for f in files[:5]
        ]
        with st.spinner(f"Indexando {len(payload)} PDF(s)..."):
            try:
                r = requests.post(
                    f"{BACKEND_URL}/ingest",
                    files=payload,
                    data={"collection": collection},
                )
                if r.ok:
                    resp = r.json()
                    st.success(resp)

                    # 🔽🔽 NUEVO: pedir la lista oficial y mostrarla
                    try:
                        rr = requests.get(
                            f"{BACKEND_URL}/collections/{collection}/docs"
                        )
                        if rr.ok:
                            data = rr.json()
                            docs = data.get("docs", [])
                            st.session_state["docs_list"] = docs  # guardo en sesión
                            with st.expander(
                                "Documentos indexados en la colección", expanded=True
                            ):
                                if docs:
                                    st.markdown("\n".join([f"- {d}" for d in docs]))
                                else:
                                    st.markdown("_No hay documentos en la colección._")
                        else:
                            st.warning(
                                f"No se pudo obtener la lista de documentos: {rr.status_code} {rr.text}"
                            )
                    except Exception as e:
                        st.warning(f"No se pudo obtener la lista de documentos: {e}")

                else:
                    st.error(f"Error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")
st.divider()
st.subheader("Colección")

col1, col2 = st.columns(2)

with col1:
    if st.button("Refrescar lista de documentos"):
        try:
            rr = requests.get(f"{BACKEND_URL}/collections/{collection}/docs")
            if rr.ok:
                data = rr.json()
                st.session_state["docs_list"] = data.get("docs", [])
            else:
                st.error(f"Error {rr.status_code}: {rr.text}")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    if st.button("Limpiar colección (borrar todo)"):
        try:
            dr = requests.delete(f"{BACKEND_URL}/collections/{collection}")
            if dr.ok:
                st.success("Colección vaciada.")
                st.session_state["docs_list"] = []
            else:
                st.error(f"Error {dr.status_code}: {dr.text}")
        except Exception as e:
            st.error(f"Error: {e}")

# Muestra la lista si existe en sesión
if "docs_list" in st.session_state:
    docs = st.session_state["docs_list"]
    with st.expander("Documentos en la colección (oficial)", expanded=False):
        if docs:
            st.markdown("\n".join([f"- {d}" for d in docs]))
        else:
            st.markdown("_(Vacío)_")

            st.divider()
st.subheader("Acciones sobre la colección")

# Asegura docs_list en sesión (puede venir de refresh manual)
if "docs_list" not in st.session_state:
    try:
        rr = requests.get(f"{BACKEND_URL}/collections/{collection}/docs")
        if rr.ok:
            st.session_state["docs_list"] = rr.json().get("docs", [])
    except:
        st.session_state["docs_list"] = []

docs_list = st.session_state.get("docs_list", [])

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Resumen",
        "Comparar documentos par",
        "Comparar documentos global",
        "Clasificación por temas",
    ]
)

# --- RESUMEN ---
with tab1:
    st.write("Genera un resumen del documento seleccionado o de toda la colección.")
    target = st.selectbox(
        "Seleccionar un documento",
        options=["<Toda la colección>"] + docs_list,
        index=0,
    )

    include_overview = st.checkbox("Incluir overview ejecutivo (colección)", value=True)

    adv = st.expander("Modo avanzado (opcional)", expanded=False)
    with adv:
        use_manual = st.checkbox("Forzar máx. chunks por doc", value=False)
        manual_k = st.slider(
            "Máx. chunks por doc (manual)",
            5,
            30,
            12,
            1,
            help="Sólo se aplica si activas 'Forzar máx. chunks por doc'.",
        )

    if st.button("Generar resumen"):
        payload = {"collection": collection}
        if target != "<Toda la colección>":
            payload["doc_title"] = target
        else:
            payload["include_overview"] = include_overview
        if use_manual:
            payload["max_per_doc"] = manual_k

        with st.spinner("Resumiendo..."):
            try:
                r = requests.post(f"{BACKEND_URL}/summary", json=payload, timeout=120)
            except Exception as e:
                st.error(f"Error de conexión: {e}")
            else:
                # 1) Si el status no es OK, intenta mostrar JSON de error o texto crudo
                if not r.ok:
                    # intenta parsear json de error
                    try:
                        err = r.json()
                    except Exception:
                        err = r.text
                    st.error(f"Error {r.status_code}")
                    st.code(err)
                else:
                    # 2) Respuesta OK: parseo robusto
                    try:
                        data = r.json()
                    except Exception:
                        data = None

                    if not isinstance(data, dict):
                        st.warning(
                            "Respuesta no-JSON del backend. Mostrando cuerpo crudo:"
                        )
                        st.code(r.text)
                    else:
                        mode = data.get("mode")

                        # --- Modo: un documento ---
                        if mode == "single" or (
                            "doc_title" in payload and "summary" in data
                        ):
                            st.success("Resumen generado")
                            st.markdown(f"### {data.get('doc_title', target)}")
                            st.markdown(data.get("summary", "_(sin texto)_"))
                            st.caption(
                                f"Chunks usados: {data.get('chunks_used', 'N/D')}"
                            )

                        # --- Modo: colección (resúmenes por documento) ---
                        elif mode == "collection_per_doc":
                            st.success("Resumen de la colección generado")
                            if data.get("overview"):
                                st.markdown("### Overview ejecutivo")
                                st.markdown(data["overview"])
                                st.divider()
                            st.markdown("### Resúmenes por documento")
                            for item in data.get("per_doc", []):
                                with st.expander(
                                    item.get("doc_title", "Documento"), expanded=False
                                ):
                                    st.markdown(item.get("summary", "_(sin texto)_"))

                        # --- Compatibilidad con respuestas antiguas ---
                        elif "global_summary" in data or "per_doc" in data:
                            if data.get("global_summary"):
                                st.markdown("### Resumen global")
                                st.markdown(data["global_summary"])
                                st.divider()
                            if data.get("per_doc"):
                                st.markdown("### Resúmenes por documento")
                                for t, s in data["per_doc"].items():
                                    with st.expander(t, expanded=False):
                                        st.markdown(s)
                            else:
                                st.info("No se recibieron resúmenes por documento.")

                        else:
                            st.warning(
                                "Formato de respuesta no reconocido. Mostrando JSON crudo:"
                            )
                            st.code(json.dumps(data, ensure_ascii=False, indent=2))

# --- COMPARAR ---
with tab2:
    st.write("Compara dos documentos de la colección.")
    if len(docs_list) < 2:
        st.info("Sube/selecciona al menos 2 documentos para comparar.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            doc_a = st.selectbox("Documento A", options=docs_list, key="cmp_a")
        with c2:
            doc_b = st.selectbox(
                "Documento B",
                options=[d for d in docs_list if d != st.session_state.get("cmp_a")],
                key="cmp_b",
            )
        if st.button("Comparar"):
            with st.spinner("Comparando..."):
                r = requests.post(
                    f"{BACKEND_URL}/compare",
                    json={"collection": collection, "doc_a": doc_a, "doc_b": doc_b},
                )
                if r.ok:
                    st.success("Comparación generada")
                    st.markdown(r.json().get("comparison", "_(sin texto)_"))
                else:
                    st.error(f"Error {r.status_code}: {r.text}")

# --- COMPARACIÓN GLOBAL ---
with tab3:
    st.write("Comparación global automática entre documentos de la colección.")

    if not docs_list or len(docs_list) < 2:
        st.info("Necesitas al menos 2 documentos en la colección.")
    else:
        if st.button("Analizar colección"):
            with st.spinner("Analizando relaciones entre documentos..."):
                try:
                    r = requests.post(
                        f"{BACKEND_URL}/compare/global",
                        json={"collection": collection},
                        timeout=120,
                    )
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
                else:
                    if not r.ok:
                        try:
                            err = r.json()
                        except Exception:
                            err = r.text
                        st.error(f"Error {r.status_code}")
                        st.code(err)
                    else:
                        try:
                            data = r.json()
                        except Exception:
                            data = None

                        if not isinstance(data, dict):
                            st.warning(
                                "Respuesta no-JSON del backend. Mostrando cuerpo crudo:"
                            )
                            st.code(r.text)
                        else:
                            docs = data.get("docs", [])
                            sim_mat = data.get("similarity_matrix", [])
                            auto_thr = data.get("auto_threshold")
                            explain = data.get("explain_threshold", "")

                            if not docs or not sim_mat:
                                st.info(
                                    "No se pudo calcular la similitud (colección muy pequeña o sin texto)."
                                )
                            else:
                                # Mapa de similitud
                                try:
                                    st.caption(
                                        "Interpretación: 1.0 = textos prácticamente idénticos; ~0.7 fuerte similitud; ~0.4 relación moderada; <0.2 baja similitud."
                                    )
                                    import pandas as pd
                                    import numpy as np

                                    df = pd.DataFrame(sim_mat, index=docs, columns=docs)

                                    # diagonal a 1.0 (auto-similitud), si no viniera ya
                                    for i in range(len(docs)):
                                        df.iat[i, i] = 1.0

                                    st.markdown(
                                        "### Mapa de similitud entre documentos"
                                    )
                                    st.caption(
                                        "La similitud coseno va de 0 (sin relación) a 1 (idénticos). La diagonal es 1.0 por definición."
                                    )
                                    st.dataframe(
                                        df.style.format("{:.3f}")
                                        .background_gradient(cmap="Greens")
                                        .set_caption(
                                            f"Umbral automático ≈ {auto_thr:.2f}"
                                            if isinstance(auto_thr, (int, float))
                                            else "Umbral automático no disponible"
                                        ),
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.warning(
                                        f"No se pudo renderizar la tabla (mostrando crudo): {e}"
                                    )
                                    st.text(sim_mat)

                                # Texto de metodología/umbral
                                if explain:
                                    st.info(explain)

                            # Pairs y grupos (si existen)
                            most_sim = data.get("most_similar_by_doc", {})
                            groups = data.get("groups", [])
                            pairs = data.get("top_pairs", [])

                            st.markdown("### Pareja más similar por documento")
                            if not docs or not most_sim:
                                st.write("- N/D")
                            else:
                                for d in docs:
                                    info = most_sim.get(d, {})
                                    st.write(
                                        f"- **{d}** ↔ {info.get('doc','N/D')} (sim={info.get('sim','N/D')})"
                                    )

                            st.markdown("### Grupos de documentos relacionados")
                            has_groups = False
                            for i, g in enumerate(groups or [], 1):
                                if len(g) > 1:
                                    has_groups = True
                                    st.write(f"- Grupo {i}: {', '.join(g)}")
                            if not has_groups:
                                st.info(
                                    "No se detectaron grupos significativos con el umbral automático."
                                )

                            st.markdown("### Comparaciones más relevantes (auto)")
                            if not pairs:
                                st.info(
                                    "No hubo pares suficientemente similares (colección muy dispar o muy pequeña)."
                                )
                            else:
                                for p in pairs[:10]:
                                    with st.expander(
                                        f"{p['a']}  ↔  {p['b']}   ·   sim={p['sim']}",
                                        expanded=False,
                                    ):
                                        st.markdown(
                                            p.get("comparison", "_(sin texto)_")
                                        )

# --- CLASIFICACIÓN ---
with tab4:
    st.write(
        "Agrupa los chunks por tópicos (k automático) y muestra mezcla por documento."
    )
    if st.button("Clasificar"):
        with st.spinner("Clasificando..."):
            r = requests.post(
                f"{BACKEND_URL}/classify", json={"collection": collection}
            )
            if r.ok:
                out = r.json()
                st.success(
                    f"Tópicos: {out.get('k')} (chunks usados: {out.get('n_chunks_used')})"
                )

                st.markdown("### Tópicos")
                for t in out.get("topics", []):
                    with st.expander(f"{t['label']} · {t['count']} chunks"):
                        st.json(t.get("examples", []))

                st.markdown("### Mezcla de tópicos por documento")
                dist = out.get("doc_topic_distribution", {})
                for doc, d in dist.items():
                    st.markdown(f"**{doc}**")
                    # d es {topic_label: proporción}
                    pretty = "\n".join([f"- {k}: {v*100:.1f}%" for k, v in d.items()])
                    st.markdown(pretty)

                st.markdown("### Top documentos por tópico")
                topd = out.get("top_docs_per_topic", {})
                for topic, ranking in topd.items():
                    pretty = "\n".join([f"- {doc}: {p*100:.1f}%" for doc, p in ranking])
                    with st.expander(topic, expanded=False):
                        st.markdown(pretty)

                # (opcional) mostrar auto_k_scores para depurar
                # st.code(json.dumps(out.get("auto_k_scores", []), indent=2))
            else:
                st.error(f"Error {r.status_code}: {r.text}")


# Render del chat existente
st.subheader("Chat")
for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)


# Entrada de chat
q = st.chat_input("Escribe tu pregunta…")
if q:
    st.session_state.chat.append(("user", q))
    # Se hace la llamada y mostramos TODO, incluso errores
    with st.spinner("Consultando..."):
        try:
            resp = requests.post(
                f"{BACKEND_URL}/chat",
                json={"collection": collection.strip(), "question": q},
            )
            if resp.ok:
                data = resp.json()
                ans = data.get("answer")
                if not ans:
                    ans = f"(Backend respondió sin 'answer')\n\nRespuesta completa:\n```\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```"
                st.session_state.chat.append(("assistant", ans))
            else:
                st.session_state.chat.append(
                    ("assistant", f"Error {resp.status_code}:\n```\n{resp.text}\n```")
                )
        except Exception as e:
            st.session_state.chat.append(
                ("assistant", f"Error de conexión:\n```\n{e}\n```")
            )

    # re-render de los mensajes.
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)


# --- DEBUG (desactivado) ---
# with st.expander("Debug", expanded=False):
#     if st.button("Probar recuperación con la última pregunta"):
#         if not q:
#             st.info("Primero envía una pregunta en el chat.")
#     else:
#         rr = requests.post(
#             f"{BACKEND_URL}/debug/retrieval",
#             json={"collection": collection, "question": q, "k": 8},
#         )
#         st.code(json.dumps(rr.json(), ensure_ascii=False, indent=2))
