from __future__ import annotations

import re
import streamlit as st

from rag_core import (
    get_retrievers,
    get_llm,
    montar_prompt,
    formatar_resposta,
    get_roles,
)


st.set_page_config(page_title="Perguntas", page_icon="❓", layout="wide")

st.title("❓ Perguntas sobre Vinhos")
st.caption("RAG usando índices FAISS pré-gerados (descrição e matches)")

with st.sidebar:
    st.subheader("Configurações")
    model_name = st.text_input("Modelo (Mistral API)", value="mistral-small-latest")
    mistral_api_key = st.text_input("MISTRAL_API_KEY (obrigatório no Cloud)", value="", type="password")
    k_desc = st.number_input("Top-K contexto", 1, 10, 3)
    k_match = st.number_input("Top-K produtos", 1, 10, 3)
    debug = st.checkbox(
        "Modo debug",
        value=False,
        help="Exibe documentos recuperados, prompts e respostas intermediárias",
    )
    st.divider()
    if st.button("Recarregar índices", help="Limpa o cache carregado em memória"):
        st.cache_resource.clear()
        st.success("Índices recarregados. Volte a executar a pergunta.")


pergunta = st.text_input(
    "Sua pergunta", placeholder="Ex.: Qual vinho combinar com carne de porco assada?"
)
executar = st.button("Executar")

if executar and pergunta.strip():
    import os as _os
    # Lê secrets do Streamlit Cloud se existirem
    secret_key = ""
    try:
        secret_key = st.secrets.get("MISTRAL_API_KEY", "")  # type: ignore[attr-defined]
    except Exception:
        pass
    key_to_use = mistral_api_key or secret_key or _os.environ.get("MISTRAL_API_KEY", "")
    if key_to_use:
        _os.environ["MISTRAL_API_KEY"] = key_to_use
    retriever_desc, retriever_match = get_retrievers(
        k_description=int(k_desc), k_matches=int(k_match)
    )
    llm = get_llm(model_name=model_name)
    if llm is None:
        st.error("Defina MISTRAL_API_KEY nos Secrets do Streamlit Cloud (ou no campo acima) para usar a API da Mistral.")
        st.stop()

    # Etapa 1: descrição (índice description)
    query_1 = pergunta
    docs_1 = retriever_desc.get_relevant_documents(query_1)
    contexto_1 = ". ".join([d.page_content for d in docs_1])
    role_1 = get_roles()[0]
    prompt_1 = montar_prompt(role_1, contexto_1, pergunta)
    resposta_1 = llm.invoke(prompt_1)

    if debug:
        with st.expander("Etapa 1 - Documentos recuperados (descrição)", expanded=False):
            for d in docs_1:
                meta = d.metadata
                st.markdown(f"- {meta}")
                st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
        with st.expander("Etapa 1 - Prompt (com role)", expanded=False):
            st.code(prompt_1)
        with st.expander("Etapa 1 - Resposta", expanded=False):
            st.write(formatar_resposta(resposta_1))

    # Etapa 2: matches (índice matches)
    query_2 = f"query: {resposta_1}"
    docs_2 = retriever_match.get_relevant_documents(query_2)
    import re as _re
    contexto_2 = ". ".join([_re.sub(r"\b\w+:\s*", "", d.page_content) for d in docs_2])
    role_2 = (
        "Você é um sommelier experiente em um restaurante 5 estrelas. Use o contexto com vinhos reais (nome, preço, descrição) "
        "para recomendar 1 a 3 rótulos ao cliente, explicando por que combinam com o prato e o preço. Seja educado e direto."
    )
    prompt_2 = montar_prompt(role_2, contexto_2, pergunta)
    resposta_final = llm.invoke(prompt_2)

    if debug:
        with st.expander("Etapa 2 - Documentos recuperados (matches)", expanded=False):
            for d in docs_2:
                meta = d.metadata
                st.markdown(f"- {meta}")
                st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
        with st.expander("Etapa 2 - Prompt (com role)", expanded=False):
            st.code(prompt_2)

    st.subheader("Resposta")
    st.write(formatar_resposta(resposta_final))

