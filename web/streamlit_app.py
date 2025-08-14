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


st.set_page_config(page_title="Wine Sommelier AI", page_icon="🍷", layout="wide")

st.title("🍷 Descreva seu prato e descubra o melhor vinho")
# st.caption("RAG usando índices FAISS pré-gerados (descrição e matches)")

with st.sidebar:
    st.subheader("Configurações")
    model_name = st.text_input("Modelo (Ollama)", value="mistral:instruct")
    k_desc = st.number_input("Top-K contexto", 1, 10, 3)
    k_match = st.number_input("Top-K produtos", 1, 10, 3)
    debug = st.checkbox(
        "Modo debug",
        value=False,
        help="Exibe documentos recuperados, prompts e respostas intermediárias",
    )
    # st.divider()
    # if st.button("Recarregar índices", help="Limpa o cache carregado em memória"):
    #     st.cache_resource.clear()
    #     st.success("Índices recarregados. Volte a executar a pergunta.")


pergunta = st.text_input(
    "Sua pergunta", placeholder="Ex.: Qual vinho combinar com carne de porco assada?"
)
executar = st.button("Executar")

if executar and pergunta.strip():
    status = st.status("Executando...", expanded=debug)
    if debug:
        status.write("Carregando bases vetoriais...")
    retriever_desc, retriever_match = get_retrievers(
        k_description=int(k_desc), k_matches=int(k_match)
    )
    if debug:
        status.write("Carregando LLM...")
    llm = get_llm(model_name=model_name)
    

    # Etapa 1: descrição (índice description)
    query_1 = pergunta
    if debug: 
        status.write("Recuperando documentos...")
    docs_1 = retriever_desc.get_relevant_documents(query_1)
    if debug:
        with st.expander("Etapa 1 - Documentos recuperados (descrição)", expanded=False):
            for d in docs_1:
                meta = d.metadata['source']
                st.markdown(f"#### {meta}")
                st.code(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
                
    contexto_1 = ". ".join([d.page_content for d in docs_1])
    role_1 = get_roles()[0]
    prompt_1 = montar_prompt(role_1, contexto_1, pergunta)
    if debug:
        with st.expander("Etapa 1 - Prompt (com role)", expanded=False):
            st.code(prompt_1)
        status.write("Consultando LLM...")
    resposta_1 = llm.invoke(prompt_1)
    resposta_1 = resposta_1[:300]

    if debug:
        with st.expander("Etapa 1 - Resposta", expanded=False):
            st.write(formatar_resposta(resposta_1))

    # Etapa 2: matches (índice matches)
    query_2 = f"query: {resposta_1}"
    if debug:
        status.write("Recuperando documentos...")
    docs_2 = retriever_match.get_relevant_documents(query_2)
    if debug:
        with st.expander("Etapa 2 - Documentos recuperados (matches)", expanded=False):
            for d in docs_2:
                meta = d.metadata['source']
                st.markdown(f"#### {meta}")
                st.code(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
    import re as _re
    contexto_2 = ". ".join([_re.sub(r"\b\w+:\s*", "", d.page_content) for d in docs_2])
    role_2 = get_roles()[1]    
    prompt_2 = montar_prompt(role_2, contexto_2, pergunta)
    if debug:
        with st.expander("Etapa 2 - Prompt (com role)", expanded=False):
            st.code(prompt_2)
        status.write("Consultando LLM...")
    resposta_final = llm.invoke(prompt_2)

    status.update(label="Processado", state="complete", expanded=False)
    with st.expander("Resposta", expanded=True):
        st.markdown(resposta_final, unsafe_allow_html=True)

