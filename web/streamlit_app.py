import streamlit as st

st.set_page_config(
    page_title="Vinho RAG",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍷 Vinho RAG")
st.caption("Navegue pelas páginas na barra lateral: Arquivos e Perguntas.")

st.markdown(
    """
    - Arquivos: visualize os PDFs e CSVs de contexto.
    - Perguntas: faça perguntas com RAG e use o modo debug para ver as etapas.
    """
)


