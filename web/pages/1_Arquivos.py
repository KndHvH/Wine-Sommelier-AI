import os
from pathlib import Path

import streamlit as st

from web.rag_core import DATA_DIR, load_documents_from_pdfs, load_documents_from_csvs


st.set_page_config(page_title="Arquivos", page_icon="📂", layout="wide")

st.title("📂 Arquivos de Contexto")
st.caption("Lista e pré-visualização dos arquivos em 'data/'.")


def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


files = sorted(DATA_DIR.glob("*"))
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Arquivos")
    for f in files:
        size = human_size(f.stat().st_size)
        st.write(f"- {f.name} ({size})")

with col2:
    st.subheader("Pré-visualização")
    tab_pdf, tab_csv = st.tabs(["PDFs", "CSVs"])

    with tab_pdf:
        st.write("Documentos OCR extraídos dos PDFs (primeiras 3 páginas por arquivo, se existirem).")
        pdf_docs = load_documents_from_pdfs()
        grouped = {}
        for d in pdf_docs:
            grouped.setdefault(d.metadata.get("source", "?"), []).append(d)
        for source, docs in grouped.items():
            st.markdown(f"**{Path(source).name}**")
            for d in docs[:3]:
                with st.expander(f"Página {d.metadata.get('page', '?')}"):
                    st.write(d.page_content[:1500] + ("..." if len(d.page_content) > 1500 else ""))

    with tab_csv:
        csv_docs = load_documents_from_csvs()
        grouped = {}
        for d in csv_docs:
            grouped.setdefault(d.metadata.get("source", "?"), []).append(d)
        for source, docs in grouped.items():
            st.markdown(f"**{Path(source).name}**")
            for d in docs[:5]:
                with st.expander(f"Linha {d.metadata.get('row', '?')}"):
                    st.write(d.page_content)


