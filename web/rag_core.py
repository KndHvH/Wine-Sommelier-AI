from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OCR deps
import pdf2image
import pytesseract

# LLM (Ollama)
from langchain_ollama import OllamaLLM
import ollama


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_TESSERACT = REPO_ROOT / "Tesseract-OCR" / "tesseract.exe"
DEFAULT_POPPLER_BIN = REPO_ROOT / "poppler" / "poppler-24.08.0" / "Library" / "bin"


def _resolve_tesseract_path() -> Optional[str]:
    if DEFAULT_TESSERACT.exists():
        return str(DEFAULT_TESSERACT)
    return None


def _resolve_poppler_path() -> Optional[str]:
    if DEFAULT_POPPLER_BIN.exists():
        return str(DEFAULT_POPPLER_BIN)
    return None


def _configure_ocr_binaries() -> None:
    tess = _resolve_tesseract_path()
    if tess:
        pytesseract.pytesseract.tesseract_cmd = tess


def _pdf_to_images(pdf_path: str) -> List[Any]:
    poppler_path = _resolve_poppler_path()
    return pdf2image.convert_from_path(pdf_path, poppler_path=poppler_path)


def _ocr_image(img) -> str:
    return pytesseract.image_to_string(img)


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@st.cache_resource(show_spinner=False)
def load_documents_from_pdfs() -> List[Document]:
    _configure_ocr_binaries()
    documents: List[Document] = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        try:
            images = _pdf_to_images(str(pdf))
        except Exception as e:
            st.warning(f"Falha ao converter PDF '{pdf.name}': {e}")
            continue
        for page_index, img in enumerate(images):
            try:
                text = _ocr_image(img)
            except Exception as e:
                st.warning(f"Falha no OCR em '{pdf.name}' pág {page_index+1}: {e}")
                text = ""
            text = _clean_text(text)
            if not text:
                continue
            documents.append(
                Document(page_content=text, metadata={"source": str(pdf), "page": page_index + 1})
            )
    return documents


@st.cache_resource(show_spinner=False)
def load_documents_from_csvs() -> List[Document]:
    from langchain.document_loaders import CSVLoader

    documents: List[Document] = []
    for csv in sorted(DATA_DIR.glob("*.csv")):
        try:
            loader = CSVLoader(str(csv), encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            st.warning(f"Falha ao ler CSV '{csv.name}': {e}")
    return documents


def _split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner=False)
def build_vectorstores() -> Tuple[FAISS, FAISS]:
    pdf_docs = load_documents_from_pdfs()
    csv_docs = load_documents_from_csvs()

    docs_description = _split_documents(pdf_docs, chunk_size=1000, chunk_overlap=200)
    docs_matches = _split_documents(csv_docs, chunk_size=5000, chunk_overlap=200)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    db_description = FAISS.from_documents(docs_description, embeddings)
    db_matches = FAISS.from_documents(docs_matches, embeddings)

    return db_description, db_matches


def get_retrievers(k_description: int = 3, k_matches: int = 3):
    db_description, db_matches = build_vectorstores()
    retriever_description = db_description.as_retriever(search_kwargs={"k": k_description})
    retriever_matches = db_matches.as_retriever(search_kwargs={"k": k_matches})
    return retriever_description, retriever_matches


def get_llm(model_name: str = "mistral:instruct") -> OllamaLLM:
    try:
        ollama.pull(model_name)
    except Exception:
        # Ignorar falhas de pull (pode já existir ou estar offline)
        pass
    return OllamaLLM(model=model_name)


def formatar_resposta(texto: str) -> str:
    texto = re.sub(r"\s+", " ", texto)
    texto = texto.replace("\uFEFF", "").strip()
    return texto


def montar_prompt(role: str, contexto: str, pergunta: str) -> str:
    return f"""
<role>
{role}
</role>
<contexto>:
{contexto}
</contexto>
<pergunta>:
{pergunta}
</pergunta>
<resposta>:
...
"""


def get_roles():
    descritor = """
        Você é um sommelier experiente.
        Sua função é recomendar vinhos assertivos ao cliente
        voce acabou de receber um prato, e deve usar apenas o contexto fornecido para a tarefa,
        descreva brevemente como deveria ser o vinho ideal para esse prato,
        falando sobre o tipo de vinho, cor, aroma, corpo, acidez, etc,
        seja sucinto e em poucas palavras
        exemplo: 
        "Vinho Branco, acididade leve e aromas frutados que harmonizam bem com o sabor cremoso do queijo"
        "Vinho Verde, fresco e frutado, combina bem com a doçura e acidez da morango. 
        "vinho branco leve, com aromas citrus ou frutados, de corpo médio, para que a saborosa carne do camarão se destaque."
    """
    sommelier = """
        Você é um sommelier experiente.
        trabalha em um renomado restaurante 5 estrelas
        Sua função é recomendar vinhos assertivos ao cliente
        voce acabou de receber uma descrição de vinho, e deve usar o contexto para construir uma recomendação apropriada,
        lembre que voce esta falando diretamente com o cliente, entao seja educado e profissional,
        também explique o porque o vinho escolhido é o melhor para o cliente, e como ele orna com o prato
        também forneça seu preço em reais
    """
    return descritor, sommelier
    

def ask_llm(llm: OllamaLLM, prompt: str) -> str:
    return llm.invoke(prompt)


