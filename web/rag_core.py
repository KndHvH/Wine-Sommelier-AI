from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# LLM (Ollama)
from langchain_ollama import OllamaLLM
import ollama


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
FAISS_DIR = REPO_ROOT / "faiss"


def _find_index_dir(preferred: Path) -> Optional[Path]:
    if (preferred / "index.faiss").exists() and (preferred / "index.pkl").exists():
        return preferred
    return None


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@st.cache_resource(show_spinner=False)
def _load_vectorstore_from_dir(dir_path: Path, _embeddings: HuggingFaceEmbeddings) -> FAISS:
    return FAISS.load_local(
        folder_path=str(dir_path),
        embeddings=_embeddings,
        allow_dangerous_deserialization=True,
    )


def _clean_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@st.cache_resource(show_spinner=False)
def load_vectorstores() -> Tuple[FAISS, FAISS]:
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # Tenta diferentes layouts de pastas
    candidates_desc = [
        FAISS_DIR / "description",
        REPO_ROOT / "faiss_description",
        REPO_ROOT / "faiss_desc",
    ]
    candidates_match = [
        FAISS_DIR / "matches",
        REPO_ROOT / "faiss_matches",
        REPO_ROOT / "faiss_match",
    ]

    desc_dir = next((p for p in candidates_desc if _find_index_dir(p)), None)
    match_dir = next((p for p in candidates_match if _find_index_dir(p)), None)

    if not desc_dir or not match_dir:
        raise FileNotFoundError(
            "Índices FAISS não encontrados. Esperado em 'faiss/description' e 'faiss/matches' (ou variantes)."
        )

    db_description = _load_vectorstore_from_dir(desc_dir, embeddings)
    db_matches = _load_vectorstore_from_dir(match_dir, embeddings)

    return db_description, db_matches


def get_index_paths() -> Tuple[Optional[Path], Optional[Path]]:
    candidates_desc = [
        FAISS_DIR / "description",
        REPO_ROOT / "faiss_description",
        REPO_ROOT / "faiss_desc",
    ]
    candidates_match = [
        FAISS_DIR / "matches",
        REPO_ROOT / "faiss_matches",
        REPO_ROOT / "faiss_match",
    ]
    desc_dir = next((p for p in candidates_desc if (p / "index.faiss").exists()), None)
    match_dir = next((p for p in candidates_match if (p / "index.faiss").exists()), None)
    return desc_dir, match_dir


def get_vectorstore_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        db_desc, db_match = load_vectorstores()
        stats["description_vectors"] = getattr(db_desc.index, "ntotal", None)
        stats["matches_vectors"] = getattr(db_match.index, "ntotal", None)
    except Exception as e:
        stats["error"] = str(e)
    return stats


def get_retrievers(k_description: int = 3, k_matches: int = 3):
    db_description, db_matches = load_vectorstores()
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


