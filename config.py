"""
config.py — Configuração central do Tarder IA.

Aqui ficam definidos:
  • A lista de modelos NVIDIA disponíveis
  • Os formatos que cada modelo aceita (texto, imagem)
  • Limites de tokens (contexto e saída)
  • Parâmetros "extras" específicos de cada modelo
    (ex.: reasoning_effort, chat_template_kwargs, seed...)
  • Constantes globais de segurança e UI

Para adicionar um novo modelo, basta acrescentar uma entrada no dicionário MODELOS.
"""

from __future__ import annotations

from typing import Any, Dict, List

# ────────────────────────────────────────────────────────────────────────────────
# ⚙️ CONSTANTES GERAIS
# ────────────────────────────────────────────────────────────────────────────────

# Tamanho máximo permitido por arquivo (em bytes) — 50 MB
MAX_FILE_SIZE_BYTES: int = 50 * 1024 * 1024

# Chave da variável de ambiente que guarda a API key da NVIDIA.
# NUNCA coloque a chave direto no código — use .env localmente ou as
# Environment Variables do Render em produção.
NVIDIA_API_KEY_ENV: str = "NVIDIA_API_KEY"

# Base URL do endpoint compatível com OpenAI da NVIDIA NIM.
NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

# Extensões reconhecidas por categoria de arquivo.
EXTENSOES_TEXTO: List[str] = [
    ".txt", ".md", ".csv", ".json", ".py", ".log",
]

EXTENSOES_IMAGEM: List[str] = [
    ".jpg", ".jpeg", ".png", ".webp",
]

EXTENSOES_PDF: List[str] = [
    ".pdf",
]

# MIME types correspondentes (usados no `accept` do st.file_uploader).
MIME_TEXTO: List[str] = [
    "text/plain", "text/markdown", "text/csv",
    "application/json", "text/x-python", "text/x-log",
]

MIME_IMAGEM: List[str] = [
    "image/jpeg", "image/png", "image/webp",
]

MIME_PDF: List[str] = [
    "application/pdf",
]

# ────────────────────────────────────────────────────────────────────────────────
# 🎯 LISTA DE MODELOS DISPONÍVEIS
# ────────────────────────────────────────────────────────────────────────────────
#
# Estrutura de cada entrada:
#   "nome"                 → nome amigável exibido na UI
#   "descricao"            → texto curto exibido no dropdown e no badge
#   "formatos"             → lista de formatos aceitos: "texto", "imagem"
#   "max_tokens_contexto"  → janela máxima de contexto (input + output)
#   "max_tokens_saida"     → limite de tokens gerados na resposta
#   "extras"               → kwargs extras repassados ao payload da API
#
MODELOS: Dict[str, Dict[str, Any]] = {
    "deepseek-ai/deepseek-v4.1-flash": {
        "nome": "DeepSeek V4.1 Flash",
        "descricao": "Multimodal rápido — textos e imagens",
        "formatos": ["texto", "imagem"],
        "max_tokens_contexto": 1048576,
        "max_tokens_saida": 8192,
        "extras": {},
    },
    "mistralai/mistral-large-4-1t-a52b": {
        "nome": "Mistral Large 4",
        "descricao": "Raciocínio avançado, multimodal",
        "formatos": ["texto", "imagem"],
        "max_tokens_contexto": 128000,
        "max_tokens_saida": 4096,
        "extras": {"reasoning_effort": "high"},
    },
    "nvidia/nemotron-3.5-bolt-e4b": {
        "nome": "Nemotron 3.5 Bolt",
        "descricao": "Rápido, raciocínio com thinking mode",
        "formatos": ["texto"],
        "max_tokens_contexto": 131072,
        "max_tokens_saida": 1024,
        "extras": {
            "chat_template_kwargs": {"enable_thinking": True},
            "parallel_tool_calls": True,
            "seed": 42,
        },
    },
    "minimaxai/minimax-m3.1-flash": {
        "nome": "MiniMax M3.1 Flash",
        "descricao": "Multimodal, contexto grande (16K saída)",
        "formatos": ["texto", "imagem"],
        "max_tokens_contexto": 200000,
        "max_tokens_saida": 16384,
        "extras": {"seed": 0},
    },
    "stepfun-ai/step-5-preview": {
        "nome": "Step 5 Preview",
        "descricao": "Visão + texto, respostas equilibradas",
        "formatos": ["texto", "imagem"],
        "max_tokens_contexto": 128000,
        "max_tokens_saida": 4096,
        "extras": {},
    },
    "z-ai/glm-5.3-flash": {
        "nome": "GLM 5.3 Flash",
        "descricao": "Leve, ótimo para texto puro e system prompts",
        "formatos": ["texto"],
        "max_tokens_contexto": 128000,
        "max_tokens_saida": 1024,
        "extras": {},
    },
}

# ────────────────────────────────────────────────────────────────────────────────
# 🎨 CONFIGURAÇÃO DE UI (usada em app.py)
# ────────────────────────────────────────────────────────────────────────────────

APP_TITULO: str = "Tarder IA"
APP_ICONE: str = "🤖"
APP_RODAPE: str = "powered by NVIDIA NIM"

# Paleta de cores aplicada via CSS inline no app.py
COR_FUNDO: str = "#0f172a"       # fundo escuro suave
COR_VERDE_NVIDIA: str = "#76b900"  # acento verde NVIDIA
COR_AZUL_ACENTO: str = "#3b82f6"   # acento azul
COR_TEXTO: str = "#e2e8f0"

# ────────────────────────────────────────────────────────────────────────────────
# 🔐 HELPERS DE ACESSO RÁPIDO
# ────────────────────────────────────────────────────────────────────────────────

def get_modelo(model_id: str) -> Dict[str, Any]:
    """Retorna o dicionário de configuração de um modelo específico.

    Lança KeyError se o modelo não existir — assim evitamos falhas silenciosas.
    """
    if model_id not in MODELOS:
        raise KeyError(f"Modelo desconhecido em config.MODELOS: {model_id!r}")
    return MODELOS[model_id]


def modelo_aceita(model_id: str, formato: str) -> bool:
    """True se o modelo aceita o formato informado ('texto' ou 'imagem')."""
    return formato in get_modelo(model_id).get("formatos", [])


def lista_modelos_nomes() -> List[str]:
    """Lista amigável usada no dropdown da sidebar."""
    return [meta["nome"] for meta in MODELOS.values()]


def model_id_por_nome(nome_amigavel: str) -> str:
    """Converte o nome amigável de volta ao ID técnico do modelo."""
    for mid, meta in MODELOS.items():
        if meta["nome"] == nome_amigavel:
            return mid
    raise KeyError(f"Nenhum modelo com nome amigável {nome_amigavel!r}")