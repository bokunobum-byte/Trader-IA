"""
api_client.py — Wrapper para a API da NVIDIA NIM.

Único ponto do projeto que fala com o provedor externo. Responsabilidades:

  • Ler a API key EXCLUSIVAMENTE de variável de ambiente
  • Criar o cliente OpenAI apontando para o base_url da NVIDIA
  • Montar o payload respeitando os `extras` de cada modelo em config.MODELOS
  • Suportar streaming (retorna generator) ou resposta única (retorna string)
  • Traduzir erros HTTP da NVIDIA em mensagens amigáveis em PT-BR

Regras de segurança:
  • NUNCA logar a API key (nem em exceções)
  • NUNCA aceitar a chave como parâmetro — sempre do ambiente
  • Falhar cedo com mensagem clara se a chave não estiver configurada
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, List, Optional

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from config import MODELOS, NVIDIA_API_KEY_ENV, NVIDIA_BASE_URL, get_modelo
from utils import mensagem_erro_api

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# 🔑 CHAVE E CLIENTE
# ────────────────────────────────────────────────────────────────────────────────


class ChaveAusenteError(RuntimeError):
    """Erro levantado quando NVIDIA_API_KEY não está definida."""


def obter_api_key() -> str:
    """Retorna a chave da NVIDIA a partir do ambiente.

    Levanta ChaveAusenteError com mensagem clara se ausente/vazia.
    Nunca inclui a chave em logs ou na exceção.
    """
    key = os.environ.get(NVIDIA_API_KEY_ENV, "").strip()
    if not key:
        raise ChaveAusenteError(
            f"Variável de ambiente {NVIDIA_API_KEY_ENV} não configurada. "
            "Defina-a localmente (.env) ou no painel do Render antes de usar o chat."
        )
    return key


def criar_cliente() -> OpenAI:
    """Cria o cliente OpenAI apontando para o endpoint NIM da NVIDIA.

    timeout=60s e max_retries=1 — evita travar a instância 512MB do Render.
    """
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=obter_api_key(),
        timeout=60.0,
        max_retries=1,
    )


# ────────────────────────────────────────────────────────────────────────────────
# 🧱 MONTAGEM DE PAYLOAD
# ────────────────────────────────────────────────────────────────────────────────


def _montar_payload(
    model_id: str,
    mensagens: List[Dict[str, Any]],
    *,
    stream: bool,
    max_tokens_override: Optional[int] = None,
) -> Dict[str, Any]:
    """Constrói o dict de kwargs enviado a chat.completions.create.

    Regras:
      • model → o ID técnico em config.MODELOS
      • messages → lista pronta do app.py
      • stream → flag
      • max_tokens → respeita o limite do modelo; override se menor
      • extras → copiados só se existirem para o modelo (inclui reasoning_effort,
        chat_template_kwargs, parallel_tool_calls, seed etc.)
    """
    meta = get_modelo(model_id)
    limite_saida = int(meta.get("max_tokens_saida", 4096))

    max_tokens = limite_saida
    if max_tokens_override is not None and max_tokens_override > 0:
        max_tokens = min(limite_saida, int(max_tokens_override))

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": mensagens,
        "stream": stream,
        "max_tokens": max_tokens,
    }

    # Copia "extras" respeitando exatamente o que o modelo declara.
    for chave, valor in (meta.get("extras") or {}).items():
        if valor is None:
            continue
        payload[chave] = valor

    return payload


# ────────────────────────────────────────────────────────────────────────────────
# 🚀 CHAMADA PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────


def gerar_resposta(
    model_id: str,
    mensagens: List[Dict[str, Any]],
    *,
    stream: bool = True,
    max_tokens_override: Optional[int] = None,
) -> Any:
    """Envia mensagens ao modelo e devolve:

        • stream=True  → Generator[str] com chunks de texto
        • stream=False → string completa

    Em caso de erro da API, levanta RuntimeError com mensagem PT-BR amigável.
    """
    if model_id not in MODELOS:
        raise ValueError(f"Modelo desconhecido: {model_id!r}")

    try:
        client = criar_cliente()
        payload = _montar_payload(
            model_id,
            mensagens,
            stream=stream,
            max_tokens_override=max_tokens_override,
        )
        logger.info(
            "Chamando modelo=%s stream=%s tokens_max=%s extras=%s",
            model_id,
            stream,
            payload.get("max_tokens"),
            list((MODELOS[model_id].get("extras") or {}).keys()),
        )

        resposta = client.chat.completions.create(**payload)

        if stream:
            return _iterar_stream(resposta)

        return _extrair_texto_unico(resposta)

    except ChaveAusenteError:
        raise
    except RateLimitError as exc:  # 429
        raise RuntimeError(mensagem_erro_api(429)) from exc
    except APITimeoutError as exc:
        raise RuntimeError(mensagem_erro_api(None, "timeout")) from exc
    except APIConnectionError as exc:
        raise RuntimeError(
            "Não foi possível conectar à NVIDIA. Verifique sua internet."
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(mensagem_erro_api(exc.status_code, str(exc))) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro inesperado ao chamar a API NVIDIA")
        raise RuntimeError(f"Erro inesperado: {exc}") from exc


# ────────────────────────────────────────────────────────────────────────────────
# 🌊 STREAMING E RESPOSTA ÚNICA
# ────────────────────────────────────────────────────────────────────────────────


def _iterar_stream(stream_resp: Any) -> Generator[str, None, None]:
    """Itera um stream OpenAI e produz apenas o texto incremental.

    Ignora chunks sem `delta.content` (pode ocorrer em thinking mode / tools).
    """
    try:
        for chunk in stream_resp:
            try:
                delta = chunk.choices[0].delta
            except (IndexError, AttributeError):
                continue
            texto = getattr(delta, "content", None)
            if texto:
                yield texto
    except APITimeoutError as exc:
        raise RuntimeError(mensagem_erro_api(None, "timeout")) from exc
    except APIStatusError as exc:
        raise RuntimeError(mensagem_erro_api(exc.status_code, str(exc))) from exc


def _extrair_texto_unico(resposta: Any) -> str:
    """Extrai texto de uma resposta única do SDK OpenAI."""
    try:
        return resposta.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        raise RuntimeError("Resposta da API em formato inesperado.") from exc


# ────────────────────────────────────────────────────────────────────────────────
# 🧪 VERIFICAÇÃO RÁPIDA
# ────────────────────────────────────────────────────────────────────────────────


def ping_modelo(model_id: str) -> str:
    """Envia uma mensagem trivial para confirmar que o modelo responde.

    Útil para um botão "Testar conexão" na sidebar.
    Levanta RuntimeError com a mesma tradução amigável dos erros normais.
    """
    return gerar_resposta(
        model_id,
        [{"role": "user", "content": "ping"}],
        stream=False,
        max_tokens_override=16,
    )