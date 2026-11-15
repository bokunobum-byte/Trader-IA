"""
utils.py — Funções auxiliares do Tarder IA.

Responsabilidades:
  • Sanitizar nomes de arquivo (defesa contra path traversal)
  • Validar tamanho dos uploads
  • Ler arquivos de texto com fallback de encoding
  • Extrair texto de PDFs (pypdf), sem OCR
  • Converter imagens em data URLs base64
  • Contar tokens aproximados (tiktoken → fallback char/4)
  • Montar o content multimodal no formato OpenAI
  • Formatar bytes para exibição amigável
"""

from __future__ import annotations

import base64
import io
import logging
import mimetypes
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image  # usado apenas para validar abertura da imagem
from pypdf import PdfReader

from config import (
    EXTENSOES_IMAGEM,
    EXTENSOES_PDF,
    EXTENSOES_TEXTO,
    MAX_FILE_SIZE_BYTES,
)

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# 🧼 SANITIZAÇÃO E VALIDAÇÃO
# ────────────────────────────────────────────────────────────────────────────────

# Regex para remover caracteres perigosos em nome de arquivo.
# Mantém: letras (incl. acentuadas), dígitos, ponto, hífen, underscore, espaço.
_NOME_SEGURO_RE = re.compile(r"[^0-9A-Za-zÀ-ÿ._\-\s]")


def sanitizar_nome(nome: str) -> str:
    """Remove partes perigosas de um nome de arquivo.

    • Corta qualquer caminho (../, /, \\)
    • Remove caracteres especiais
    • Trunca a 120 caracteres
    • Retorna 'arquivo' se o resultado ficar vazio
    """
    if not nome:
        return "arquivo"
    # Remove diretórios (path traversal)
    nome = nome.replace("\\", "/").split("/")[-1]
    # Remove caracteres fora da whitelist
    nome = _NOME_SEGURO_RE.sub("_", nome).strip()
    # Trunca
    nome = nome[:120] or "arquivo"
    return nome


def tamanho_legivel(num_bytes: int) -> str:
    """Converte bytes para KB/MB/GB legíveis."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"


def validar_tamanho(num_bytes: int) -> Tuple[bool, str]:
    """Retorna (ok, mensagem_de_erro)."""
    if num_bytes <= 0:
        return False, "Arquivo vazio."
    if num_bytes > MAX_FILE_SIZE_BYTES:
        return False, (
            f"Arquivo excede o limite de "
            f"{tamanho_legivel(MAX_FILE_SIZE_BYTES)}."
        )
    return True, ""


def classificar_arquivo(nome: str) -> str:
    """Classifica por extensão: 'texto', 'imagem', 'pdf' ou 'desconhecido'."""
    ext = _extensao(nome)
    if ext in EXTENSOES_TEXTO:
        return "texto"
    if ext in EXTENSOES_IMAGEM:
        return "imagem"
    if ext in EXTENSOES_PDF:
        return "pdf"
    return "desconhecido"


def _extensao(nome: str) -> str:
    """Retorna a extensão em minúsculas, com ponto ('.txt')."""
    if "." not in nome:
        return ""
    return "." + nome.rsplit(".", 1)[-1].lower()


# ────────────────────────────────────────────────────────────────────────────────
# 📄 LEITURA DE ARQUIVOS
# ────────────────────────────────────────────────────────────────────────────────

def ler_texto(bytes_arquivo: bytes) -> str:
    """Lê texto com fallback utf-8 → latin-1.

    Se ambos falharem, retorna string vazia e loga o erro.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return bytes_arquivo.decode(enc)
        except UnicodeDecodeError:
            continue
    logger.warning("Não foi possível decodificar arquivo de texto.")
    return ""


def extrair_texto_pdf(bytes_arquivo: bytes) -> Tuple[str, bool]:
    """Extrai texto de um PDF.

    Retorna (texto, tem_texto). Se não houver texto, o chamador deve avisar
    ao usuário que o modelo atual não faz OCR.
    """
    try:
        reader = PdfReader(io.BytesIO(bytes_arquivo))
        partes: List[str] = []
        for pagina in reader.pages:
            try:
                partes.append(pagina.extract_text() or "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao extrair página de PDF: %s", exc)
        texto = "\n\n".join(p for p in partes if p.strip())
        return texto, bool(texto.strip())
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao abrir PDF: %s", exc)
        return "", False


def imagem_para_data_url(bytes_arquivo: bytes, mime: Optional[str] = None) -> str:
    """Converte bytes de imagem em data URL base64 ('data:image/png;base64,...').

    Valida a imagem com Pillow antes (defesa contra payloads maliciosos).
    Lança ValueError se a imagem for inválida.
    """
    try:
        with Image.open(io.BytesIO(bytes_arquivo)) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Imagem inválida ou corrompida: {exc}") from exc

    if not mime:
        mime = mimetypes.guess_type("x." + _extensao("x.png").lstrip("."))[0]
    if not mime or not mime.startswith("image/"):
        mime = "image/png"

    b64 = base64.b64encode(bytes_arquivo).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ────────────────────────────────────────────────────────────────────────────────
# 🧩 PROCESSAMENTO UNIFICADO DE UPLOAD
# ────────────────────────────────────────────────────────────────────────────────

def processar_upload(
    nome_original: str,
    bytes_arquivo: bytes,
) -> Dict[str, Any]:
    """Processa um arquivo enviado e devolve um dicionário padronizado.

    Estrutura de retorno:
        {
          "nome": str,              # nome sanitizado
          "tipo": "texto"|"imagem"|"pdf"|"desconhecido",
          "tamanho": int,           # bytes
          "conteudo": str,          # texto extraído OU data URL (imagem)
          "erro": str,              # vazio se ok
        }
    """
    nome = sanitizar_nome(nome_original)
    tipo = classificar_arquivo(nome)
    resultado: Dict[str, Any] = {
        "nome": nome,
        "tipo": tipo,
        "tamanho": len(bytes_arquivo),
        "conteudo": "",
        "erro": "",
    }

    ok, msg = validar_tamanho(len(bytes_arquivo))
    if not ok:
        resultado["erro"] = msg
        return resultado

    try:
        if tipo == "texto":
            resultado["conteudo"] = ler_texto(bytes_arquivo)

        elif tipo == "pdf":
            texto, tem_texto = extrair_texto_pdf(bytes_arquivo)
            if not tem_texto:
                resultado["erro"] = (
                    "PDF sem texto extraível. O Tarder IA não faz OCR — "
                    "tente outro arquivo ou cole o texto manualmente."
                )
            resultado["conteudo"] = texto

        elif tipo == "imagem":
            mime = mimetypes.guess_type(nome)[0]
            resultado["conteudo"] = imagem_para_data_url(bytes_arquivo, mime)

        else:
            resultado["erro"] = f"Extensão não suportada: {nome}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao processar upload %s", nome)
        resultado["erro"] = f"Erro ao processar arquivo: {exc}"

    return resultado


# ────────────────────────────────────────────────────────────────────────────────
# 🧮 CONTAGEM DE TOKENS
# ────────────────────────────────────────────────────────────────────────────────

# tiktoken pode não estar disponível em todos os ambientes; fazemos fallback.
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENC = None


def contar_tokens(texto: str) -> int:
    """Conta tokens aproximados. Usa tiktoken; fallback = len(texto)//4."""
    if not texto:
        return 0
    if _ENC is not None:
        try:
            return len(_ENC.encode(texto))
        except Exception:  # noqa: BLE001
            pass
    # Heurística conservadora (~4 caracteres por token)
    return max(1, len(texto) // 4)


def estimar_tokens_mensagens(mensagens: List[Dict[str, Any]]) -> int:
    """Estima tokens de uma lista de mensagens no formato OpenAI.

    Considera textos, data URLs de imagem (com peso fixo) e overhead por mensagem.
    """
    total = 0
    for msg in mensagens:
        total += 4  # overhead por mensagem
        content = msg.get("content")
        if isinstance(content, str):
            total += contar_tokens(content)
        elif isinstance(content, list):
            for parte in content:
                if not isinstance(parte, dict):
                    continue
                if parte.get("type") == "text":
                    total += contar_tokens(parte.get("text", ""))
                elif parte.get("type") == "image_url":
                    # Imagens têm custo variável; usamos estimativa fixa por bloco.
                    total += 850
    return total


# ────────────────────────────────────────────────────────────────────────────────
# 🧷 MONTAGEM DE CONTEÚDO MULTIMODAL
# ────────────────────────────────────────────────────────────────────────────────

def montar_content(
    texto_usuario: str,
    anexos: List[Dict[str, Any]],
    modelo_aceita_imagem: bool,
) -> Any:
    """Monta o campo `content` de uma mensagem do usuário.

    Regras:
      • Sem imagens válidas → string simples (texto + transcrições)
      • Com imagens e modelo aceita → lista multimodal OpenAI
      • Se modelo NÃO aceita imagem → ignora imagens e devolve string,
        deixando o app.py avisar o usuário (verificação de UI).

    Anexos devem vir de `processar_upload`, com as chaves nome/tipo/conteudo.
    """
    blocos_texto: List[str] = []
    if texto_usuario.strip():
        blocos_texto.append(texto_usuario.strip())

    imagens_data_urls: List[str] = []
    for anexo in anexos:
        tipo = anexo.get("tipo")
        conteudo = anexo.get("conteudo", "")
        if not conteudo:
            continue
        if tipo == "texto":
            blocos_texto.append(
                f"[Arquivo: {anexo['nome']}]\n```\n{conteudo}\n```"
            )
        elif tipo == "pdf":
            blocos_texto.append(
                f"[PDF: {anexo['nome']}]\n```\n{conteudo}\n```"
            )
        elif tipo == "imagem" and modelo_aceita_imagem:
            imagens_data_urls.append(conteudo)

    texto_consolidado = "\n\n".join(blocos_texto) if blocos_texto else "(sem texto)"

    if not imagens_data_urls:
        return texto_consolidado

    blocos: List[Dict[str, Any]] = [{"type": "text", "text": texto_consolidado}]
    for url in imagens_data_urls:
        blocos.append({"type": "image_url", "image_url": {"url": url}})
    return blocos


# ────────────────────────────────────────────────────────────────────────────────
# 🧠 CHIPS DE ANEXOS (usados pela sidebar)
# ────────────────────────────────────────────────────────────────────────────────

def descrever_anexo(anexo: Dict[str, Any]) -> str:
    """Retorna uma linha amigável: '📎 nome (tipo, tamanho)'."""
    emoji = {
        "texto": "📄",
        "pdf": "📕",
        "imagem": "🖼️",
        "desconhecido": "❔",
    }.get(anexo.get("tipo", ""), "📎")
    nome = anexo.get("nome", "arquivo")
    tamanho = tamanho_legivel(int(anexo.get("tamanho", 0)))
    return f"{emoji} {nome} — {anexo.get('tipo','?')}, {tamanho}"


# ────────────────────────────────────────────────────────────────────────────────
# 🧯 MENSAGENS DE ERRO DA API
# ────────────────────────────────────────────────────────────────────────────────

def mensagem_erro_api(status_code: Optional[int], erro_str: str = "") -> str:
    """Traduz códigos HTTP da NVIDIA em mensagens amigáveis em português."""
    mapa = {
        401: "Chave de API inválida ou expirada. Verifique NVIDIA_API_KEY.",
        402: "Créditos esgotados para este modelo. Tente outro modelo.",
        403: "Acesso negado — o modelo pode exigir aprovação prévia.",
        404: "Modelo não encontrado — verifique o ID em config.py.",
        429: "Limite de requisições atingido (40 RPM). Aguarde alguns segundos.",
        500: "Erro interno no provedor. Tente novamente em instantes.",
        503: "Serviço temporariamente indisponível. Tente novamente.",
    }
    if status_code in mapa:
        return mapa[status_code]
    if "timeout" in erro_str.lower() or "timed out" in erro_str.lower():
        return "O modelo demorou demais. Tente um arquivo menor ou outro modelo."
    return f"Erro inesperado: {erro_str or 'sem detalhes'}"