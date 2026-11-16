"""
app.py — Interface Streamlit principal do Tarder IA.

Responsabilidades:
  • Carregar variáveis de ambiente (.env local)
  • Aplicar CSS escuro moderno (verde NVIDIA + azul)
  • Renderizar sidebar (logo, seletor de modelo, uploads, tokens, limpar)
  • Renderizar chat com histórico e streaming
  • Avisar sobre incompatibilidade imagem/modelo e estouro de tokens
  • Nunca expor a API key (delegado a api_client)

Execução local:
    streamlit run app.py
"""

from __future__ import annotations

# ── Carrega .env local (no-op no Render: lá as env vars já existem) ────────
from dotenv import load_dotenv

load_dotenv()

from typing import Any, Dict, List

import streamlit as st

from api_client import ChaveAusenteError, gerar_resposta, obter_api_key, ping_modelo
from config import (
    APP_ICONE,
    APP_RODAPE,
    APP_TITULO,
    COR_AZUL_ACENTO,
    COR_FUNDO,
    COR_TEXTO,
    COR_VERDE_NVIDIA,
    EXTENSOES_IMAGEM,
    EXTENSOES_PDF,
    EXTENSOES_TEXTO,
    MAX_FILE_SIZE_BYTES,
    MODELOS,
)
from utils import (
    contar_tokens,
    descrever_anexo,
    estimar_tokens_mensagens,
    montar_content,
    processar_upload,
    tamanho_legivel,
)

# ────────────────────────────────────────────────────────────────────────────────
# ⚙️ CONFIGURAÇÃO INICIAL
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title=APP_TITULO,
    page_icon=APP_ICONE,
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────────────────────
# 🎨 CSS INLINE
# ────────────────────────────────────────────────────────────────────────────────


def _injetar_css() -> None:
    """Injeta o CSS escuro moderno com acentos NVIDIA."""
    css = f"""
    <style>
        .stApp {{
            background-color: {COR_FUNDO};
            color: {COR_TEXTO};
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
        }}
        .main .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 5rem;
            max-width: 1100px;
        }}
        [data-testid="stSidebar"] {{
            background-color: #0b1220;
            border-right: 1px solid #1e293b;
        }}
        [data-testid="stSidebar"] * {{
            color: {COR_TEXTO};
        }}
        /* Bolhas de chat */
        [data-testid="stChatMessage"] {{
            background-color: #1e293b;
            border-radius: 18px;
            padding: 12px 18px;
            margin-bottom: 8px;
            border: 1px solid #33415580;
        }}
        /* Botões */
        .stButton > button {{
            background-color: {COR_AZUL_ACENTO};
            color: #ffffff;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background-color 0.15s ease;
        }}
        .stButton > button:hover {{
            background-color: {COR_VERDE_NVIDIA};
            color: #0f172a;
        }}
        /* Botões de ação crítica */
        .stButton > button[kind="secondary"] {{
            background-color: #1e293b;
            color: {COR_TEXTO};
            border: 1px solid #334155;
        }}
        /* Input do chat */
        [data-testid="stChatInput"] textarea {{
            background-color: #1e293b !important;
            color: {COR_TEXTO} !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
        }}
        /* Área de upload */
        [data-testid="stFileUploader"] section {{
            background-color: #0b1220;
            border: 1px dashed #334155;
            border-radius: 10px;
        }}
        /* Títulos e código */
        h1, h2, h3 {{ color: {COR_VERDE_NVIDIA}; }}
        code {{
            background-color: #0b1220;
            color: {COR_VERDE_NVIDIA};
            padding: 2px 6px;
            border-radius: 4px;
        }}
        pre code {{
            background-color: #0b1220 !important;
        }}
        a {{ color: {COR_AZUL_ACENTO}; }}
        /* Métrica (token counter) */
        [data-testid="stMetricValue"] {{
            color: {COR_VERDE_NVIDIA} !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: #94a3b8 !important;
        }}
        /* Divider */
        hr {{ border-color: #1e293b; }}
        /* Selectbox */
        [data-testid="stSelectbox"] > div > div {{
            background-color: #1e293b;
            border-color: #334155;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# 🧠 ESTADO DA SESSÃO
# ────────────────────────────────────────────────────────────────────────────────


def _init_state() -> None:
    """Inicializa chaves de sessão de forma idempotente."""
    if "mensagens" not in st.session_state:
        # Cada item: {"role", "content", "display_text"?, "anexos_nomes"?, "modelo_id"?}
        st.session_state.mensagens = []

    if "anexos" not in st.session_state:
        # Cada item: dict vindo de utils.processar_upload
        st.session_state.anexos = []

    if "modelo_id" not in st.session_state:
        st.session_state.modelo_id = list(MODELOS.keys())[0]

    if "upload_nonce" not in st.session_state:
        # Serve para resetar o st.file_uploader ao trocar de modelo / remover anexo
        st.session_state.upload_nonce = 0


# ────────────────────────────────────────────────────────────────────────────────
# 🔎 HELPERS DE ESTADO
# ────────────────────────────────────────────────────────────────────────────────


def _modelo_atual() -> Dict[str, Any]:
    return MODELOS[st.session_state.modelo_id]


def _mensagens_para_api() -> List[Dict[str, Any]]:
    """Retorna histórico no formato aceito pela API (apenas role + content)."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.mensagens
    ]


def _estimar_tokens_totais() -> int:
    """Tokens do histórico + anexos pendentes."""
    total = estimar_tokens_mensagens(_mensagens_para_api())
    for anexo in st.session_state.anexos:
        if anexo["tipo"] in ("texto", "pdf"):
            total += contar_tokens(anexo.get("conteudo", ""))
        elif anexo["tipo"] == "imagem":
            total += 850  # estimativa fixa por imagem
    return total


def _tem_imagem_anexada() -> bool:
    return any(a["tipo"] == "imagem" for a in st.session_state.anexos)


def _incompatibilidade_imagem() -> bool:
    """True se anexo de imagem existe mas o modelo ativo não aceita imagem."""
    if "imagem" in _modelo_atual()["formatos"]:
        return False
    return _tem_imagem_anexada()


# ────────────────────────────────────────────────────────────────────────────────
# 🖼️ SIDEBAR
# ────────────────────────────────────────────────────────────────────────────────


def _render_sidebar() -> None:
    with st.sidebar:
        # Logo
        st.markdown(
            f"""
            <div style="padding: 4px 0 14px 0;">
                <div style="font-size: 1.7em; font-weight: 700; color: {COR_VERDE_NVIDIA};">
                    {APP_ICONE} {APP_TITULO}
                </div>
                <div style="font-size: 0.72em; color: #64748b; margin-top: 2px;">
                    NVIDIA NIM · multimodal
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_seletor_modelo()
        st.divider()

        # Aviso antecipado de incompatibilidade
        if _incompatibilidade_imagem():
            st.warning(
                "Este modelo não aceita imagens. "
                "Remova o anexo ou troque de modelo."
            )

        _render_uploader()
        _render_anexos_chips()

        st.divider()
        _render_token_counter()
        st.divider()
        _render_acoes()

        # Rodapé
        st.markdown(
            f"""
            <div style="text-align:center; color:#475569; font-size:0.72em; margin-top:24px;">
                {APP_RODAPE}
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_seletor_modelo() -> None:
    """Dropdown com nome + descrição do modelo."""
    opcoes = [meta["nome"] for meta in MODELOS.values()]
    nome_para_id = {meta["nome"]: mid for mid, meta in MODELOS.items()}

    nome_atual = _modelo_atual()["nome"]
    idx_atual = opcoes.index(nome_atual) if nome_atual in opcoes else 0

    escolhido = st.selectbox("Modelo", opcoes, index=idx_atual)
    novo_id = nome_para_id[escolhido]

    if novo_id != st.session_state.modelo_id:
        st.session_state.modelo_id = novo_id
        # Reseta o uploader para não confundir formatos
        st.session_state.upload_nonce += 1

    meta = _modelo_atual()
    st.caption(meta["descricao"])
    st.caption(
        f"Contexto: **{meta['max_tokens_contexto']:,}** tokens · "
        f"Saída máx.: **{meta['max_tokens_saida']:,}**"
    )


def _render_uploader() -> None:
    """Uploader dinâmico conforme os formatos do modelo ativo.

    IMPORTANTE — leia antes de "consertar":

    Passamos APENAS extensões (sem ponto) ao `type=` do file_uploader.
    NÃO adicione MIME types aqui. Misturar MIME types (ex.: "text/plain")
    com extensões (ex.: "png") faz com que alguns browsers — notavelmente
    Chrome/Edge no Windows — abram o diálogo nativo já filtrado pelo
    primeiro MIME, escondendo imagens e PDFs por trás do dropdown.

    Só com extensões, o diálogo abre em "Todos os arquivos suportados" e
    mostra tudo lado a lado. Se o comportamento voltar a quebrar, revise
    esta função primeiro.
    """
    meta = _modelo_atual()
    formatos = meta["formatos"]

    exts: List[str] = []
    if "texto" in formatos:
        exts += EXTENSOES_TEXTO
    if "imagem" in formatos:
        exts += EXTENSOES_IMAGEM
    # PDF sempre disponível (extraído como texto pelo pypdf)
    exts += EXTENSOES_PDF

    # Só extensões sem ponto — máxima compatibilidade entre SOs/browsers.
    accept = [e.lstrip(".") for e in exts]

    # Lista os formatos aceitos de forma legível no help do widget.
    formatos_legiveis = ", ".join(sorted({e.lstrip(".").upper() for e in exts}))

    st.markdown("##### 📎 Anexos")
    arquivos = st.file_uploader(
        "Envie arquivos (múltiplos)",
        type=accept,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_nonce}",
        label_visibility="collapsed",
        help=(
            f"Tamanho máx.: {tamanho_legivel(MAX_FILE_SIZE_BYTES)} por arquivo.\n\n"
            f"Formatos aceitos neste modelo: {formatos_legiveis}."
        ),
    )

    if arquivos:
        _processar_novos_uploads(arquivos)


def _processar_novos_uploads(arquivos: List[Any]) -> None:
    """Processa os uploads do widget e adiciona à lista de anexos (sem duplicar)."""
    existentes = {(a["nome"], a["tamanho"]) for a in st.session_state.anexos}

    for arq in arquivos:
        try:
            bytes_arq = arq.read()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao ler **{arq.name}**: {exc}")
            continue

        resultado = processar_upload(arq.name, bytes_arq)

        if resultado["erro"]:
            st.error(f"**{resultado['nome']}**: {resultado['erro']}")
            continue

        chave = (resultado["nome"], resultado["tamanho"])
        if chave in existentes:
            continue
        existentes.add(chave)
        st.session_state.anexos.append(resultado)


def _render_anexos_chips() -> None:
    """Chips com nome/tipo/tamanho + botão de remover por arquivo."""
    if not st.session_state.anexos:
        st.caption("_Nenhum anexo pendente._")
        return

    removidos: List[int] = []
    for i, anexo in enumerate(list(st.session_state.anexos)):
        col_info, col_btn = st.columns([5, 1])
        with col_info:
            st.markdown(
                f"<div style='background:#1e293b;padding:6px 10px;"
                f"border-radius:8px;margin-bottom:4px;font-size:0.82em;'>"
                f"{descrever_anexo(anexo)}</div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(
                "🗑️",
                key=f"rm_anexo_{i}_{st.session_state.upload_nonce}",
                help="Remover anexo",
            ):
                removidos.append(i)

    if removidos:
        for i in sorted(removidos, reverse=True):
            st.session_state.anexos.pop(i)
        st.session_state.upload_nonce += 1
        st.rerun()


def _render_token_counter() -> None:
    """Mostra tokens usados vs. limite e avisa perto do teto."""
    meta = _modelo_atual()
    limite = meta["max_tokens_contexto"]
    total = _estimar_tokens_totais()
    pct = (total / limite) if limite else 0.0

    st.markdown("##### 🧮 Contexto")
    st.metric(
        label="Tokens no histórico + anexos",
        value=f"{total:,}",
        delta=f"/ {limite:,}",
    )
    if pct >= 0.85:
        st.warning(
            f"⚠️ {pct * 100:.0f}% do contexto utilizado. "
            "Considere limpar a conversa ou trocar para um modelo com contexto maior."
        )
    elif pct >= 0.6:
        st.info(f"ℹ️ {pct * 100:.0f}% do contexto utilizado.")


def _render_acoes() -> None:
    """Botões Limpar + Testar conexão."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Limpar conversa", use_container_width=True):
            _limpar_conversa()
    with col2:
        if st.button("🧪 Testar conexão", use_container_width=True):
            _testar_conexao()


def _limpar_conversa() -> None:
    st.session_state.mensagens = []
    st.session_state.anexos = []
    st.session_state.upload_nonce += 1
    st.rerun()


def _testar_conexao() -> None:
    model_id = st.session_state.modelo_id
    try:
        resultado = ping_modelo(model_id)
        trecho = (resultado or "").strip()[:80] or "(vazio)"
        st.success(f"✅ Conexão OK — resposta do modelo: `{trecho}`")
    except ChaveAusenteError as exc:
        st.error(str(exc))
    except RuntimeError as exc:
        st.error(str(exc))


# ────────────────────────────────────────────────────────────────────────────────
# 💬 ÁREA PRINCIPAL
# ────────────────────────────────────────────────────────────────────────────────


def _render_badge_modelo() -> None:
    """Badge no topo da área principal com o modelo ativo."""
    meta = _modelo_atual()
    st.markdown(
        f"""
        <div style="
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(90deg, {COR_VERDE_NVIDIA}1a, {COR_AZUL_ACENTO}1a);
            border: 1px solid {COR_VERDE_NVIDIA}55;
            border-radius: 999px;
            padding: 6px 16px;
            margin-bottom: 14px;
        ">
            <span style="color:{COR_VERDE_NVIDIA}; font-weight: 700;">
                ● {meta['nome']}
            </span>
            <span style="color:#94a3b8; font-size: 0.85em;">
                {meta['descricao']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_mensagem(msg: Dict[str, Any]) -> None:
    """Renderiza uma única mensagem do histórico."""
    role = msg["role"]
    avatar = "🧑" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        # Badge do modelo nas respostas do assistente
        if role == "assistant" and "modelo_id" in msg:
            meta = MODELOS.get(msg["modelo_id"])
            if meta:
                st.markdown(
                    f"<div style='color:{COR_VERDE_NVIDIA}; "
                    f"font-size:0.72em; font-weight:600; margin-bottom:6px;'>"
                    f"● {meta['nome']}</div>",
                    unsafe_allow_html=True,
                )

        if role == "user":
            display = msg.get("display_text", "")
            st.markdown(display or "_(mensagem vazia)_")
            nomes = msg.get("anexos_nomes") or []
            if nomes:
                chips = " ".join(f"`📎 {n}`" for n in nomes)
                st.markdown(chips)
        else:
            st.markdown(msg.get("content", ""))


def _render_main() -> None:
    meta = _modelo_atual()
    _render_badge_modelo()

    # Estado vazio (welcome)
    if not st.session_state.mensagens:
        st.markdown(
            f"""
            <div style="text-align:center; padding: 60px 20px; color:#64748b;">
                <h2 style="color:{COR_TEXTO}; margin-bottom: 8px;">Olá! 👋</h2>
                <p style="font-size: 1.05em;">
                    Escolha um modelo na barra lateral, anexe arquivos se quiser e comece a conversar.
                </p>
                <p style="font-size: 0.9em;">
                    Modelo ativo:
                    <b style="color:{COR_VERDE_NVIDIA};">{meta['nome']}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Histórico
    for msg in st.session_state.mensagens:
        _render_mensagem(msg)

    # Input do chat
    prompt = st.chat_input("Digite sua mensagem...")
    if prompt:
        _processar_envio(prompt)


# ────────────────────────────────────────────────────────────────────────────────
# 🚀 ENVIO DE MENSAGEM
# ────────────────────────────────────────────────────────────────────────────────


def _processar_envio(prompt: str) -> None:
    model_id = st.session_state.modelo_id
    meta = MODELOS[model_id]

    # 1) Verificação antecipada de chave
    try:
        obter_api_key()
    except ChaveAusenteError as exc:
        st.error(str(exc))
        return

    # 2) Monta content multimodal (texto + anexos)
    anexos_snapshot = list(st.session_state.anexos)
    try:
        content = montar_content(
            prompt,
            anexos_snapshot,
            modelo_aceita_imagem=("imagem" in meta["formatos"]),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao montar conteúdo: {exc}")
        return

    # 3) Registra a mensagem do usuário imediatamente (preserva o texto)
    user_msg: Dict[str, Any] = {
        "role": "user",
        "content": content,
        "display_text": prompt,
        "anexos_nomes": [a["nome"] for a in anexos_snapshot],
    }
    st.session_state.mensagens.append(user_msg)

    # 4) Consome anexos e reseta o uploader
    st.session_state.anexos = []
    st.session_state.upload_nonce += 1

    # 5) Renderiza a mensagem do usuário
    _render_mensagem(user_msg)

    # 6) Chama a API com streaming
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            f"<div style='color:{COR_VERDE_NVIDIA}; "
            f"font-size:0.72em; font-weight:600; margin-bottom:6px;'>"
            f"● {meta['nome']}</div>",
            unsafe_allow_html=True,
        )
        resposta = ""
        try:
            stream = gerar_resposta(
                model_id, _mensagens_para_api(), stream=True
            )
            resposta = st.write_stream(stream) or ""
        except RuntimeError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Erro inesperado: {exc}")
            return

    # 7) Persiste a resposta
    if resposta and resposta.strip():
        st.session_state.mensagens.append(
            {
                "role": "assistant",
                "content": resposta,
                "modelo_id": model_id,
            }
        )
    else:
        st.warning("⚠️ O modelo não retornou conteúdo. Tente novamente.")


# ────────────────────────────────────────────────────────────────────────────────
# 🏁 ENTRYPOINT
# ────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    _injetar_css()
    _init_state()
    _render_sidebar()
    _render_main()


if __name__ == "__main__":
    main()
