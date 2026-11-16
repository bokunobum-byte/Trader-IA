"""
app.py — Interface Streamlit principal do Tarder IA.

Responsabilidades:
  • Carregar variáveis de ambiente (.env local)
  • Aplicar CSS escuro moderno (verde NVIDIA + azul)
  • Renderizar sidebar (logo, seletor de modelo, tokens, limpar)
  • Renderizar chat com histórico, streaming e upload DENTRO do input
  • Enviar arquivos junto com a mensagem para o modelo ativo
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

# Lista unificada de extensões aceitas pelo chat_input.
# Aceita TODOS os tipos suportados pelo app; a validação por modelo é feita
# no envio (o modelo ativo pode rejeitar imagens, por exemplo).
_EXTENSOES_ACEITAS: List[str] = (
    EXTENSOES_TEXTO + EXTENSOES_IMAGEM + EXTENSOES_PDF
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
        /* Input do chat + botão de anexo */
        [data-testid="stChatInput"] {{
            background-color: #1e293b !important;
            border-radius: 14px !important;
            border: 1px solid #334155 !important;
        }}
        [data-testid="stChatInput"] textarea {{
            background-color: transparent !important;
            color: {COR_TEXTO} !important;
        }}
        /* Chips de anexos nas mensagens */
        .anexo-chip {{
            display: inline-block;
            background-color: #33415580;
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 4px 10px;
            margin: 3px 3px 0 0;
            font-size: 0.8em;
            color: {COR_TEXTO};
        }}
        /* Títulos e código */
        h1, h2, h3 {{ color: {COR_VERDE_NVIDIA}; }}
        code {{
            background-color: #0b1220;
            color: {COR_VERDE_NVIDIA};
            padding: 2px 6px;
            border-radius: 4px;
        }}
        pre code {{ background-color: #0b1220 !important; }}
        a {{ color: {COR_AZUL_ACENTO}; }}
        /* Métrica (token counter) */
        [data-testid="stMetricValue"] {{ color: {COR_VERDE_NVIDIA} !important; }}
        [data-testid="stMetricLabel"] {{ color: #94a3b8 !important; }}
        hr {{ border-color: #1e293b; }}
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
        # Cada item: {"role", "content", "display_text"?, "anexos_info"?, "modelo_id"?}
        st.session_state.mensagens = []

    if "modelo_id" not in st.session_state:
        st.session_state.modelo_id = list(MODELOS.keys())[0]


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
    """Tokens do histórico (inclui imagens e textos dos anexos já enviados)."""
    return estimar_tokens_mensagens(_mensagens_para_api())


def _historico_tem_imagem() -> bool:
    """True se alguma mensagem anterior já contém imagem multimodal."""
    for msg in st.session_state.mensagens:
        content = msg.get("content")
        if isinstance(content, list):
            for parte in content:
                if isinstance(parte, dict) and parte.get("type") == "image_url":
                    return True
    return False


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

        # Aviso: histórico tem imagem, modelo atual não aceita
        if _historico_tem_imagem() and "imagem" not in _modelo_atual()["formatos"]:
            st.warning(
                "O histórico contém imagens, mas o modelo atual não as aceita. "
                "Limpe a conversa ou troque para um modelo multimodal."
            )

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

    meta = _modelo_atual()
    st.caption(meta["descricao"])
    st.caption(
        f"Contexto: **{meta['max_tokens_contexto']:,}** tokens · "
        f"Saída máx.: **{meta['max_tokens_saida']:,}**"
    )


def _render_token_counter() -> None:
    """Mostra tokens usados vs. limite e avisa perto do teto."""
    meta = _modelo_atual()
    limite = meta["max_tokens_contexto"]
    total = _estimar_tokens_totais()
    pct = (total / limite) if limite else 0.0

    st.markdown("##### 🧮 Contexto")
    st.metric(
        label="Tokens no histórico",
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


def _render_anexos_da_mensagem(anexos_info: List[Dict[str, Any]]) -> None:
    """Mostra chips com os anexos enviados junto à mensagem."""
    if not anexos_info:
        return
    chips = []
    for a in anexos_info:
        emoji = {"texto": "📄", "pdf": "📕", "imagem": "🖼️"}.get(a["tipo"], "📎")
        tamanho = tamanho_legivel(int(a["tamanho"]))
        chips.append(
            f"<span class='anexo-chip'>{emoji} {a['nome']} · {tamanho}</span>"
        )
    st.markdown(
        "<div style='margin-top:8px;'>" + "".join(chips) + "</div>",
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
            _render_anexos_da_mensagem(msg.get("anexos_info") or [])
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
                    Escolha um modelo na barra lateral, digite sua mensagem e
                    anexe arquivos pelo 📎 dentro da caixa de texto.
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

    # Dica de uso (só enquanto estiver vazio o histórico)
    if not st.session_state.mensagens:
        st.caption(
            f"💡 Formatos aceitos: **{', '.join(e.lstrip('.').upper() for e in _EXTENSOES_ACEITAS)}** "
            f"· máx. **{tamanho_legivel(MAX_FILE_SIZE_BYTES)}** por arquivo"
        )

    # Input do chat COM upload integrado (📎 dentro da caixa).
    #
    # IMPORTANTE: no st.chat_input o parâmetro correto é accept_file="multiple".
    # NÃO use accept_multiple_files=True — esse parâmetro só existe no
    # st.file_uploader e gera TypeError no chat_input.
    resultado = st.chat_input(
        "Digite sua mensagem ou anexe um arquivo...",
        accept_file="multiple",
        file_type=[e.lstrip(".") for e in _EXTENSOES_ACEITAS],
    )
    if resultado:
        _processar_envio(resultado)


# ────────────────────────────────────────────────────────────────────────────────
# 🚀 ENVIO DE MENSAGEM
# ────────────────────────────────────────────────────────────────────────────────


def _extrair_texto_e_arquivos(resultado: Any) -> tuple[str, List[Any]]:
    """Compatível com str (Streamlit antigo) e ChatInputValue (>=1.40)."""
    if isinstance(resultado, str):
        return resultado, []
    texto = getattr(resultado, "text", "") or ""
    arquivos = getattr(resultado, "files", None) or []
    return texto, list(arquivos)


def _processar_envio(resultado: Any) -> None:
    """Processa um envio do chat_input (texto + arquivos opcionais)."""
    prompt, arquivos = _extrair_texto_e_arquivos(resultado)
    if not prompt.strip() and not arquivos:
        return

    model_id = st.session_state.modelo_id
    meta = MODELOS[model_id]

    # 1) Verificação antecipada de chave
    try:
        obter_api_key()
    except ChaveAusenteError as exc:
        st.error(str(exc))
        return

    # 2) Processa cada arquivo em memória
    anexos: List[Dict[str, Any]] = []
    for arq in arquivos:
        try:
            bytes_arq = arq.read()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Falha ao ler **{arq.name}**: {exc}")
            continue

        proc = processar_upload(arq.name, bytes_arq)
        if proc["erro"]:
            st.error(f"**{proc['nome']}**: {proc['erro']}")
            continue
        anexos.append(proc)

    # 3) Incompatibilidade imagem/modelo (considera envio atual E histórico)
    tem_imagem_novo = any(a["tipo"] == "imagem" for a in anexos)
    if (tem_imagem_novo or _historico_tem_imagem()) and "imagem" not in meta["formatos"]:
        st.error(
            f"**{meta['nome']}** não aceita imagens. "
            "Remova o arquivo, troque para um modelo multimodal, "
            "ou limpe a conversa se o histórico tiver imagens."
        )
        return

    # 4) Monta content multimodal (texto + arquivos)
    try:
        content = montar_content(
            prompt,
            anexos,
            modelo_aceita_imagem=("imagem" in meta["formatos"]),
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erro ao montar conteúdo: {exc}")
        return

    # 5) Registra a mensagem do usuário (preserva texto + info dos anexos)
    user_msg: Dict[str, Any] = {
        "role": "user",
        "content": content,
        "display_text": prompt or "(anexo enviado)",
        "anexos_info": [
            {"nome": a["nome"], "tipo": a["tipo"], "tamanho": a["tamanho"]}
            for a in anexos
        ],
    }
    st.session_state.mensagens.append(user_msg)

    # 6) Renderiza a mensagem do usuário
    _render_mensagem(user_msg)

    # 7) Chama a API com streaming
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

    # 8) Persiste a resposta
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
