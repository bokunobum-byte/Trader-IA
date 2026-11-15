# 🤖 Tarder IA

Interface web moderna e leve para conversar com **múltiplos modelos de IA da NVIDIA NIM**, com upload de arquivos, streaming de respostas e seletor de modelo dinâmico.

> **Stack:** Streamlit · OpenAI SDK (compatível com NVIDIA) · pypdf · tiktoken  
> **Deploy:** otimizado para instância gratuita do Render (512 MB RAM)

---

## 📸 Visão geral

- 🎯 **Seletor de modelo** com 6 modelos NVIDIA (DeepSeek, Mistral, Nemotron, MiniMax, Step, GLM)
- 📎 **Upload inteligente** — texto, imagens e PDFs; formatos aceitos se adaptam ao modelo ativo
- 💬 **Chat com histórico** e streaming em tempo real (`st.write_stream`)
- 🧮 **Contador de tokens** com alertas em 60% e 85% do contexto
- 🎨 **Design escuro moderno** — verde NVIDIA (`#76b900`) + azul (`#3b82f6`)
- 🔒 **Chave da API nunca sai do backend** — sempre via variável de ambiente

---

## 🧩 Modelos disponíveis

| ID técnico | Nome | Formatos | Contexto | Saída |
|---|---|---|---|---|
| `deepseek-ai/deepseek-v4.1-flash` | DeepSeek V4.1 Flash | texto, imagem | 1.048.576 | 8.192 |
| `mistralai/mistral-large-4-1t-a52b` | Mistral Large 4 | texto, imagem | 128.000 | 4.096 |
| `nvidia/nemotron-3.5-bolt-e4b` | Nemotron 3.5 Bolt | texto | 131.072 | 1.024 |
| `minimaxai/minimax-m3.1-flash` | MiniMax M3.1 Flash | texto, imagem | 200.000 | 16.384 |
| `stepfun-ai/step-5-preview` | Step 5 Preview | texto, imagem | 128.000 | 4.096 |
| `z-ai/glm-5.3-flash` | GLM 5.3 Flash | texto | 128.000 | 1.024 |

> ⚠️ Modelos sem "imagem" na lista de formatos **não aceitam upload de imagem**. A UI avisa automaticamente.

---

## 🚀 Rodar localmente

### 1. Clonar e criar ambiente virtual

**Windows (PowerShell ou CMD):**

```powershell
git clone https://github.com/<seu-usuario>/tarder-ia.git
cd tarder-ia

python -m venv venv
venv\Scripts\activate