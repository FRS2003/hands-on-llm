"""VSD 文献智能助手 v2 — 聊天界面 + 多轮对话 + BGE-M3"""
import streamlit as st
import os, pickle, numpy as np

st.set_page_config(page_title="VSD 文献智能助手", page_icon="📚", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "vsd_rag_db")

# ── 聊天气泡 ──
st.markdown("""
<style>
.user-msg { background: #e3f2fd; border-radius: 12px 12px 4px 12px;
    padding: 10px 16px; margin: 8px 0 8px auto; max-width: 75%; text-align: right; }
.ast-msg  { background: #f5f5f5; border-radius: 12px 12px 12px 4px;
    padding: 10px 16px; margin: 8px auto 8px 0; max-width: 85%; }
.msg-label { font-size: 0.8em; color: #888; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

# ── 模型加载 ──
@st.cache_resource
def load_models():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from sentence_transformers import SentenceTransformer
    import faiss

    embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    index = faiss.read_index(os.path.join(DB_DIR, "faiss_en.index"))
    with open(os.path.join(DB_DIR, "texts_en.pkl"), "rb") as f:
        texts = pickle.load(f)
    with open(os.path.join(DB_DIR, "metas_en.pkl"), "rb") as f:
        metas = pickle.load(f)

    reranker = None
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
    except Exception:
        pass

    return embedder, index, texts, metas, reranker


def retrieve(embedder, index, texts, metas, reranker, query, top_k=5, oversample=20):
    import faiss
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, oversample)

    candidates, seen = [], set()
    for idx, score in zip(indices[0], scores[0]):
        pid = metas[idx]["id"]
        if pid not in seen:
            seen.add(pid)
            candidates.append({
                "id": pid, "title": metas[idx]["title"],
                "journal": metas[idx]["journal"], "year": metas[idx]["year"],
                "text": texts[idx], "score": float(score),
            })
        if len(candidates) >= oversample:
            break

    if reranker and len(candidates) > top_k:
        pairs = [(query, c["text"]) for c in candidates]
        rs = reranker.predict(pairs)
        for c, s in zip(candidates, rs):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    return candidates[:top_k]


# ── 初始化 ──
if "messages" not in st.session_state:
    st.session_state.messages = []

embedder, index, texts, metas, reranker = load_models()

# ── 侧边栏 ──
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input(
        "DeepSeek API Key", type="password",
        value=st.session_state.get("api_key_ui", ""),
        help="在 https://platform.deepseek.com 获取",
    )
    if api_key:
        st.session_state["api_key"] = api_key
        st.session_state["api_key_ui"] = api_key
        st.success("✅ 已设置")

    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.header("📖 关于")
    st.markdown("""
    **VSD 文献智能助手**基于 RAG 技术，覆盖 **38 篇**先心病深度学习论文。

    **技术栈：**
    - 🧠 BGE-M3 嵌入（1024维）
    - 🗂️ FAISS 向量检索
    - 🎯 BGE-Reranker 精排
    - 🤖 DeepSeek API（流式输出）
    - 🎈 Streamlit
    """)
    st.caption("BGE-M3 + FAISS + DeepSeek 构建")


# ── 主界面 ──
st.title("📚 VSD 文献智能助手")
st.caption("基于 38 篇先天性心脏病深度学习论文 · 支持多轮对话 · 中英文检索")

# 消息容器
msg_container = st.container()

# 渲染历史
with msg_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        css_class = "user-msg" if role == "user" else "ast-msg"
        label = "🧑 你" if role == "user" else "🤖 助手"
        st.markdown(
            f'<div class="{css_class}"><div class="msg-label">{label}</div>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        if msg.get("sources") and role == "assistant":
            with st.expander("📄 参考来源"):
                for j, s in enumerate(msg["sources"]):
                    sc = f"{s.get('rerank_score', s['score']):.3f}"
                    st.markdown(
                        f"**[{j + 1}] {s['title']}** "
                        f"（{s['journal']}, {s['year']}）· 相关度 {sc}"
                    )

# 输入
query = st.chat_input("请输入您的问题……")

if query:
    # 1. 存用户消息
    st.session_state.messages.append({"role": "user", "content": query})

    # 2. 检索
    with st.spinner("🔍 检索文献中……"):
        results = retrieve(embedder, index, texts, metas, reranker, query)

    # 3. 构建 prompt
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[{i + 1}] {r['title']}（{r['journal']}, {r['year']}）：{r['text'][:400]}"
        )
    context = "\n\n".join(context_parts)

    history_parts = []
    recent = st.session_state.messages[:-1][-6:]
    for m in recent:
        role = "用户" if m["role"] == "user" else "助手"
        history_parts.append(f"{role}：{m['content'][:200]}")
    history = "\n".join(history_parts) if history_parts else "（无历史对话）"

    prompt = f"""你是 VSD（室间隔缺损）深度学习研究助手。请根据以下论文内容回答用户问题。

对话历史：
{history}

相关论文：
{context}

用户问题：{query}

请用中文回答，标注引用来源如 [1][2]。如果论文无法回答，请如实说明。"""

    # 4. 生成回答
    api_key = st.session_state.get("api_key", "")
    if api_key:
        with st.spinner("🤖 正在生成回答……"):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

                # 流式收集（后台收完再显示，避免 DOM 冲突）
                stream = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=800, stream=True,
                )
                full_answer = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        full_answer += chunk.choices[0].delta.content

                st.session_state.messages.append({
                    "role": "assistant", "content": full_answer, "sources": results,
                })
                st.rerun()

            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", "content": f"❌ API 调用失败：{e}", "sources": results,
                })
                st.rerun()
    else:
        no_key_parts = [
            "💡 **未设置 API Key**\n\n在左侧输入 DeepSeek API Key 即可启用 AI 回答。\n",
            f"---\n**检索结果（Top-{len(results)}）：**",
        ]
        for j, r in enumerate(results):
            no_key_parts.append(f"\n**[{j + 1}] {r['title']}** · 相关度 {r['score']:.3f}")

        st.session_state.messages.append({
            "role": "assistant", "content": "\n".join(no_key_parts), "sources": results,
        })
        st.rerun()
