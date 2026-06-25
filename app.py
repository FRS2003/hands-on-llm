"""VSD Literature Assistant — RAG-powered Q&A with Streamlit.
Deployed on HuggingFace Spaces.
"""
import streamlit as st
import os, pickle, numpy as np

st.set_page_config(page_title="VSD Literature Assistant", page_icon="📚", layout="wide")

# Paths relative to repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "vsd_rag_db")


@st.cache_resource
def load_retriever():
    """Load embedding model and FAISS index. Cached across sessions."""
    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    index = faiss.read_index(os.path.join(DB_DIR, "faiss_v5cn.index"))
    with open(os.path.join(DB_DIR, "texts_v5cn.pkl"), "rb") as f:
        texts = pickle.load(f)
    with open(os.path.join(DB_DIR, "metas_v5cn.pkl"), "rb") as f:
        metas = pickle.load(f)
    return model, index, texts, metas


def retrieve(model, index, texts, metas, query, top_k=5):
    """Two-stage retrieval: FAISS coarse search + paper-level dedup."""
    q_emb = model.encode([query]).astype("float32")
    import faiss
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k * 3)
    seen, results = set(), []
    for idx, score in zip(indices[0], scores[0]):
        pid = metas[idx]["id"]
        if pid not in seen:
            seen.add(pid)
            results.append({
                "id": pid,
                "title": metas[idx]["title"],
                "journal": metas[idx]["journal"],
                "year": metas[idx]["year"],
                "text": texts[idx][:500],
                "score": float(score),
            })
        if len(results) >= top_k:
            break
    return results


# ── UI ──
st.title("📚 VSD Literature Assistant")
st.caption(
    "RAG-powered Q&A over 38 papers on congenital heart disease & deep learning | "
    "[GitHub](https://github.com) · Built with FAISS + BGE + Streamlit"
)

model, index, texts, metas = load_retriever()

query = st.text_input(
    "Ask a question about VSD, echocardiography, or deep learning:",
    placeholder="e.g., What are the challenges in VSD subtype classification?",
)

if query:
    with st.spinner("Searching literature..."):
        results = retrieve(model, index, texts, metas, query)

    # ── Retrieved Papers ──
    st.markdown("### 📄 Retrieved Papers")
    cols = st.columns(len(results))
    for i, (col, r) in enumerate(zip(cols, results)):
        with col:
            score_color = (
                "green" if r["score"] > 0.6
                else "orange" if r["score"] > 0.45
                else "red"
            )
            st.markdown(f"**#{i + 1}** :{score_color}[{r['score']:.3f}]")
            st.markdown(f"**{r['title']}**")
            st.caption(f"{r['journal']}, {r['year']}")

    # ── Answer Section ──
    st.markdown("---")
    st.markdown("### 📝 AI Answer")

    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[{i + 1}] {r['title']} ({r['journal']}, {r['year']}): {r['text'][:300]}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a VSD (ventricular septal defect) deep learning research assistant.
Based on the following papers, answer the user's question. Cite sources using [1][2] notation.

PAPERS:
{context}

QUESTION: {query}

Answer concisely with citations. If the papers don't cover the question, say so."""

    # Try DeepSeek API if key provided
    api_key = st.session_state.get("api_key", "")
    if api_key:
        with st.spinner("Generating answer via DeepSeek..."):
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=800,
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
            except Exception as e:
                st.error(f"API call failed: {e}")
    else:
        st.info("💡 Enter your DeepSeek API key in the sidebar to enable live AI answers.")
        with st.expander("📋 Click to copy the prompt (ready for manual use)"):
            st.code(prompt, language="text")

    # ── Paper Details ──
    st.markdown("---")
    st.markdown("### 🔍 Paper Details")
    for i, r in enumerate(results):
        with st.expander(f"[{i + 1}] {r['title'][:80]}... (score: {r['score']:.3f})"):
            st.markdown("**Full Text Excerpt:**")
            st.text(r["text"])
            st.markdown(f"**Journal:** {r['journal']} ({r['year']})")

    # ── Debug ──
    with st.expander("🔧 Debug: Full Prompt"):
        st.code(prompt, language="text")


# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        help="Get one at https://platform.deepseek.com",
    )
    if api_key:
        st.session_state["api_key"] = api_key
        st.success("API key set ✓")

    st.header("📖 About")
    st.markdown("""
    **VSD Literature Assistant** is a RAG (Retrieval-Augmented Generation) system
    built on **38 papers** covering:
    - VSD clinical guidelines & epidemiology
    - Deep learning in echocardiography
    - Self-supervised learning (DINOv2)
    - Data leakage methodology
    - Biomedical signal processing

    **Tech Stack:**
    - 🗂️ FAISS vector search (512-dim, IP)
    - 🧠 BGE-small-zh-v1.5 embeddings
    - 📝 Chinese full-text chunked encoding
    - 🎈 Streamlit frontend
    - 🤖 DeepSeek API (optional)

    **Author:** Graduate student project — CV → AI career transition
    """)

    st.header("🔬 Test Queries")
    st.markdown("""
    Try these:
    - *DINOv2在医学影像中有什么优势？*
    - *VSD亚型分类的主要挑战是什么？*
    - *数据泄漏对深度学习研究有什么影响？*
    - *Cheng等2024年的工作有什么局限性？*
    - *Gao等2025年的BSPC论文和我们的研究有什么区别？*
    """)

    st.caption("Built with ❤️ using FAISS + BGE + Streamlit")
