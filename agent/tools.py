"""
Agent tools for academic literature research.

Each tool is a standalone function with a clear input/output contract.
The agent can call these via function-calling, and the ToolRegistry dispatches them.
"""

import os
import re
import json
import pickle
import math
import numpy as np
from collections import Counter

# The agent will run on the server with GPU. Set RAG paths here.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ── Lazy model loading (loaded once on first use) ──

_embedder = None
_index = None
_texts = None
_metas = None
_reranker = None
_kb = None
_pmc = None
_llm_client = None


def _load_rag():
    """Lazy-load RAG infrastructure (BGE-M3, FAISS, Reranker, KB, BM25)."""
    global _embedder, _index, _texts, _metas, _reranker, _kb, _pmc, _bm25_idf, _bm25_docs
    if _embedder is not None:
        return

    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss

    _embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
    _index = faiss.read_index("vsd_rag_db/faiss_200.index")
    with open("vsd_rag_db/texts_200.pkl", "rb") as f:
        _texts = pickle.load(f)
    with open("vsd_rag_db/metas_200.pkl", "rb") as f:
        _metas = pickle.load(f)
    _reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")

    with open("data/papers_200.json", "r", encoding="utf-8") as f:
        _kb = {p["id"]: p for p in json.load(f)}

    try:
        with open("../rag_project/pmc_fulltexts.json", "r") as f:
            _pmc = {int(k): v for k, v in json.load(f).items()}
    except FileNotFoundError:
        try:
            with open("pmc_fulltexts.json", "r") as f:
                _pmc = {int(k): v for k, v in json.load(f).items()}
        except FileNotFoundError:
            _pmc = {}

    # Build light BM25 index
    def tokenize(text):
        return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(t) > 1]

    tokenized = [tokenize(t) for t in _texts]
    doc_count = len(tokenized)
    avg_dl = sum(len(t) for t in tokenized) / doc_count
    k1, b = 1.5, 0.75

    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    _bm25_idf = {term: math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    _bm25_docs = {"tokenized": tokenized, "doc_lens": [len(t) for t in tokenized],
                   "avg_dl": avg_dl, "k1": k1, "b": b}

# BM25 globals
_bm25_idf = None
_bm25_docs = None


def _get_llm():
    """Lazy-load DeepSeek client."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY","")
        _llm_client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _llm_client


# ── Tool Implementations ──

def search_papers(query: str, top_k: int = 8) -> dict:
    """
    Hybrid search: BGE-M3 FAISS + BM25 keyword + Reranker.

    Args:
        query: Natural language search query (English preferred).
        top_k: Number of results to return (default 8).

    Returns:
        dict with 'results' list of {id, title, journal, year, score, abstract_preview}.
    """
    _load_rag()

    # ── 1. FAISS vector search ──
    q_emb = _embedder.encode([query]).astype("float32")
    import faiss
    faiss.normalize_L2(q_emb)
    f_scores, f_indices = _index.search(q_emb, 50)

    vec_scores = {}  # paper_id -> best vector score
    for idx, score in zip(f_indices[0], f_scores[0]):
        pid = _metas[int(idx)]["id"]
        if pid not in vec_scores or score > vec_scores[pid]:
            vec_scores[pid] = float(score)

    # ── 2. BM25 keyword search ──
    bm25_scores = {}  # paper_id -> best BM25 score
    q_tokens = [t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 1]

    for i, doc_tokens in enumerate(_bm25_docs["tokenized"]):
        pid = _metas[i]["id"]
        score = 0.0
        tf_c = Counter(doc_tokens)
        dl = _bm25_docs["doc_lens"][i]
        for term in q_tokens:
            if term in _bm25_idf:
                tf = tf_c.get(term, 0)
                if tf > 0:
                    score += _bm25_idf[term] * tf * (_bm25_docs["k1"] + 1) / (
                        tf + _bm25_docs["k1"] * (1 - _bm25_docs["b"] + _bm25_docs["b"] * dl / _bm25_docs["avg_dl"]))
        if score > 0:
            bm25_scores[pid] = max(bm25_scores.get(pid, 0), score)

    # ── 3. Merge: normalize + weighted sum ──
    all_pids = set(vec_scores.keys()) | set(bm25_scores.keys())
    f_max = max(vec_scores.values()) if vec_scores else 1.0
    b_max = max(bm25_scores.values()) if bm25_scores else 1.0

    merged = []
    for pid in all_pids:
        vs = vec_scores.get(pid, 0) / f_max if f_max > 0 else 0
        bs = bm25_scores.get(pid, 0) / b_max if b_max > 0 else 0
        hybrid_score = 0.6 * vs + 0.4 * bs  # vector dominant, BM25 complementary
        merged.append({"id": pid, "score": hybrid_score, "vec_score": vs, "bm25_score": bs})

    merged.sort(key=lambda x: x["score"], reverse=True)
    candidates = merged[:20]

    # ── 4. Reranker precision ranking ──
    # Get text for each candidate
    text_map = {}
    for i, m in enumerate(_metas):
        if m["id"] not in text_map:
            text_map[m["id"]] = _texts[i]

    if len(candidates) > top_k:
        pairs = [(query, text_map.get(c["id"], "")) for c in candidates[:20]]
        rs = _reranker.predict(pairs)
        for c, s in zip(candidates[:20], rs):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    # ── 5. Format output ──
    results = []
    for c in candidates[:top_k]:
        pid = c["id"]
        p = _kb.get(pid, {})
        results.append({
            "id": pid,
            "title": p.get("title", ""),
            "journal": p.get("journal", ""),
            "year": p.get("year", ""),
            "score": round(c.get("rerank_score", c["score"]), 3),
            "abstract_preview": text_map.get(pid, "")[:300],
        })

    return {"query": query, "results": results, "total_found": len(results)}


def read_section(paper_id: int, section: str = "abstract") -> dict:
    """
    Read a specific section of a paper.

    Args:
        paper_id: The paper ID (integer).
        section: One of: abstract, methods, results, discussion, full.

    Returns:
        dict with paper_id, section, and content.
    """
    _load_rag()

    p = _kb.get(paper_id)
    if not p:
        return {"error": f"Paper {paper_id} not found."}

    if section == "abstract":
        content = p.get("en_abstract", p.get("summary", ""))
        return {"paper_id": paper_id, "section": "abstract",
                "title": p["title"], "content": content[:2000]}

    # For full-text sections, use PMC if available
    pmc_text = _pmc.get(paper_id, "")
    if not pmc_text:
        return {"paper_id": paper_id, "section": section,
                "title": p["title"],
                "content": f"No full text available for paper {paper_id}. Only abstract is accessible.",
                "available_sections": ["abstract"]}

    # Simple section extraction by keyword matching
    section_keywords = {
        "methods": ["method", "model architecture", "training", "data ", "dataset", "we propose",
                     "our approach", "pipeline", "framework"],
        "results": ["result", "achieved", "accuracy", "AUC", "performance", "outperform",
                     "Table", "Figure", "Dice", "sensitivity", "specificity"],
        "discussion": ["discussion", "limitation", "conclusion", "future work", "we found",
                        "our study demonstrates", "in summary"],
        "introduction": ["introduction", "background", "related work", "prior work",
                          "congenital heart disease", "deep learning has"],
    }

    if section in section_keywords:
        keywords = section_keywords[section]
        # Split text into paragraphs and find relevant ones
        paragraphs = [p.strip() for p in pmc_text.split(". ") if len(p.strip()) > 50]
        relevant = [p for p in paragraphs if any(kw.lower() in p.lower() for kw in keywords)]
        if relevant:
            content = ". ".join(relevant[:5])
            return {"paper_id": paper_id, "section": section,
                    "title": p["title"], "content": content[:2000]}
        else:
            return {"paper_id": paper_id, "section": section,
                    "title": p["title"],
                    "content": pmc_text[:2000],
                    "note": "Section not clearly delineated; returning beginning of full text."}

    return {"paper_id": paper_id, "section": section,
            "title": p["title"], "content": pmc_text[:2000]}


def extract_claims(paper_id: int) -> dict:
    """
    Extract key factual claims from a paper using LLM.

    Args:
        paper_id: The paper ID (integer).

    Returns:
        dict with paper_id, title, and a list of claims.
    """
    _load_rag()
    client = _get_llm()

    p = _kb.get(paper_id)
    if not p:
        return {"error": f"Paper {paper_id} not found."}

    abstract = p.get("en_abstract", p.get("summary", ""))
    pmc_text = _pmc.get(paper_id, "")
    text = (pmc_text or abstract)[:3000]

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": (
            "Extract 3-5 key factual claims from this academic paper excerpt. "
            "Each claim should be one sentence, specific, and verifiable. "
            "Focus on: methodology, quantitative results, limitations, datasets used.\n\n"
            f"Paper ID {paper_id}: {p['title']}\n\n{text}"
        )}],
        temperature=0.1, max_tokens=300,
    )

    claims_text = resp.choices[0].message.content.strip()
    claims = [c.strip("- ").strip() for c in claims_text.split("\n") if c.strip()]

    return {
        "paper_id": paper_id,
        "title": p["title"],
        "claims": claims,
    }


def compare_papers(paper_ids: list[int], dimension: str = "methodology") -> dict:
    """
    Compare multiple papers on a specific dimension.

    Args:
        paper_ids: List of paper IDs to compare (max 5).
        dimension: Comparison dimension (methodology, results, limitations, data, contributions).

    Returns:
        dict with comparison results.
    """
    _load_rag()
    client = _get_llm()

    if len(paper_ids) > 5:
        paper_ids = paper_ids[:5]

    papers_text = []
    for pid in paper_ids:
        p = _kb.get(pid)
        if p:
            ab = p.get("en_abstract", "")[:500]
            papers_text.append(f"[{pid}] {p['title']} ({p['journal']}, {p['year']}): {ab}")

    if not papers_text:
        return {"error": "No valid paper IDs provided."}

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": (
            f"Compare the following papers on the dimension: {dimension}. "
            "Create a structured comparison. Be specific about differences.\n\n"
            + "\n\n".join(papers_text)
        )}],
        temperature=0.2, max_tokens=500,
    )

    return {
        "dimension": dimension,
        "papers_compared": paper_ids,
        "comparison": resp.choices[0].message.content.strip(),
    }


def verify_claim(claim: str, paper_id: int) -> dict:
    """
    Verify whether a specific claim is supported by a paper's content.

    Args:
        claim: The claim to verify (one sentence).
        paper_id: The paper ID to check against.

    Returns:
        dict with verdict (supported/partially_supported/not_supported/uncertain) and evidence.
    """
    _load_rag()
    client = _get_llm()

    p = _kb.get(paper_id)
    if not p:
        return {"error": f"Paper {paper_id} not found."}

    abstract = p.get("en_abstract", "")
    pmc_text = _pmc.get(paper_id, "")
    text = (pmc_text or abstract)[:3000]

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": (
            "Verify whether the following claim is supported by the paper excerpt. "
            "Respond with: supported, partially_supported, not_supported, or uncertain. "
            "Then provide the specific evidence (quote or paraphrase) from the text.\n\n"
            f"Claim: {claim}\n\n"
            f"Paper [{paper_id}] {p['title']}:\n{text}"
        )}],
        temperature=0.0, max_tokens=200,
    )

    response = resp.choices[0].message.content.strip()

    # Parse verdict
    verdict = "uncertain"
    for v in ["supported", "partially_supported", "not_supported", "uncertain"]:
        if v in response.lower():
            verdict = v
            break

    return {
        "claim": claim,
        "paper_id": paper_id,
        "verdict": verdict,
        "evidence": response,
    }


# ── Tool Schema Definitions (for OpenAI function calling) ──

TOOL_SCHEMAS = {
    "search_papers": {
        "description": "Search the academic paper database for papers relevant to a query. "
                       "Use this as your first action for any new subtopic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query in English. Be specific with technical terms."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 8, max 10)."
                },
            },
            "required": ["query"],
        },
    },
    "read_section": {
        "description": "Read a specific section of a paper (abstract, methods, results, discussion, or full). "
                       "Use this when search results are insufficient and you need more detail.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "integer",
                    "description": "The paper ID number."
                },
                "section": {
                    "type": "string",
                    "enum": ["abstract", "methods", "results", "discussion", "introduction", "full"],
                    "description": "Which section to read."
                },
            },
            "required": ["paper_id", "section"],
        },
    },
    "extract_claims": {
        "description": "Extract key factual claims from a paper. "
                       "Use this before citing a paper to ensure you have its claims right.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "integer",
                    "description": "The paper ID number."
                },
            },
            "required": ["paper_id"],
        },
    },
    "compare_papers": {
        "description": "Compare multiple papers on a specific dimension (methodology, results, limitations, data, contributions).",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of paper IDs to compare (max 5)."
                },
                "dimension": {
                    "type": "string",
                    "enum": ["methodology", "results", "limitations", "data", "contributions"],
                    "description": "Dimension to compare on."
                },
            },
            "required": ["paper_ids", "dimension"],
        },
    },
    "verify_claim": {
        "description": "Verify whether a specific claim is supported by a paper. "
                       "Use this before including a claim in your final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": "The exact claim to verify (one sentence)."
                },
                "paper_id": {
                    "type": "integer",
                    "description": "The paper ID to verify against."
                },
            },
            "required": ["claim", "paper_id"],
        },
    },
}


def register_all_tools(registry) -> None:
    """Register all 5 tools into a ToolRegistry instance.

    Args:
        registry: ToolRegistry instance from agent_core.py.
    """
    registry.register(
        "search_papers", search_papers,
        TOOL_SCHEMAS["search_papers"]["description"],
        TOOL_SCHEMAS["search_papers"]["parameters"],
    )
    registry.register(
        "read_section", read_section,
        TOOL_SCHEMAS["read_section"]["description"],
        TOOL_SCHEMAS["read_section"]["parameters"],
    )
    registry.register(
        "extract_claims", extract_claims,
        TOOL_SCHEMAS["extract_claims"]["description"],
        TOOL_SCHEMAS["extract_claims"]["parameters"],
    )
    registry.register(
        "compare_papers", compare_papers,
        TOOL_SCHEMAS["compare_papers"]["description"],
        TOOL_SCHEMAS["compare_papers"]["parameters"],
    )
    registry.register(
        "verify_claim", verify_claim,
        TOOL_SCHEMAS["verify_claim"]["description"],
        TOOL_SCHEMAS["verify_claim"]["parameters"],
    )
