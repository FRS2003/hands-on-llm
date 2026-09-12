"""
Build dual-index for 208 papers:

Index A (coarse): Title + first 1-2 sentences of abstract (~150 chars).
                  High signal density, fast narrowing to top candidates.
Index B (fine):   Full abstract split into 2-3 chunks (~500 chars each).
                  More content for precision matching within candidates.

Both use BGE-M3 1024-dim, stored in separate FAISS indexes.
Index B vectors carry paper_id for filtered search.
"""
import os, json, pickle, numpy as np, re
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("agent")

print("Loading papers...")
with open("data/papers_200.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
print(f"Papers: {len(papers)}")

print("Loading BGE-M3 on GPU...")
from sentence_transformers import SentenceTransformer
import faiss
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
dim = embedder.get_sentence_embedding_dimension()

# ── Build Index A: Title + first 1-2 sentences ──
texts_a, metas_a = [], []
for p in papers:
    ab = p.get("en_abstract") or p.get("summary") or ""
    # Take first 2 sentences (~150-200 chars)
    sentences = re.split(r"(?<=[.!?])\s+", ab)
    short_ab = " ".join(sentences[:2])[:250]

    text_a = (
        f"Title: {p['title']}. "
        f"Authors: {p.get('authors', '')}. "
        f"Journal: {p.get('journal', '')} ({p.get('year', '')}). "
        f"Summary: {short_ab}"
    )
    texts_a.append(text_a)
    metas_a.append({"id": p["id"], "title": p["title"],
                     "journal": p.get("journal", ""), "year": p.get("year", 0),
                     "type": "coarse"})

print(f"Index A: {len(texts_a)} vectors")

emb_a = embedder.encode(texts_a, show_progress_bar=True, batch_size=16)
emb_a = np.array(emb_a).astype("float32")
faiss.normalize_L2(emb_a)
index_a = faiss.IndexFlatIP(dim)
index_a.add(emb_a)

# ── Build Index B: Full abstract split into ~500-char chunks ──
texts_b, metas_b = [], []
for p in papers:
    ab = p.get("en_abstract") or p.get("summary") or ""
    # Split into ~500 char chunks at sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", ab)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < 500:
            current += s + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s + " "
    if current.strip():
        chunks.append(current.strip())

    if not chunks:
        chunks = [ab[:500]]

    for ci, chunk in enumerate(chunks):
        text_b = (
            f"[Paper {p['id']}] {p['title']}. "
            f"{p.get('authors', '')}. "
            f"Abstract part {ci+1}/{len(chunks)}: {chunk}"
        )
        texts_b.append(text_b)
        metas_b.append({"id": p["id"], "title": p["title"],
                         "journal": p.get("journal", ""), "year": p.get("year", 0),
                         "chunk_id": ci, "total_chunks": len(chunks), "type": "fine"})

avg_chunks = len(texts_b) / len(papers)
print(f"Index B: {len(texts_b)} vectors ({avg_chunks:.1f} chunks/paper)")

emb_b = embedder.encode(texts_b, show_progress_bar=True, batch_size=16)
emb_b = np.array(emb_b).astype("float32")
faiss.normalize_L2(emb_b)
index_b = faiss.IndexFlatIP(dim)
index_b.add(emb_b)

# ── Save ──
os.makedirs("vsd_rag_db", exist_ok=True)
faiss.write_index(index_a, "vsd_rag_db/faiss_a.index")
faiss.write_index(index_b, "vsd_rag_db/faiss_b.index")
with open("vsd_rag_db/texts_a.pkl", "wb") as f: pickle.dump(texts_a, f)
with open("vsd_rag_db/metas_a.pkl", "wb") as f: pickle.dump(metas_a, f)
with open("vsd_rag_db/texts_b.pkl", "wb") as f: pickle.dump(texts_b, f)
with open("vsd_rag_db/metas_b.pkl", "wb") as f: pickle.dump(metas_b, f)

print(f"\nSaved:")
print(f"  faiss_a.index: {index_a.ntotal} vectors (coarse)")
print(f"  faiss_b.index: {index_b.ntotal} vectors (fine)")

# ── Quick comparison test ──
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")

def single_search(query, top_k=5):
    """Single-index: search Index A only (like current RAG)."""
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index_a.search(q_emb, 20)
    candidates, seen = [], set()
    for idx, score in zip(indices[0], scores[0]):
        pid = metas_a[int(idx)]["id"]
        if pid not in seen:
            seen.add(pid)
            candidates.append({"id": pid, "score": float(score), "text": texts_a[int(idx)]})
        if len(candidates) >= 20: break
    pairs = [(query, c["text"]) for c in candidates[:20]]
    rs = reranker.predict(pairs)
    for c, s in zip(candidates[:20], rs): c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
    seen2, final = set(), []
    for c in candidates:
        if c["id"] not in seen2: seen2.add(c["id"]); final.append(c)
        if len(final) >= top_k: break
    return final

def dual_search(query, top_k=5, coarse_k=30):
    """Dual-index: Index A coarse → Index B fine for top candidates."""
    # Step 1: Coarse search on Index A
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index_a.search(q_emb, coarse_k)
    candidate_ids = set()
    for idx in indices[0]:
        candidate_ids.add(metas_a[int(idx)]["id"])

    # Step 2: Fine search on Index B, filtered to candidate IDs
    scores_b, indices_b = index_b.search(q_emb, 100)
    fine_candidates, seen = [], set()
    for idx, score in zip(indices_b[0], scores_b[0]):
        pid = metas_b[int(idx)]["id"]
        if pid in candidate_ids and pid not in seen:
            seen.add(pid)
            fine_candidates.append({"id": pid, "score": float(score), "text": texts_b[int(idx)]})
        if len(fine_candidates) >= 20: break

    # Reranker
    if len(fine_candidates) > top_k:
        pairs = [(query, c["text"]) for c in fine_candidates[:20]]
        rs = reranker.predict(pairs)
        for c, s in zip(fine_candidates[:20], rs): c["rerank_score"] = float(s)
        fine_candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    seen2, final = set(), []
    for c in fine_candidates:
        if c["id"] not in seen2: seen2.add(c["id"]); final.append(c)
        if len(final) >= top_k: break
    return final

# Test queries
test_queries = [
    ("DINOv2 medical imaging experiments", [21]),
    ("EchoNet-Dynamic patient-level data leakage", [10]),
    ("Cheng 2024 study limitations", [12]),
    ("data augmentation VSD ultrasound classification", [14]),
    ("self-supervised learning advantages medical imaging", [19, 20]),
    ("transformer architecture echocardiography analysis", []),  # any relevant hit
]

print("\n" + "=" * 70)
print("Single-index vs Dual-index 对比测试")
print("=" * 70)

for q, expected in test_queries:
    single = single_search(q, top_k=5)
    dual = dual_search(q, top_k=5)
    s_ids = [r["id"] for r in single[:3]]
    d_ids = [r["id"] for r in dual[:3]]
    exp_hit_s = bool(set(s_ids) & set(expected)) if expected else "N/A"
    exp_hit_d = bool(set(d_ids) & set(expected)) if expected else "N/A"
    status = "BETTER" if exp_hit_d and not exp_hit_s else ("SAME" if exp_hit_d == exp_hit_s else "WORSE")
    print(f"{status:7s} | {q[:45]}...")
    print(f"         Single: {s_ids}")
    print(f"         Dual:   {d_ids}")
    print()

print("Done!")
