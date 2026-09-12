"""Generate LLM summaries from PMC full text and rebuild index."""
import os, json, pickle, numpy as np, re, time
os.chdir("data")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")

with open("vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
with open("pmc_fulltexts.json", "r") as f:
    pmc = {int(k): v for k, v in json.load(f).items()}

print(f"PMC papers: {len(pmc)}, Total papers: {len(papers)}")

summarized = 0
for p in papers:
    pid = p["id"]
    ft = pmc.get(pid, "")
    if not ft or len(ft) < 500:
        continue

    title = p["title"]
    print(f"[{pid}] Summarizing: {title[:50]}...", end=" ", flush=True)

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": (
                "Summarize this academic paper in 3-5 English sentences. "
                "Include: (1) the research problem, (2) method used, (3) key quantitative results, "
                "(4) any limitations or shortcomings, (5) what datasets were used and whether they are public. "
                "Use concise academic English. Only output the summary.\n\n"
                f"Title: {title}\n"
                f"Content (excerpt): {ft[:3000]}"
            )}],
            temperature=0.2, max_tokens=300,
        )
        summary = resp.choices[0].message.content.strip()
        # Append as LLM-generated enrichment
        old_ab = p.get("en_abstract", "")
        p["en_abstract"] = f"{old_ab}\n[LLM Summary] {summary}"
        p["abstract_source"] = p.get("abstract_source", "?") + "_llm"
        print(f"OK ({len(summary)} chars)")
        summarized += 1
    except Exception as e:
        print(f"ERR: {e}")

    time.sleep(0.5)

print(f"\nSummarized: {summarized} papers")

# Save
with open("vsd_rag_enriched.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)

# ── Rebuild FAISS ──
print("Rebuilding index...")
from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")

all_texts, all_metas = [], []
for p in papers:
    text = (
        f"Title: {p['title']}. Authors: {p['authors']}. "
        f"Journal: {p['journal']} ({p['year']}). "
        f"Abstract: {p.get('en_abstract', '')}"
    )
    all_texts.append(text)
    all_metas.append({
        "id": p["id"], "title": p["title"], "journal": p["journal"],
        "year": p["year"], "chunk_id": 0, "total_chunks": 1,
    })

cn = sum(1 for t in all_texts if re.search(r"[一-鿿]", t))
print(f"Chinese content: {cn}/38")

embeddings = embedder.encode(all_texts, show_progress_bar=True, batch_size=8)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(1024)
index.add(embeddings)

faiss.write_index(index, "vsd_rag_db/faiss_llm.index")
with open("vsd_rag_db/texts_llm.pkl", "wb") as f: pickle.dump(all_texts, f)
with open("vsd_rag_db/metas_llm.pkl", "wb") as f: pickle.dump(all_metas, f)
print(f"Saved: faiss_llm.index ({index.ntotal} vectors)")

# ── Verify the 3 critical queries ──
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")

def search(q, top_k=5):
    q_emb = embedder.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 30)
    candidates, seen = [], set()
    for idx, score in zip(indices[0], scores[0]):
        pid = all_metas[int(idx)]["id"]
        if pid not in seen:
            seen.add(pid)
            candidates.append({"id": pid, "score": float(score), "text": all_texts[int(idx)]})
        if len(candidates) >= 30: break

    pairs = [(q, c["text"]) for c in candidates[:20]]
    rs = reranker.predict(pairs)
    for c, s in zip(candidates[:20], rs): c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    seen2, final = set(), []
    for c in candidates:
        if c["id"] not in seen2: seen2.add(c["id"]); final.append(c)
        if len(final) >= top_k: break
    return final

critical = [
    ("Cheng 2024 limitations", [12]),
    ("VSD classification public datasets", [10, 33]),
    ("imbalanced classification heart disease VSD", [14, 17]),
]
print("\n=== Critical Queries ===")
for q, expected in critical:
    res = search(q)
    top3 = [r["id"] for r in res[:3]]
    hit = set(top3) & set(expected)
    print(f"{'HIT' if hit else 'MISS'} expect={expected} got={top3} | {q}")
