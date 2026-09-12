import os, json, pickle, numpy as np, re
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("agent")

print("Loading 208 papers...")
with open("data/papers_200.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
print(f"Papers: {len(papers)}")
print(f"With PMID: {sum(1 for p in papers if p.get('pmid'))}")
print(f"With DOI: {sum(1 for p in papers if p.get('doi'))}")

print("Loading BGE-M3 on GPU...")
from sentence_transformers import SentenceTransformer
import faiss
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
dim = embedder.get_sentence_embedding_dimension()
print(f"Dimension: {dim}")

all_texts, all_metas = [], []
for p in papers:
    ab = p.get("en_abstract") or p.get("summary") or ""
    text = (
        f"Title: {p['title']}. "
        f"Authors: {p.get('authors', '')}. "
        f"Journal: {p.get('journal', '')} ({p.get('year', '')}). "
        f"Abstract: {ab}"
    )
    all_texts.append(text)
    all_metas.append({
        "id": p["id"], "title": p["title"],
        "journal": p.get("journal", ""), "year": p.get("year", 0),
        "pmid": p.get("pmid", ""), "doi": p.get("doi", ""),
        "chunk_id": 0, "total_chunks": 1,
    })

cn = sum(1 for t in all_texts if re.search(r"[一-鿿]", t))
print(f"Chinese content: {cn}/{len(all_texts)}")

print("Encoding with BGE-M3 (GPU)...")
embeddings = embedder.encode(all_texts, show_progress_bar=True, batch_size=16)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS: {index.ntotal} vectors, dim={dim}")

os.makedirs("vsd_rag_db", exist_ok=True)
faiss.write_index(index, "vsd_rag_db/faiss_200.index")
with open("vsd_rag_db/texts_200.pkl", "wb") as f:
    pickle.dump(all_texts, f)
with open("vsd_rag_db/metas_200.pkl", "wb") as f:
    pickle.dump(all_metas, f)
print("Saved: faiss_200.index, texts_200.pkl, metas_200.pkl")

# Quick test
q = "DINOv2 medical imaging experiments"
q_emb = embedder.encode([q]).astype("float32")
faiss.normalize_L2(q_emb)
scores, indices = index.search(q_emb, 5)
seen = set()
print(f"\nTest: {q}")
for idx, score in zip(indices[0], scores[0]):
    m = all_metas[int(idx)]
    pid = m["id"]
    if pid not in seen:
        seen.add(pid)
        print(f"  [{pid}] score={score:.3f} {m['title'][:50]}...")
    if len(seen) >= 3:
        break
print("Done!")
