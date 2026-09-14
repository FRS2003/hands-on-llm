import os
import json, pickle, numpy as np, os, re
os.chdir("data")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

with open("vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# Add author/year metadata prefix to help exact-match queries
for p in papers:
    first_author = p["authors"].split(",")[0].split(";")[0].strip()
    first_author_last = first_author.split()[-1] if " " in first_author else first_author
    year = p.get("year", "")

    ab = p.get("en_abstract", "")
    # Only add metadata prefix if abstract doesn't already start with it
    if not ab.lower().startswith(first_author_last.lower()):
        p["en_abstract"] = f"{first_author_last} et al. ({year}) - {ab}"

# Save and rebuild
with open("vsd_rag_enriched.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)

from sentence_transformers import SentenceTransformer
import faiss
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")

all_texts, all_metas = [], []
for p in papers:
    text = f"Title: {p['title']}. Authors: {p['authors']}. Journal: {p['journal']} ({p['year']}). Abstract: {p.get('en_abstract', '')}"
    all_texts.append(text)
    all_metas.append({"id": p["id"], "title": p["title"], "journal": p["journal"], "year": p["year"], "chunk_id": 0, "total_chunks": 1})

cn = sum(1 for t in all_texts if re.search(r"[一-鿿]", t))
print(f"Chinese: {cn}/38")

embeddings = embedder.encode(all_texts, show_progress_bar=True, batch_size=8)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(1024)
index.add(embeddings)
faiss.write_index(index, "vsd_rag_db/faiss_fix2.index")
with open("vsd_rag_db/texts_fix2.pkl", "wb") as f: pickle.dump(all_texts, f)
with open("vsd_rag_db/metas_fix2.pkl", "wb") as f: pickle.dump(all_metas, f)
print(f"Saved: faiss_fix2.index ({index.ntotal} vectors)")

# Quick test
for q in ["Cheng 2024 limitations", "Cheng limitations"]:
    q_emb = embedder.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 38)
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if all_metas[int(idx)]["id"] == 12:
            print(f"[12] rank={rank+1} score={score:.4f} | {q}")
            break
