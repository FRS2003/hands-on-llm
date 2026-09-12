import os
import json, pickle, numpy as np, os, re
os.chdir("data")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

with open("vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# Strategy: keep LLM summary + restore author prefix for exact match
for p in papers:
    ab = p.get("en_abstract", "")
    first_author = p["authors"].split(",")[0].strip()
    last_name = first_author.split()[-1] if " " in first_author else first_author
    year = p.get("year", "")
    
    # Remove any existing LLM summary to avoid duplication
    if "[LLM Summary]" in ab:
        ab = ab.split("[LLM Summary]")[0].strip()
    
    # Ensure author prefix
    if not ab.lower().startswith(last_name.lower()):
        # Find the original abstract (before LLM)
        if "[LLM Summary]" in p.get("en_abstract", ""):
            pass  # We already split above
        # Re-add author prefix
        pass

# Actually simpler: just use the author_prefix + LLM summary version
# The enrich JSON already has the author-prefixed abstracts from fix_v2
# We need to merge: author_prefix_abstract + LLM summary

# Load the current state (which has LLM summaries)
with open("vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# For each paper: author prefix + truncated original abstract + LLM summary
for p in papers:
    ab = p.get("en_abstract", "")
    
    # Get author info
    first_author = p["authors"].split(",")[0].strip()
    last_name = first_author.split()[-1] if " " in first_author else first_author
    year = p.get("year", "")
    
    # Split LLM off
    llm_part = ""
    orig_part = ab
    if "[LLM Summary]" in ab:
        parts = ab.split("[LLM Summary]" , 1)
        orig_part = parts[0].strip()
        llm_part = "[LLM Summary] " + parts[1].strip()
    
    # Ensure author prefix on orig part
    if not orig_part.lower().startswith(last_name.lower()):
        # Find the real start (skip Title/Authors headers if present)
        title_start = orig_part.lower().find(p["title"][:30].lower()) if p["title"] else -1
        if title_start > 0:
            orig_part = orig_part[title_start:]
        orig_part = f"{last_name} et al. ({year}) - {orig_part}"
    
    # Combine: short author-tagged version + LLM summary
    p["en_abstract"] = f"{orig_part[:1200]} {llm_part}"

with open("vsd_rag_enriched.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)

# Rebuild
from sentence_transformers import SentenceTransformer
import faiss
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")

all_texts, all_metas = [], []
for p in papers:
    text = f"Title: {p['title']}. Authors: {p['authors']}. Journal: {p['journal']} ({p['year']}). Abstract: {p.get('en_abstract', '')}"
    all_texts.append(text)
    all_metas.append({"id": p["id"], "title": p["title"], "journal": p["journal"], "year": p["year"], "chunk_id": 0, "total_chunks": 1})

embeddings = embedder.encode(all_texts, show_progress_bar=True, batch_size=8)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(1024)
index.add(embeddings)
faiss.write_index(index, "vsd_rag_db/faiss_hybrid2.index")
with open("vsd_rag_db/texts_hybrid2.pkl", "wb") as f: pickle.dump(all_texts, f)
with open("vsd_rag_db/metas_hybrid2.pkl", "wb") as f: pickle.dump(all_metas, f)
print(f"Saved: faiss_hybrid2.index ({index.ntotal} vectors)")

# Quick test
for q in ["Cheng 2024 limitations", "VSD classification public datasets", "imbalanced classification heart disease"]:
    q_emb = embedder.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 38)
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        pid = all_metas[int(idx)]["id"]
        if pid in [12, 10, 33, 14, 17]:
            print(f"[{pid}] rank={rank+1:2d} score={score:.4f} | {q}")
    print()
