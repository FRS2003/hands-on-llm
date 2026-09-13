import os, json, pickle, numpy as np, re
os.chdir("data")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
dim = embedder.get_sentence_embedding_dimension()

with open("vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    articles = json.load(f)
print(f"Loaded {len(articles)} papers")

def chunk_text(text, size=200):
    sentences = re.split(r"(?<=[.!?]) ", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < size:
            cur += s + " "
        else:
            if cur.strip(): chunks.append(cur.strip())
            cur = s + " "
    if cur.strip(): chunks.append(cur.strip())
    return chunks

all_texts, all_metas = [], []
ft_count, ab_count = 0, 0

for a in articles:
    aid = a["id"]
    ft = a.get("pmc_fulltext", "")
    if ft and len(ft) > 500:
        # Use PMC full text
        chunks = chunk_text(ft)
        for i, chunk in enumerate(chunks):
            metadata = f"[Paper {aid}: {a['title']}. {a['journal']} ({a['year']}). Authors: {a['authors']}] "
            all_texts.append(metadata + chunk)
            all_metas.append({"id": aid, "title": a["title"], "journal": a["journal"], "year": a["year"], "chunk_id": i, "total_chunks": len(chunks)})
        ft_count += 1
    else:
        # Use English abstract
        abstract = a.get("en_abstract", "")
        text = f"Title: {a['title']}. Authors: {a['authors']}. Journal: {a['journal']} ({a['year']}). Abstract: {abstract}"
        all_texts.append(text)
        all_metas.append({"id": aid, "title": a["title"], "journal": a["journal"], "year": a["year"], "chunk_id": 0, "total_chunks": 1})
        ab_count += 1

cn = sum(1 for t in all_texts if re.search(r"[一-鿿]", t))
print(f"Full text papers: {ft_count}, Abstract papers: {ab_count}")
print(f"Chinese chunks: {cn}/{len(all_texts)}")
print(f"Total chunks: {len(all_texts)}")

embeddings = embedder.encode(all_texts, show_progress_bar=True, batch_size=8)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS: {index.ntotal} vectors, dim={dim}")

faiss.write_index(index, "vsd_rag_db/faiss_pmc.index")
with open("vsd_rag_db/texts_pmc.pkl", "wb") as f: pickle.dump(all_texts, f)
with open("vsd_rag_db/metas_pmc.pkl", "wb") as f: pickle.dump(all_metas, f)
print("Saved: faiss_pmc.index")
