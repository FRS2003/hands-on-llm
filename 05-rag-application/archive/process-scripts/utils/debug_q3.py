import os, pickle, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")
from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_fix.index")
with open("vsd_rag_db/texts_fix.pkl", "rb") as f: texts = pickle.load(f)
with open("vsd_rag_db/metas_fix.pkl", "rb") as f: metas = pickle.load(f)

# Check paper 12's text
for i, m in enumerate(metas):
    if m["id"] == 12:
        print(f"[12] Text ({len(texts[i])} chars):")
        print(texts[i][-300:])  # last 300 chars where "limitation" should be
        print()

# Test both Chinese and English queries
for q in ["Cheng等2024年工作的局限性是什么？", "Cheng 2024 limitations"]:
    q_emb = embedder.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 38)  # search ALL

    print(f"Query: {q}")
    # Find where paper 12 ranks
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if metas[int(idx)]["id"] == 12:
            print(f"  [12] rank={rank+1} score={score:.4f}")
            break
    print()
