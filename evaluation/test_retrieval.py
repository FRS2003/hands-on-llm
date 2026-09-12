import os, pickle, numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir("data")

from sentence_transformers import SentenceTransformer
import faiss

print("Loading model...")
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print("Model loaded.")

index = faiss.read_index("vsd_rag_db/faiss_v5cn.index")
with open("vsd_rag_db/texts_v5cn.pkl", "rb") as f: texts = pickle.load(f)
with open("vsd_rag_db/metas_v5cn.pkl", "rb") as f: metas = pickle.load(f)
print(f"Index: {index.ntotal} chunks")

seen_ids = set()
for m in metas:
    seen_ids.add(m["id"])
print(f"Papers: {len(seen_ids)}")

queries = [
    "DINOv2在医学影像中有什么优势",
    "VSD亚型分类的主要挑战",
    "数据泄漏对深度学习研究的影响",
    "Cheng等2024年工作的局限性",
]
for q in queries:
    q_emb = model.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 5)
    seen = set()
    print(f"Q: {q}")
    for idx, score in zip(indices[0], scores[0]):
        m = metas[idx]
        pid = m["id"]
        if pid not in seen:
            seen.add(pid)
            short = m["title"][:50]
            print(f"  #{len(seen)} [{pid}] {short} score={score:.3f}")
        if len(seen) >= 3:
            break
    print()

print("Done!")
