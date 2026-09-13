import os, pickle, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")
from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_llm.index")
with open("vsd_rag_db/metas_llm.pkl", "rb") as f: metas = pickle.load(f)

queries = [
    "Cheng 2024 limitations",
    "Cheng et al 2024 study limitations",
    "limitations of deep learning CHD screening study",
    "VSD classification public datasets",
    "imbalanced classification heart disease",
]

for q in queries:
    q_emb = embedder.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 38)
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        pid = metas[int(idx)]["id"]
        if pid in [12, 10, 33, 14, 17]:
            print(f"[{pid}] rank={rank+1:2d} score={score:.4f} | {q}")
