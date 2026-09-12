import os, pickle, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_en.index")
with open("vsd_rag_db/texts_en.pkl", "rb") as f: texts = pickle.load(f)
with open("vsd_rag_db/metas_en.pkl", "rb") as f: metas = pickle.load(f)

# Test different query formulations for the 3 missed queries
missed = [
    ("Cheng 2024 limitations", [12], [
        "limitations of Cheng et al 2024 congenital heart disease detection study",
        "shortcomings and weaknesses of multi-view echocardiography CHD screening framework",
        "what are the drawbacks of three-class normal ASD VSD classification",
    ]),
    ("VSD datasets", [10, 33], [
        "publicly available echocardiography datasets for ventricular septal defect",
        "open source congenital heart disease ultrasound benchmark dataset",
    ]),
    ("imbalanced data classification VSD", [14, 17], [
        "class imbalance handling techniques for congenital heart disease deep learning",
        "oversampling augmentation rare class ventricular septal defect classification",
    ]),
]

for orig_q, expected, variants in missed:
    # Original query
    q_emb = embedder.encode([orig_q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 5)
    seen = set()
    orig_hit = False
    for idx in scores[0]:
        pid = metas[int(idx)]["id"]
        if pid in seen: continue
        seen.add(pid)
        if pid in expected:
            orig_hit = True
            break

    # Best variant
    best_score = 0
    best_var = ""
    for var in variants:
        q_emb = embedder.encode([var]).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, 3)
        seen_v = set()
        for idx, score in zip(indices[0], scores[0]):
            pid = metas[int(idx)]["id"]
            if pid in seen_v: continue
            seen_v.add(pid)
            if pid in expected and score > best_score:
                best_score = score
                best_var = var

    print(f"Q: {orig_q}  Original={'HIT' if orig_hit else 'MISS'}")
    if best_var:
        print(f"   Best variant (score={best_score:.3f}): {best_var[:80]}")
    else:
        print(f"   No variant hit")
    print()
