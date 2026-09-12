"""Test HyDE: rewrite query as academic abstract, then search."""
import os, pickle, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import faiss

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")

embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_en.index")
with open("vsd_rag_db/texts_en.pkl", "rb") as f: texts = pickle.load(f)
with open("vsd_rag_db/metas_en.pkl", "rb") as f: metas = pickle.load(f)
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")


def hyde_rewrite(user_query):
    """Rewrite query as an academic abstract sentence."""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": (
            "Convert this question into ONE formal academic sentence as if from a paper abstract. "
            "Only output the sentence.\n\n"
            f"Question: {user_query}"
        )}],
        temperature=0.1, max_tokens=60,
    )
    return resp.choices[0].message.content.strip()


def search(query, top_k=5):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k * 4)

    candidates, seen = [], set()
    for idx, score in zip(indices[0], scores[0]):
        pid = metas[int(idx)]["id"]
        if pid not in seen:
            seen.add(pid)
            candidates.append({"id": pid, "score": float(score), "text": texts[int(idx)]})
        if len(candidates) >= top_k * 4: break

    if len(candidates) > top_k:
        pairs = [(query, c["text"]) for c in candidates[:20]]
        rs = reranker.predict(pairs)
        for c, s in zip(candidates[:20], rs): c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    seen2, final = set(), []
    for c in candidates:
        if c["id"] not in seen2:
            seen2.add(c["id"]); final.append(c)
        if len(final) >= top_k: break
    return final


tests = [
    ("Cheng 2024 limitations", [12]),
    ("VSD classification public datasets", [10, 33]),
    ("imbalanced classification VSD", [14, 17]),
    ("DINOv2 medical imaging experiments", [21]),
    ("EchoNet data leakage prevention", [10]),
    ("Gao BSPC paper difference from our study", [13]),
]

print("HyDE vs Original Query:")
for q, expected in tests:
    hyde_q = hyde_rewrite(q)

    res_orig = search(q)
    top3_orig = [r["id"] for r in res_orig[:3]]
    hit_orig = set(top3_orig) & set(expected)

    res_hyde = search(hyde_q)
    top3_hyde = [r["id"] for r in res_hyde[:3]]
    hit_hyde = set(top3_hyde) & set(expected)

    status = "BETTER" if hit_hyde and not hit_orig else ("WORSE" if not hit_hyde and hit_orig else "SAME")
    print(f"{status:7s} | Q: {q}")
    print(f"         HyDE: {hyde_q[:90]}")
    print(f"         Orig top3={top3_orig} {'HIT' if hit_orig else 'MISS'}")
    print(f"         HyDE top3={top3_hyde} {'HIT' if hit_hyde else 'MISS'}")
    print()
