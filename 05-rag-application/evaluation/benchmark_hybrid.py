"""Benchmark: FAISS vs FAISS+BM25 hybrid search"""
import os, pickle, numpy as np, re, math
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from collections import Counter

print("Loading models...")
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_en.index")
with open("vsd_rag_db/texts_en.pkl", "rb") as f:
    texts = pickle.load(f)
with open("vsd_rag_db/metas_en.pkl", "rb") as f:
    metas = pickle.load(f)

reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
print(f"Loaded: {len(texts)} chunks, {index.ntotal} vectors")

# ── Build BM25 index ──
def tokenize(text):
    """Simple English tokenization"""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]

# Tokenize all chunks
tokenized = [tokenize(t) for t in texts]
doc_count = len(tokenized)
avg_dl = sum(len(t) for t in tokenized) / doc_count

# BM25 parameters
k1, b = 1.5, 0.75

# Compute IDF
df = Counter()
for tokens in tokenized:
    df.update(set(tokens))

idf = {}
for term, freq in df.items():
    idf[term] = math.log(1 + (doc_count - freq + 0.5) / (freq + 0.5))

# Compute doc lengths
doc_lens = [len(t) for t in tokenized]

def bm25_score(query, doc_idx):
    """BM25 score for a single document"""
    q_tokens = tokenize(query)
    doc_tokens = tokenized[doc_idx]
    dl = doc_lens[doc_idx]

    score = 0.0
    tf_counter = Counter(doc_tokens)

    for term in q_tokens:
        if term not in idf:
            continue
        tf = tf_counter.get(term, 0)
        if tf == 0:
            continue
        numerator = idf[term] * tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / avg_dl)
        score += numerator / denominator
    return score

def bm25_search(query, top_k=100):
    """BM25 search returning doc indices and scores"""
    scores = []
    for i in range(doc_count):
        s = bm25_score(query, i)
        if s > 0:
            scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

print("BM25 index built.")

# ── Search functions ──
def faiss_search(query, top_k=20):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({"idx": int(idx), "score": float(score)})
    return results

def hybrid_search(query, alpha=0.6, top_k=20):
    """Combine FAISS (semantic) + BM25 (keyword) with weighted scores"""
    # Get FAISS results
    faiss_results = faiss_search(query, top_k=50)
    # Get BM25 results
    bm25_results = bm25_search(query, top_k=50)

    # Normalize scores to [0, 1]
    f_max = max(r["score"] for r in faiss_results) if faiss_results else 1.0
    b_max = max(r[1] for r in bm25_results) if bm25_results else 1.0

    # Combine scores per document (by paper ID for dedup)
    combined = {}  # paper_id -> best combined score
    idx_map = {}   # paper_id -> best chunk idx

    for r in faiss_results:
        pid = metas[r["idx"]]["id"]
        f_score = r["score"] / f_max if f_max > 0 else 0
        combined_score = alpha * f_score
        if pid not in combined or combined_score > combined[pid]:
            combined[pid] = combined_score
            idx_map[pid] = r["idx"]

    for bm_idx, bm_score in bm25_results:
        pid = metas[bm_idx]["id"]
        b_score = bm_score / b_max if b_max > 0 else 0
        existing = combined.get(pid, 0)
        combined_score = existing + (1 - alpha) * b_score
        # Only add BM25 contribution if better
        if combined_score > combined.get(pid, -1):
            combined[pid] = combined_score
            if pid not in idx_map:
                idx_map[pid] = bm_idx

    # Sort and return top-k
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    results = []
    for pid, score in ranked[:top_k]:
        results.append({
            "id": pid,
            "title": metas[idx_map[pid]]["title"],
            "journal": metas[idx_map[pid]]["journal"],
            "year": metas[idx_map[pid]]["year"],
            "text": texts[idx_map[pid]],
            "score": score,
        })
    return results

def dedup_and_rerank(candidates, query, top_k=5):
    """Dedup by paper ID and rerank"""
    pairs = [(query, c["text"]) for c in candidates[:20]]
    rs = reranker.predict(pairs)
    for c, s in zip(candidates[:20], rs):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    seen = set()
    final = []
    for c in candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            final.append(c)
        if len(final) >= top_k:
            break
    return final


# ── Test ──
test_cases = [
    {"q": "DINOv2论文有没有在医学影像上做实验？", "expect": [21]},
    {"q": "EchoNet-Dynamic数据集如何保证患者级数据不泄漏？", "expect": [10]},
    {"q": "Cheng等2024年工作的局限性是什么？", "expect": [12]},
    {"q": "Gao等2025年BSPC论文和我们的研究有什么本质区别？", "expect": [13]},
    {"q": "Arnaout等2021年的产前CHD筛查模型能区分VSD亚型吗？", "expect": [11]},
    {"q": "ventricular septal defect subtype classification challenges", "expect": [17, 8]},
    {"q": "室间隔缺损经导管封堵术的适应证是什么", "expect": [1, 3]},
    {"q": "What are the limitations of self-supervised learning in medical imaging?", "expect": [19, 20]},
    {"q": "自监督学习相比监督学习在医学影像分类中有什么优势？", "expect": [19, 20]},
    {"q": "深度学习处理超声心动图视频数据的常用架构有哪些？", "expect": [10, 7]},
    {"q": "多模态多视图的CHD诊断框架是怎么设计的？", "expect": [12]},
    {"q": "如何对超声图像做数据增强来提升VSD分类性能？", "expect": [14]},
    {"q": "医学AI研究中的数据泄漏有哪些常见形式？如何避免？", "expect": [34, 35]},
    {"q": "患者级别划分和图像级别划分有什么不同？为什么重要？", "expect": [10, 34]},
    {"q": "深度学习VSD分类的公开数据集有哪些？", "expect": [10, 33]},
    {"q": "什么是不平衡数据分类？在心脏病诊断中怎么处理？", "expect": [14, 17]},
    {"q": "ResNet和DINOv2在医学影像任务中应该怎么选择？", "expect": [19, 24]},
    {"q": "Transformer架构在超声心动图分析中的应用有哪些？", "expect": [22, 23]},
    {"q": "Swin-Transformer和CNN在医学图像分类中各有什么优劣？", "expect": [13, 24]},
    {"q": "深度学习模型在先天性心脏病筛查中的临床应用效果如何？", "expect": [11, 12]},
]

print("\n" + "=" * 80)
print("FAISS vs Hybrid(BM25+FAISS) vs Hybrid+Reranker  Recall@3 对比")
print("=" * 80)

f_hits, h_hits, hr_hits = 0, 0, 0
total = len(test_cases)

for i, tc in enumerate(test_cases):
    q = tc["q"]
    expected = set(tc["expect"])

    # FAISS only
    f_res = faiss_search(q, top_k=50)
    seen = set()
    f_top3 = []
    for r in f_res:
        pid = metas[r["idx"]]["id"]
        if pid not in seen:
            seen.add(pid)
            f_top3.append(pid)
        if len(f_top3) >= 3:
            break
    f_hit = 1 if (set(f_top3) & expected) else 0
    f_hits += f_hit

    # Hybrid
    h_res = hybrid_search(q, alpha=0.6, top_k=3)
    h_top3 = [r["id"] for r in h_res]
    h_hit = 1 if (set(h_top3) & expected) else 0
    h_hits += h_hit

    # Hybrid + Reranker
    hr_res = hybrid_search(q, alpha=0.6, top_k=20)
    hr_final = dedup_and_rerank(hr_res, q, top_k=3)
    hr_top3 = [r["id"] for r in hr_final]
    hr_hit = 1 if (set(hr_top3) & expected) else 0
    hr_hits += hr_hit

    status = "NEW" if hr_hit > f_hit else ("LOST" if hr_hit < f_hit else "SAME")
    print(f"[{i+1:2d}] {status} FAISS={f_hit} Hybrid={h_hit} Hybrid+Rerank={hr_hit} | {q[:45]}...")

print(f"\n{'='*80}")
print(f"FAISS only        : {f_hits}/{total} = {f_hits/total:.1%}")
print(f"Hybrid (BM25+Vec) : {h_hits}/{total} = {h_hits/total:.1%}")
print(f"Hybrid + Reranker : {hr_hits}/{total} = {hr_hits/total:.1%}")
print(f"{'='*80}")
