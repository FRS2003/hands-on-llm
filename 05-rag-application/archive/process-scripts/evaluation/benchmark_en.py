"""Reranker benchmark: FAISS only vs FAISS + Reranker recall comparison."""
import os, pickle, numpy as np
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

print("Loading models...")
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_en.index")
with open("vsd_rag_db/texts_en.pkl", "rb") as f:
    texts = pickle.load(f)
with open("vsd_rag_db/metas_en.pkl", "rb") as f:
    metas = pickle.load(f)

reranker = None
try:
    reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
    print("Reranker loaded.")
except Exception as e:
    print(f"No reranker: {e}")

# 20 queries with expected correct paper IDs
test_cases = [
    # === 精准命中 ===
    {"q": "DINOv2论文有没有在医学影像上做实验？", "expect": [21]},
    {"q": "EchoNet-Dynamic数据集如何保证患者级数据不泄漏？", "expect": [10]},
    {"q": "Cheng等2024年工作的局限性是什么？", "expect": [12]},
    {"q": "Gao等2025年BSPC论文和我们的研究有什么本质区别？", "expect": [13]},
    {"q": "Arnaout等2021年的产前CHD筛查模型能区分VSD亚型吗？", "expect": [11]},
    # === 跨语言 ===
    {"q": "ventricular septal defect subtype classification challenges", "expect": [17, 8]},
    {"q": "室间隔缺损经导管封堵术的适应证是什么", "expect": [1, 3]},
    {"q": "What are the limitations of self-supervised learning in medical imaging?", "expect": [19, 20]},
    # === 方法论 ===
    {"q": "自监督学习相比监督学习在医学影像分类中有什么优势？", "expect": [19, 20]},
    {"q": "深度学习处理超声心动图视频数据的常用架构有哪些？", "expect": [10, 7]},
    {"q": "多模态多视图的CHD诊断框架是怎么设计的？", "expect": [12]},
    {"q": "如何对超声图像做数据增强来提升VSD分类性能？", "expect": [14]},
    # === 数据方法学 ===
    {"q": "医学AI研究中的数据泄漏有哪些常见形式？如何避免？", "expect": [34, 35]},
    {"q": "患者级别划分和图像级别划分有什么不同？为什么重要？", "expect": [10, 34]},
    {"q": "深度学习VSD分类的公开数据集有哪些？", "expect": [10, 33]},
    {"q": "什么是不平衡数据分类？在心脏病诊断中怎么处理？", "expect": [14, 17]},
    # === 模型比较 ===
    {"q": "ResNet和DINOv2在医学影像任务中应该怎么选择？", "expect": [19, 24]},
    {"q": "Transformer架构在超声心动图分析中的应用有哪些？", "expect": [22, 23]},
    {"q": "Swin-Transformer和CNN在医学图像分类中各有什么优劣？", "expect": [13, 24]},
    # === 临床应用 ===
    {"q": "深度学习模型在先天性心脏病筛查中的临床应用效果如何？", "expect": [11, 12]},
]


def search(query, use_reranker=False, top_k=5, oversample=20):
    q_emb = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, oversample)

    candidates, seen = [], set()
    for idx, score in zip(indices[0], scores[0]):
        pid = metas[idx]["id"]
        if pid not in seen:
            seen.add(pid)
            candidates.append({"id": pid, "score": float(score)})
        if len(candidates) >= oversample:
            break

    if use_reranker and reranker:
        pairs = [(query, texts[i]) for i in range(len(candidates))]
        # We need to map candidates back to text indices
        # Rebuild with text refs
        text_indices = []
        seen2 = set()
        for idx, score in zip(indices[0], scores[0]):
            pid = metas[idx]["id"]
            if pid not in seen2:
                seen2.add(pid)
                text_indices.append(idx)
            if len(text_indices) >= oversample:
                break
        pairs = [(query, texts[ti]) for ti in text_indices]
        rs = reranker.predict(pairs)
        for c, s in zip(candidates, rs):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    return candidates[:top_k]


print("\n" + "=" * 80)
print("BGE-M3  vs  BGE-M3 + Reranker  召回率对比测试 (20 queries)")
print("=" * 80)

faiss_hits = 0
faiss_total = 0
rerank_hits = 0
rerank_total = 0

for i, tc in enumerate(test_cases):
    q = tc["q"]
    expected = set(tc["expect"])

    f_results = search(q, use_reranker=False)
    f_top3 = set(r["id"] for r in f_results[:3])
    f_hit = 1 if (f_top3 & expected) else 0
    faiss_hits += f_hit
    faiss_total += 1

    r_hit = f_hit
    r_results = f_results
    if reranker:
        r_results = search(q, use_reranker=True)
        r_top3 = set(r["id"] for r in r_results[:3])
        r_hit = 1 if (r_top3 & expected) else 0
        rerank_hits += r_hit
        rerank_total += 1

    status = "BETTER" if r_hit > f_hit else ("SAME" if r_hit == f_hit else "WORSE")
    print(f"\n[{i+1:2d}] {status}")
    print(f"    Q: {q[:55]}")
    print(f"    Expect: {tc['expect']}")
    f_ids = [str(r['id']) for r in f_results[:3]]
    print(f"    FAISS only : Top-3={f_ids}  {'HIT' if f_hit else 'MISS'}")
    if reranker:
        r_ids = [str(r['id']) for r in r_results[:3]]
        print(f"    + Reranker  : Top-3={r_ids}  {'HIT' if r_hit else 'MISS'}")

print("\n" + "=" * 80)
print(f"FAISS only  Recall@3: {faiss_hits}/{faiss_total} = {faiss_hits/faiss_total:.1%}")
if reranker:
    print(f"+ Reranker  Recall@3: {rerank_hits}/{rerank_total} = {rerank_hits/rerank_total:.1%}")
    delta = rerank_hits - faiss_hits
    print(f"Improvement: {'+' if delta>=0 else ''}{delta} queries")
print("=" * 80)
