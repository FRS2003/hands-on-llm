"""Benchmark: FAISS + Query Expansion + Reranker"""
import os, pickle, numpy as np, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.chdir("data")

from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
import faiss

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY","")
client = OpenAI(api_key=DEEPSEEK_KEY, base_url="https://api.deepseek.com")

print("Loading models...")
embedder = SentenceTransformer("BAAI/bge-m3", device="cuda")
index = faiss.read_index("vsd_rag_db/faiss_en.index")
with open("vsd_rag_db/texts_en.pkl", "rb") as f:
    texts = pickle.load(f)
with open("vsd_rag_db/metas_en.pkl", "rb") as f:
    metas = pickle.load(f)

reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
print("All models loaded.\n")

# Cache for expanded queries
query_cache = {}


def expand_query(cn_query):
    """Use DeepSeek to expand Chinese query into English academic keywords."""
    if cn_query in query_cache:
        return query_cache[cn_query]

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": (
                    "Convert this Chinese research question into concise English academic search keywords "
                    "(include technical terms, synonyms, and key concepts). Keep it under 15 words. "
                    "Only output the English keywords, nothing else.\n\n"
                    f"Question: {cn_query}"
                ),
            }],
            temperature=0.1,
            max_tokens=50,
        )
        expanded = resp.choices[0].message.content.strip()
        query_cache[cn_query] = expanded
        return expanded
    except Exception as e:
        print(f"  Query expansion error: {e}")
        return cn_query


def search(query, use_expansion=False, use_reranker=True, top_k=5, oversample=20):
    search_query = query
    if use_expansion:
        en_keywords = expand_query(query)
        search_query = f"{query} {en_keywords}"

    q_emb = embedder.encode([search_query]).astype("float32")
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

    if use_reranker:
        pairs = [(search_query, texts[idx]) for idx in indices[0][:len(candidates)]]
        rs = reranker.predict(pairs)
        for c, s in zip(candidates, rs):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    return candidates[:top_k]


# Test queries with expected answers
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

print("=" * 80)
print("FAISS vs FAISS+QueryExpansion vs FAISS+QE+Reranker  对比测试")
print("=" * 80)

results = {"faiss": 0, "qe": 0, "qe_rerank": 0}
total = len(test_cases)

for i, tc in enumerate(test_cases):
    q = tc["q"]
    expected = set(tc["expect"])

    # 1. FAISS only (original)
    f_results = search(q, use_expansion=False, use_reranker=False)
    f_hit = 1 if (set(r["id"] for r in f_results[:3]) & expected) else 0
    results["faiss"] += f_hit

    # 2. FAISS + Query Expansion (no reranker)
    qe_results = search(q, use_expansion=True, use_reranker=False)
    qe_hit = 1 if (set(r["id"] for r in qe_results[:3]) & expected) else 0
    results["qe"] += qe_hit

    # 3. FAISS + QE + Reranker
    qer_results = search(q, use_expansion=True, use_reranker=True)
    qer_hit = 1 if (set(r["id"] for r in qer_results[:3]) & expected) else 0
    results["qe_rerank"] += qer_hit

    status = (
        "NEW" if qer_hit > f_hit
        else "LOST" if qer_hit < f_hit
        else "SAME"
    )
    en_kw = query_cache.get(q, "")[:60]
    print(f"\n[{i+1:2d}] {status}")
    print(f"    Q: {q[:50]}")
    print(f"    EN: {en_kw}")
    print(f"    FAISS={f_hit}  +QE={qe_hit}  +QE+Rerank={qer_hit}  | Expect={tc['expect']}")

    time.sleep(0.3)

print("\n" + "=" * 80)
print(f"FAISS only          : {results['faiss']}/{total} = {results['faiss']/total:.1%}")
print(f"FAISS + QE          : {results['qe']}/{total} = {results['qe']/total:.1%}")
print(f"FAISS + QE + Rerank : {results['qe_rerank']}/{total} = {results['qe_rerank']/total:.1%}")
print(f"QE boost on FAISS   : +{results['qe'] - results['faiss']}")
print(f"QE+Rerank boost     : +{results['qe_rerank'] - results['faiss']}")
print("=" * 80)
