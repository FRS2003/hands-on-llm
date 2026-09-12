"""QE for FAISS search + ORIGINAL query for Reranker"""
import os, pickle, numpy as np, time
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

cache = {}
def expand_query(q):
    if q in cache: return cache[q]
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"Convert to English academic search keywords (max 10 words): {q}"}],
        temperature=0.1, max_tokens=30,
    )
    cache[q] = resp.choices[0].message.content.strip()
    return cache[q]

def search_and_rerank(orig_query, search_query, oversample=20, top_k=5):
    # FAISS with search query
    q_emb = embedder.encode([search_query]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, oversample)

    candidates, seen = [], set()
    for idx, score in zip(indices[0], scores[0]):
        pid = metas[int(idx)]["id"]
        if pid not in seen:
            seen.add(pid)
            candidates.append({"id": pid, "score": float(score), "text": texts[int(idx)]})
        if len(candidates) >= oversample: break

    # Reranker with ORIGINAL query
    if len(candidates) > top_k:
        pairs = [(orig_query, c["text"]) for c in candidates[:min(20, len(candidates))]]
        rs = reranker.predict(pairs)
        for c, s in zip(candidates[:len(rs)], rs): c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x.get("rerank_score", x["score"]), reverse=True)

    seen2, final = set(), []
    for c in candidates:
        if c["id"] not in seen2: seen2.add(c["id"]); final.append(c)
        if len(final) >= top_k: break
    return final

tests = [
    ("DINOv2论文有没有在医学影像上做实验？", [21]),
    ("EchoNet-Dynamic数据集如何保证患者级数据不泄漏？", [10]),
    ("Cheng等2024年工作的局限性是什么？", [12]),
    ("Gao等2025年BSPC论文和我们的研究有什么本质区别？", [13]),
    ("Arnaout等2021年的产前CHD筛查模型能区分VSD亚型吗？", [11]),
    ("ventricular septal defect subtype classification challenges", [17, 8]),
    ("室间隔缺损经导管封堵术的适应证是什么", [1, 3]),
    ("自监督学习相比监督学习在医学影像分类中有什么优势？", [19, 20]),
    ("多模态多视图的CHD诊断框架是怎么设计的？", [12]),
    ("如何对超声图像做数据增强来提升VSD分类性能？", [14]),
    ("医学AI研究中的数据泄漏有哪些常见形式？", [34, 35]),
    ("深度学习VSD分类的公开数据集有哪些？", [10, 33]),
    ("什么是不平衡数据分类？在心脏病诊断中怎么处理？", [14, 17]),
    ("ResNet和DINOv2在医学影像任务中应该怎么选择？", [19, 24]),
    ("Transformer架构在超声心动图分析中的应用有哪些？", [22, 23]),
    ("Swin-Transformer和CNN在医学图像分类中各有什么优劣？", [13, 24]),
    ("深度学习模型在先天性心脏病筛查中的临床应用效果如何？", [11, 12]),
    ("What are the limitations of self-supervised learning in medical imaging?", [19, 20]),
    ("患者级别划分和图像级别划分有什么不同？为什么重要？", [10, 34]),
    ("深度学习处理超声心动图视频数据的常用架构有哪些？", [10, 7]),
]

f_hits = qe_hits = split_hits = 0
print(f"{'Q':3s} {'FAISS':5s} {'QE':5s} {'SPLIT':5s} | Query")
print("-" * 60)

for i, (q, expected) in enumerate(tests):
    exp = set(expected)

    # FAISS only (original query)
    r = search_and_rerank(q, q, oversample=20, top_k=3)
    f_hit = 1 if set(c["id"] for c in r) & exp else 0
    f_hits += f_hit

    # QE: search with expanded, rerank with expanded
    eq = expand_query(q)
    r2 = search_and_rerank(eq, eq, oversample=20, top_k=3)
    qe_hit = 1 if set(c["id"] for c in r2) & exp else 0
    qe_hits += qe_hit

    # SPLIT: search with expanded, rerank with ORIGINAL
    r3 = search_and_rerank(q, eq, oversample=20, top_k=3)
    split_hit = 1 if set(c["id"] for c in r3) & exp else 0
    split_hits += split_hit

    s1 = "H" if f_hit else "."
    s2 = "H" if qe_hit else "."
    s3 = "H" if split_hit else "."
    star = " NEW" if split_hit and not f_hit else ""
    print(f"{i+1:2d}   {s1}     {s2}     {s3}   {star} | {q[:45]}")

    time.sleep(0.3)

print(f"\nFAISS only : {f_hits}/20 = {f_hits/20:.0%}")
print(f"QE both    : {qe_hits}/20 = {qe_hits/20:.0%}")
print(f"QE split   : {split_hits}/20 = {split_hits/20:.0%}")
