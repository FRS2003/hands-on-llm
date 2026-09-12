import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("Downloading BGE-Reranker-v2-m3...")
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cuda")
print("Loaded!")
scores = reranker.compute_score([["DINOv2 limitations", "DINOv2 does not do medical imaging experiments."], ["DINOv2 limitations", "The weather is nice today."]])
print(f"Scores: {scores}")
print("OK")
