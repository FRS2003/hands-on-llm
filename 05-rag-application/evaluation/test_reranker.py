import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("Loading BGE-Reranker-v2-m3...")
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, device="cuda")
print("Reranker loaded.")

pairs = [["DINOv2 limitations in medical imaging", "DINOv2 paper experiments are restricted to natural images. No medical imaging experiments are included."],
         ["DINOv2 limitations in medical imaging", "The sky is blue and the weather is nice."]]
scores = reranker.compute_score(pairs)
print(f"Relevant: {scores[0]:.3f}, Irrelevant: {scores[1]:.3f}")
print("Reranker test OK!")
