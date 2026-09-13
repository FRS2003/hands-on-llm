import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("Loading bge-reranker-v2-m3 via CrossEncoder...")
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
print("Loaded!")
scores = reranker.predict([
    ("DINOv2 limitations in medical imaging", "DINOv2 paper experiments are restricted to natural images. No medical imaging experiments are included."),
    ("DINOv2 limitations in medical imaging", "The sky is blue and the weather is nice."),
])
print(f"Relevant: {scores[0]:.3f}, Irrelevant: {scores[1]:.3f}")
print("OK!")
