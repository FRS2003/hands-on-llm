import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("Loading BGE Reranker via sentence-transformers...")
from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-base", device="cuda")
print("Loaded!")
pairs = [("DINOv2 limitations in medical imaging", "DINOv2 paper experiments are restricted to natural images. No medical imaging experiments are included."),
         ("DINOv2 limitations in medical imaging", "The sky is blue and the weather is nice.")]
scores = model.predict(pairs)
print(f"Relevant: {scores[0]:.3f}, Irrelevant: {scores[1]:.3f}")
print("Reranker OK!")
