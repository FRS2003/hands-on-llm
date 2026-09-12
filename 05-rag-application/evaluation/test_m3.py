import os
os.chdir("data")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("Testing BGE-M3 load on GPU...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", device="cuda")
print(f"Loaded. Dim: {model.get_sentence_embedding_dimension()}")
print("BGE-M3 test OK")
