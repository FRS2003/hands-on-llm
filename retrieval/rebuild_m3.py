"""Rebuild FAISS index with BGE-M3 (English original texts, 1024-dim)."""
import os, json, pickle, numpy as np

os.chdir("data")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("Loading BGE-M3 model on GPU...")
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("BAAI/bge-m3", device="cuda")
dim = model.get_sentence_embedding_dimension()
print(f"Loaded. Dimension: {dim}")

# Load knowledge base
with open("vsd_rag_knowledge.json", "r", encoding="utf-8") as f:
    articles = json.load(f)
print(f"Loaded {len(articles)} papers")

# ── English full-text for 5 key papers ──
fulltext = {
    10: (
        "Video-based AI for beat-to-beat assessment of cardiac function (EchoNet-Dynamic). "
        "Authors: Ouyang D, He B, Ghorbani A, et al. (Stanford University). "
        "Journal: Nature, 2020, 580(7802): 252-256. "
        "Method: Video-level deep learning model using R(2+1)D convolutional architecture. "
        "Input is apical-4-chamber echocardiogram videos, output is left ventricular segmentation mask and EF prediction. "
        "Uses semantic segmentation + regression joint training. "
        "Data: 10,030 annotated echocardiogram videos, publicly released, largest public echo dataset at the time. "
        "Key Results: LV segmentation Dice 0.92. EF prediction MAE 4.1%. "
        "Heart failure classification AUC 0.97 internal, 0.96 external validation. "
        "Model variance comparable or lower than human experts. "
        "Patient-level split: Authors explicitly use patient-level split in methods, "
        "train/validation/test sets split by patient ID, no cross-set leakage. "
        "Contribution: First video-level end-to-end echocardiogram assessment. "
        "Released large public dataset and code."
    ),
    11: (
        "An ensemble of neural networks provides expert-level prenatal detection of complex congenital heart disease. "
        "Authors: Arnaout R, Curran L, Zhao Y, Levine JC, Chinn E, Moon-Grady AJ. "
        "Journal: Nature Medicine, 2021, 27(5): 882-891. "
        "Method: Three-stage ensemble neural network. "
        "Stage 1 view classification CNN identifies 5 guideline-recommended fetal cardiac standard views. "
        "Stage 2 diagnosis classification: each view independently trained as normal vs CHD binary classifier. "
        "Stage 3 composite diagnosis: rule-based classifier integrates predictions across views. "
        "Data: 107,823 images from 1,326 retrospective ultrasound studies. "
        "Multi-center: UCSF training + internal test, Boston Children's external validation. "
        "Key Results: Population-level testing AUC 0.99, sensitivity 95%, specificity 96%, NPV 100%. "
        "External validation AUC 0.89. Model sensitivity 88% comparable to clinicians 86%, "
        "specificity 90% significantly better than clinicians 68%. "
        "Limitations: Primarily normal vs CHD binary screening, does not distinguish specific CHD types like VSD subtypes."
    ),
    12: (
        "Development and validation of a deep-learning network for detecting congenital heart disease "
        "from multi-view multi-modal transthoracic echocardiograms. "
        "Authors: Cheng M, Wang J, Liu X, et al. "
        "Journal: Research, 2024, 7: 0319. "
        "Method: Hierarchical DL framework using 5 standard TTE views with 2D grayscale + Doppler dual-modality images. "
        "Depthwise separable convolution multi-channel network handles missing views/modalities. "
        "Two-stage pipeline: view classification then CHD screening normal/ASD/VSD three-class + high-risk region visualization. "
        "Data: 1,932 children 1,255 normal, 292 ASD, 385 VSD, Beijing Children's Hospital 2018-2022. "
        "Key Results: View classification accuracy 0.989. CHD screening AUC 0.996 single-center, 0.990 cross-center. "
        "Three-class accuracy 0.991 single-center, 0.986 cross-center. "
        "Limitations: Only normal/ASD/VSD three-class, does not distinguish VSD five subtypes. "
        "ASD vs VSD discrimination precision not separately reported. Data primarily from children, lacks adult data."
    ),
    13: (
        "Diagnosis and classification of congenital heart disease in infants based on Gated Swin-Transformer "
        "with Multi-scale Feature Fusion. "
        "Authors: Gao Y, Yang H, Pan J, Guo T, Wang W. "
        "Journal: Biomedical Signal Processing and Control, 2025, 108: 107919. "
        "Method: Gated Swin-Transformer (G-Swin-T) with multi-scale feature fusion. "
        "Gating units adaptively adjust signal flow weights. Entropy regularization encourages discriminative predictions. "
        "Uses SMOTE oversampling and SpecAugment spectral random time-frequency masking. "
        "Transfer learning from esc-50 dataset with layer-wise learning rate fine-tuning. "
        "Data: 423 infants aged 0-3 years, heart sound signals, 6-class classification VSD/ASD/PDA/PAH/other CHD/normal. "
        "External validation on public ZCHSound dataset 941 children + 318 neonates. "
        "Key Results: 96.66% six-class accuracy on self-built dataset, 96.62% on ZCHSound high-quality set, "
        "85.92% on low-quality neonate set 30.22% improvement over original. "
        "Important distinction: This study uses phonocardiogram PCG heart sound signals, NOT echocardiogram images. "
        "Classification is coarse-grained CHD screening between different diseases, NOT VSD internal subtype classification. "
        "VSD is only one of six classes. Fundamentally different from our study in both data modality and classification granularity."
    ),
    21: (
        "DINOv2: Learning robust visual features without supervision. "
        "Authors: Oquab M, Darcet T, Moutakanni T, et al. (Meta AI Research & Inria). "
        "Journal: Transactions on Machine Learning Research, 2024. arXiv: 2304.07193. "
        "Method: Self-supervised visual foundation model combining DINO image-level self-distillation, "
        "iBOT patch-level masked prediction, and SwAV Sinkhorn-Knopp centering. "
        "KoLeo regularizer ensures uniform feature distribution. "
        "Teacher ViT-g 1.1B params 1536-dim embeddings trained via student-teacher mutual distillation, "
        "then distilled to ViT-B/14 86M. Custom FlashAttention gives 2x training speedup, 1/3 memory of original iBOT. "
        "Data: LVD-142M dataset, 142M images filtered from 1.2B raw web images via self-supervised retrieval pipeline, "
        "deduplication and copy detection. 20-node 8xV100 cluster completes filtering in 2 days. "
        "Key Results: ImageNet-1K frozen features + linear classification surpasses OpenCLIP. "
        "PCA visualization shows patch features automatically separate foreground/background. "
        "Cross-domain semantic matching of similar parts across species. "
        "Depth estimation and semantic segmentation without fine-tuning achieve competitive performance. "
        "Important limitation: Paper experiments restricted to natural images and general vision tasks. "
        "No medical imaging experiments ultrasound, X-ray, CT, MRI are included. "
        "Authors discuss methodological framework could generalize to arbitrary image domains but do not verify this. "
        "DINOv2 performance on echocardiography depends entirely on whether pretrained features transfer to grayscale ultrasound domain."
    ),
}


def chunk_text(text, chunk_size=200):
    """Split text into overlapping ~200-char chunks at sentence boundaries."""
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks


# Build texts and metadata
all_texts = []
all_metas = []
chunk_stats = {}

for a in articles:
    aid = a["id"]
    if aid in fulltext:
        chunks = chunk_text(fulltext[aid])
        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metas.append({
                "id": aid, "title": a["title"], "journal": a["journal"],
                "year": a["year"], "chunk_id": i, "total_chunks": len(chunks),
            })
        chunk_stats[aid] = len(chunks)
    else:
        text = (
            f"Title: {a['title']}. Authors: {a['authors']}. "
            f"Journal: {a['journal']} ({a['year']}). "
            f"Keywords: {', '.join(a['keywords'])}. "
            f"Summary: {a['summary']}. Key Findings: {a['key_findings']}."
        )
        all_texts.append(text)
        all_metas.append({
            "id": aid, "title": a["title"], "journal": a["journal"],
            "year": a["year"], "chunk_id": 0, "total_chunks": 1,
        })
        chunk_stats[aid] = 1

print(f"\nTotal chunks: {len(all_texts)}")
print(f"Unique papers: {len(set(m['id'] for m in all_metas))}")
for aid, n in sorted(chunk_stats.items()):
    flag = " [FULL]" if aid in fulltext else ""
    print(f"  [{aid}] {n} chunks{flag}")

# Encode with BGE-M3 on GPU
print("\nEncoding with BGE-M3 (GPU)...")
embeddings = model.encode(all_texts, show_progress_bar=True, batch_size=16)
embeddings = np.array(embeddings).astype("float32")
print(f"Embeddings: {embeddings.shape}")

# L2 normalize for cosine similarity (IndexFlatIP)
faiss.normalize_L2(embeddings)

# Build FAISS index
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

# Save
os.makedirs("vsd_rag_db", exist_ok=True)
faiss.write_index(index, "vsd_rag_db/faiss_m3.index")
with open("vsd_rag_db/texts_m3.pkl", "wb") as f:
    pickle.dump(all_texts, f)
with open("vsd_rag_db/metas_m3.pkl", "wb") as f:
    pickle.dump(all_metas, f)
print("\nSaved: vsd_rag_db/faiss_m3.index, texts_m3.pkl, metas_m3.pkl")

# ── Quick test ──
print("\n" + "=" * 60)
print("QUICK TEST")
print("=" * 60)

queries = [
    "DINOv2在医学影像中有什么优势",
    "VSD subtype classification challenges",
    "data leakage impact on deep learning",
    "Cheng 2024 limitations",
    "Gao BSPC paper difference from our study",
]
for q in queries:
    q_emb = model.encode([q]).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, 5)
    seen = set()
    print(f"\nQ: {q}")
    for idx, score in zip(indices[0], scores[0]):
        m = all_metas[idx]
        pid = m["id"]
        if pid not in seen:
            seen.add(pid)
            tag = f"[chunk {m['chunk_id']}/{m['total_chunks']-1}]" if m["total_chunks"] > 1 else "[meta]"
            print(f"  #{len(seen)} [{pid}] {m['title'][:55]}... score={score:.3f} {tag}")
        if len(seen) >= 2:
            break

print("\nDone!")
