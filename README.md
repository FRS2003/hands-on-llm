---
title: VSD Literature Assistant
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---

# 📚 VSD Literature Assistant

**RAG-powered Q&A over 38 papers on congenital heart disease & deep learning.**

## Overview

A Retrieval-Augmented Generation (RAG) system that answers research questions about Ventricular Septal Defect (VSD) classification, echocardiography deep learning, and related topics — backed by a curated knowledge base of 38 academic papers.

## Features

- 🔍 **FAISS vector search** with BGE-small-zh-v1.5 embeddings (512-dim)
- 🇨🇳 **Chinese full-text encoding** for 5 key papers
- 🤖 **DeepSeek API** integration for live AI-generated answers
- 📄 **Paper-level deduplication** with score-ranked results
- 📝 **Source citations** — every answer links back to specific papers

## Tech Stack

| Component | Technology |
|---|---|
| Vector DB | FAISS (IndexFlatIP) |
| Embeddings | BGE-small-zh-v1.5 |
| Reranker | FAISS coarse search + paper dedup |
| Frontend | Streamlit |
| LLM | DeepSeek API (optional) |

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Graduate student project — CV to AI career transition.
