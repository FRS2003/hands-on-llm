"""
Expand paper collection from 38 to 200 using Semantic Scholar API.

Strategy: Search for papers across broader CHD/AI topics,
deduplicate with existing 38 papers, collect English abstracts.
"""
import json
import re
import time
import urllib.request
import urllib.parse

# Load existing papers to avoid duplicates
with open("data/vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    existing = json.load(f)

existing_dois = set()
existing_titles = set()
for p in existing:
    doi = (p.get("doi") or "").lower().strip()
    if doi:
        existing_dois.add(doi)
    title = p["title"].lower().strip().rstrip(".")
    existing_titles.add(title)

print(f"Existing: {len(existing)} papers, {len(existing_dois)} DOIs")

# ── Search queries for expanding scope ──
# Covering: CHD, echo, cardiac, fetal, pediatric, segmentation, SSL
SEARCH_QUERIES = [
    "congenital heart disease deep learning ultrasound",
    "echocardiography artificial intelligence diagnosis",
    "ventricular septal defect machine learning",
    "fetal echocardiography deep learning screening",
    "pediatric cardiology neural network",
    "cardiac ultrasound segmentation transformer",
    "heart disease medical image classification CNN",
    "self-supervised learning medical imaging echocardiography",
    "atrial septal defect deep learning detection",
    "tetralogy of Fallot machine learning ultrasound",
    "cardiac function assessment deep learning echo",
    "point-of-care ultrasound AI heart",
    "multi-modal congenital heart disease diagnosis",
    "explainable AI medical imaging cardiology",
    "semi-supervised learning echocardiography",
    "domain adaptation medical ultrasound cardiac",
    "lightweight deep learning portable echocardiography",
    "3D echocardiography reconstruction deep learning",
    "contrastive learning medical image cardiac",
    "foundation model medical imaging cardiology",
]

new_papers = []
new_dois = set()
paper_id = 39  # Start after existing 38


def search_semantic_scholar(query: str, limit: int = 20) -> list[dict]:
    """Search Semantic Scholar API."""
    try:
        q = urllib.parse.quote(query)
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&limit={limit}&fields=title,authors,journal,year,abstract,externalIds,publicationDate"
        req = urllib.request.Request(url, headers={"User-Agent": "AgentRAG/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        return data.get("data", [])
    except Exception as e:
        print(f"  Search error ({query[:30]}...): {e}")
        return []


print(f"\nSearching across {len(SEARCH_QUERIES)} queries...\n")

for q_idx, query in enumerate(SEARCH_QUERIES):
    print(f"[{q_idx+1}/{len(SEARCH_QUERIES)}] {query[:60]}...", end=" ", flush=True)
    results = search_semantic_scholar(query, limit=15)

    added = 0
    for r in results:
        title = (r.get("title") or "").strip()
        if not title:
            continue

        # Deduplicate by title (fuzzy)
        title_lower = title.lower().rstrip(".")
        if title_lower in existing_titles:
            continue

        # Deduplicate by DOI
        ext_ids = r.get("externalIds") or {}
        doi = (ext_ids.get("DOI") or "").lower().strip()
        if doi and (doi in existing_dois or doi in new_dois):
            continue

        # Deduplicate within new batch
        if title_lower in {p["title"].lower().rstrip(".") for p in new_papers}:
            continue

        # Basic quality filter
        abstract = (r.get("abstract") or "").strip()
        if len(abstract) < 100:
            continue

        # Only papers from 2018+ (deep learning era)
        year = r.get("year") or 0
        if year and year < 2018:
            continue

        authors_list = r.get("authors") or []
        authors = ", ".join(a.get("name", "") for a in authors_list[:5])

        journal = ""
        if r.get("journal"):
            journal = r["journal"].get("name", "") or ""

        # Generate keywords from title + abstract
        text_for_kw = f"{title} {abstract[:500]}"
        keywords = list(set(
            kw for kw in [
                "deep learning", "echocardiography", "congenital heart disease",
                "ultrasound", "CNN", "transformer", "segmentation", "classification",
                "VSD", "ASD", "CHD", "fetal", "pediatric", "self-supervised",
                "cardiac", "medical imaging", "AI", "neural network"
            ]
            if kw.lower() in text_for_kw.lower()
        ))

        paper = {
            "id": paper_id,
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "doi": doi,
            "category": "deep learning - cardiology",
            "keywords": keywords,
            "summary": abstract[:300],
            "en_abstract": abstract[:2000],
            "abstract_source": "semantic_scholar",
            "key_findings": "",
        }
        new_papers.append(paper)
        if doi:
            new_dois.add(doi)
        paper_id += 1
        added += 1

        if len(new_papers) >= 162:  # 38 existing + 162 new = 200
            break

    print(f"+{added} new (total: {len(new_papers)})")

    if len(new_papers) >= 162:
        break

    time.sleep(1.5)  # Rate limit

# ── Save ──
all_papers = existing + new_papers

with open("data/papers_200.json", "w", encoding="utf-8") as f:
    json.dump(all_papers, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"Done! Total: {len(all_papers)} papers ({len(existing)} existing + {len(new_papers)} new)")
print(f"Saved to: data/papers_200.json")
