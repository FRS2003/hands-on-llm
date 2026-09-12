import json
with open("data/vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

targets = [
    ("Q3: Cheng 2024 limitations", [12]),
    ("Q12: VSD public datasets", [10, 33]),
    ("Q13: imbalanced classification VSD", [14, 17]),
]

for label, pids in targets:
    print(f"\n{'='*60}")
    print(label)
    for pid in pids:
        p = next(p for p in papers if p["id"] == pid)
        ab = p.get("en_abstract", "")
        has_lim = any(w in ab.lower() for w in ["limitation", "drawback", "shortcoming", "however"])
        has_ds = any(w in ab.lower() for w in ["dataset", "public", "benchmark"])
        has_imb = any(w in ab.lower() for w in ["imbalance", "augmentation", "rare", "over-sample"])
        print(f"\n[{pid}] {p['title'][:60]}")
        print(f"  limitation words: {has_lim} | dataset words: {has_ds} | imbalance words: {has_imb}")
        print(f"  Abstract: {ab[:350]}")
