import json
with open("data/vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

for pid in [12, 10, 33]:
    p = next(p for p in papers if p["id"] == pid)
    ab = p.get("en_abstract", "")
    if "[LLM Summary]" in ab:
        parts = ab.split("[LLM Summary]")
        print(f"\n=== [{pid}] {p['title'][:50]} ===")
        print(f"LLM: {parts[1][:400]}")
        # Check for key terms
        llm = parts[1].lower()
        print(f"  has 'limitation': {'limitation' in llm}")
        print(f"  has 'dataset' or 'public': {'dataset' in llm or 'public' in llm}")
        print(f"  has 'VSD': {'vsd' in llm}")
