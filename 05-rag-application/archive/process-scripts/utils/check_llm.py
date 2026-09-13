import json
with open("data/vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

for pid in [12, 10, 33]:
    p = next(p for p in papers if p["id"] == pid)
    ab = p.get("en_abstract", "")
    # Find the LLM summary part
    if "[LLM Summary]" in ab:
        parts = ab.split("[LLM Summary]")
        print(f"
=== [{pid}] {p['title'][:50]} ===")
        print(f"Original ({len(parts[0])} chars): {parts[0][:200]}...")
        print(f"LLM Summary ({len(parts[1])} chars): {parts[1][:300]}...")
