import json
with open("data/vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
with open("data/pmc_fulltexts.json", "r") as f:
    pmc = {int(k): v for k, v in json.load(f).items()}

print("Papers with PMC full text but short abstract:")
for p in papers:
    pid = p["id"]
    ab_len = len(p.get("en_abstract", ""))
    has_pmc = pid in pmc
    if ab_len < 200 and has_pmc:
        print(f"  [{pid}] ab={ab_len} chars, PMC={len(pmc[pid])} chars  {p['title'][:50]}")
