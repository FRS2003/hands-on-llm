import json
with open("data/vsd_rag_enriched.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
with open("data/pmc_fulltexts.json", "r") as f:
    pmc = {int(k): v for k, v in json.load(f).items()}

for p in papers:
    pid = p["id"]
    ab = p.get("en_abstract", "")
    pmc_text = pmc.get(pid, "")
    print(f"[{pid:2d}] ab={len(ab):5d}  pmc={len(pmc_text):5d}  src={p.get('abstract_source','?'):20s}  {p['title'][:45]}")
