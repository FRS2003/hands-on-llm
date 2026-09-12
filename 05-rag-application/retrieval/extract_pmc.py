import json, re

with open("data/pmc_fulltexts.json", "r") as f:
    pmc = {int(k): v for k, v in json.load(f).items()}

for pid in [12, 10, 33, 14, 17]:
    text = pmc.get(pid, "")
    if not text:
        print(f"\n[{pid}] NO PMC TEXT")
        continue

    sentences = re.split(r"[.!?] ", text)

    lim = [s for s in sentences if any(w in s.lower() for w in ["limitation", "drawback", "shortcoming", "however", "only three", "not distinguish"])][:3]
    ds = [s for s in sentences if any(w in s.lower() for w in ["dataset", "public", "benchmark", "open", "release", "available"])][:3]
    aug = [s for s in sentences if any(w in s.lower() for w in ["augment", "imbalance", "rare class", "over-sample", "smote", "minority"])][:3]

    print(f"\n{'='*50}")
    print(f"[{pid}] PMC: {len(text)} chars")
    if lim: print(f"  LIM: {lim[0][:200]}")
    if ds: print(f"  DS:  {ds[0][:200]}")
    if aug: print(f"  AUG: {aug[0][:200]}")
