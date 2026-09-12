import json, re

with open("data/pmc_fulltexts.json", "r") as f:
    pmc = {int(k): v for k, v in json.load(f).items()}

for pid in [12, 33, 10]:
    text = pmc.get(pid, "")
    sents = re.split(r"[.!?] ", text)
    print(f"\n=== [{pid}] ===")

    keywords = ["limitation", "class", "subtype", "three-class", "dataset", "public", "open", "benchmark", "release"]
    for kw in keywords:
        matches = [s.strip() for s in sents if kw.lower() in s.lower() and len(s) > 40]
        for m in matches[:1]:
            print(f"  [{kw}]: {m[:200]}")
