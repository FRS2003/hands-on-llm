import os
"""Quick recall test on 20 questions (no agent loop, just search_papers)."""
import sys, json, os
sys.path.insert(0, "agent")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir("agent")

from tools import search_papers

with open("data/benchmark_100.json", "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Take first 20 L1+L2 questions that have expected papers
test_questions = [q for q in benchmark["questions"][:40] if q.get("expected_papers")][:20]

hits = 0
total = 0
print(f"{'Q':4s} {'Lvl':4s} {'Hit':6s} | Expected -> Got")
print("-" * 50)

for q in test_questions:
    qid = q["id"]
    level = q["level"]
    expected = set(q.get("expected_papers", []))

    results = search_papers(q["question"])
    found = set(r["id"] for r in results.get("results", [])[:5])

    hit = found & expected
    hits += len(hit)
    total += len(expected)

    print(f"{qid:3d}  {level:4s}  {len(hit)}/{len(expected):3d} | {sorted(expected)} -> {sorted(found)[:5]}")

print(f"\nRecall: {hits}/{total} = {hits/total:.1%}")
