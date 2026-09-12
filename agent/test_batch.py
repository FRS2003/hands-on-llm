import os
"""Batch test agent on 5 questions (L1, L2, L3)."""
import sys, json, os
sys.path.insert(0, "agent")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir("agent")

from openai import OpenAI
from agent_core_v2 import AutonomousAgent, ToolRegistry
from tools import register_all_tools

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")
registry = ToolRegistry()
register_all_tools(registry)
agent = AutonomousAgent(tools=registry, llm_client=client)

# 5 test questions across difficulty levels
tests = [
    # L1: Simple fact retrieval
    {
        "id": 1, "level": "L1",
        "question": "DINOv2论文有没有在医学影像上做实验？",
        "expected_papers": [21],
    },
    {
        "id": 21, "level": "L1",
        "question": "胎儿超声心动图的AI辅助诊断目前发展到什么程度？",
        "expected_papers": [39],
    },
    # L2: Comparison
    {
        "id": 41, "level": "L2",
        "question": "比较EchoNet-Dynamic和Cheng 2024在患者级数据划分上的做法有什么不同",
        "expected_papers": [10, 12],
    },
    {
        "id": 45, "level": "L2",
        "question": "比较数据增强在VSD超声分类中的不同方法及其效果",
        "expected_papers": [14, 29],
    },
    # L3: Multi-step reasoning
    {
        "id": 81, "level": "L3",
        "question": "综合现有文献，VSD亚型自动分类面临的最大方法论挑战是什么？从数据、模型、评估三个维度回答",
        "expected_papers": [14, 17, 12],
    },
]

for t in tests:
    print(f"\n{'='*70}")
    print(f"[{t['level']}] Q{t['id']}: {t['question'][:80]}")
    print(f"{'='*70}")

    def log_step(step):
        icon = {"search_papers": "🔍", "read_section": "📖", "extract_claims": "📋",
                "compare_papers": "⚖️", "verify_claim": "✅"}.get(step.action, "💭")
        if step.action:
            print(f"  {icon} {step.action}: {step.thought[:80]}...")
        else:
            print(f"  💬 Answer: {(step.observation or '')[:200]}...")

    state = agent.run(t["question"], stream_callback=log_step)

    # Check recall
    found = set()
    for s in state.steps:
        if s.observation:
            try:
                data = json.loads(s.observation)
                for r in data.get("results", []):
                    found.add(r.get("id"))
            except:
                pass

    hit = found & set(t["expected_papers"])
    print(f"  📊 Steps: {len(state.steps)}, Papers found: {sorted(found)[:10]}, Expected: {t['expected_papers']}, Hit: {sorted(hit)}")
