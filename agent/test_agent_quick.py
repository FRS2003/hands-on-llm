import os
import sys, json, os
sys.path.insert(0, "agent")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.chdir("agent")

from openai import OpenAI
from agent_core import ResearchAgent, ToolRegistry
from tools import register_all_tools

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY",""), base_url="https://api.deepseek.com")
registry = ToolRegistry()
register_all_tools(registry)
agent = ResearchAgent(tools=registry, llm_client=client)

question = "DINOv2论文有没有在医学影像上做实验？"
print(f"Question: {question}")
print("=" * 60)

def log_step(step):
    print(f"\n[Step] {step.thought[:100]}...")
    if step.action:
        print(f"  Tool: {step.action}")
        obs = (step.observation or "")[:200]
        print(f"  Result: {obs}...")
    else:
        print(f"  Answer: {(step.observation or '')[:400]}...")

state = agent.run(question, stream_callback=log_step)
print(f"\nDone in {len(state.steps)} steps.")
print(f"Tools used: {[s.action for s in state.steps if s.action]}")

# Check if correct paper was found
for step in state.steps:
    if step.action == "search_papers" and step.observation:
        if "21" in step.observation or "DINOv2" in step.observation:
            print("PASS: Found DINOv2 paper [21]")
            break
else:
    print("CHECK: Verify if paper [21] was retrieved")
