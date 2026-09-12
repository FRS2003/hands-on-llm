"""Diagnostic: run a simple task and dump raw step data to diagnose_result.json."""
import json, os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from mini_code_core import MiniCodeAgent, ToolRegistry, AgentStep
from tools import register_all_tools, set_memory_system
from memory import MemorySystem
from skill_router import SkillRouter

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
memory = MemorySystem()
set_memory_system(memory)

registry = ToolRegistry()
register_all_tools(registry)
router = SkillRouter(skills_dir="./skills")
agent = MiniCodeAgent(tools=registry, llm_client=client, model="deepseek-v4-pro",
                      memory_system=memory, skill_router=router)

task = "读 cli.py 告诉我它做什么"
print(f"Task: {task}")
start = time.time()
state = agent.run(task)
elapsed = time.time() - start

# Dump ALL step data
dump = {
    "task": task,
    "success": any(s.action == "submit_result" for s in state.steps),
    "steps_count": len(state.steps),
    "elapsed": elapsed,
    "steps": []
}
for i, s in enumerate(state.steps):
    dump["steps"].append({
        "index": i + 1,
        "action": s.action,
        "thought": s.thought,
        "action_input": s.action_input,
        "observation": s.observation,
    })

# Last step detail
last = state.steps[-1] if state.steps else None
if last:
    print(f"\nLast step: action={last.action}")
    print(f"  thought: {last.thought}")
    print(f"  action_input ({len(last.action_input or '')} chars): {(last.action_input or '')[:200]}")
    print(f"  observation ({len(last.observation or '')} chars): {(last.observation or '')[:200]}")

with open("diagnose_result.json", "w", encoding="utf-8") as f:
    json.dump(dump, f, ensure_ascii=False, indent=2)
print(f"\nFull dump written to diagnose_result.json ({len(dump['steps'])} steps)")
