"""Test: MiniCode builds a calculator from scratch."""
import os, sys, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from mini_code_core import MiniCodeAgent, ToolRegistry
from tools import register_all_tools
from memory import MemorySystem
from skill_router import SkillRouter

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
registry = ToolRegistry()
register_all_tools(registry)
memory = MemorySystem()
router = SkillRouter(skills_dir="./skills")
agent = MiniCodeAgent(tools=registry, llm_client=client, memory_system=memory, skill_router=router)

task = (
    "Create a calculator program in test_calc/calc.py with these features:\n"
    "1. Functions: add, subtract, multiply, divide (handle division by zero)\n"
    "2. A main() function that reads input like '5 + 3' and prints the result\n"
    "3. Support for +, -, *, / operators\n"
    "4. Type 'quit' to exit\n\n"
    "Write the complete program and test it with run_command."
)

print(f"Task: Build a calculator\n")
start = time.time()
state = agent.run(task)
elapsed = time.time() - start

print(f"Steps: {len(state.steps)}  Time: {elapsed:.0f}s")
for i, step in enumerate(state.steps):
    action = step.action or "think"
    print(f"  {i+1}. [{action}] {step.thought[:80]}")
    if step.observation and step.action not in ("read_file",):
        obs = step.observation
        if len(obs) > 300:
            obs = obs[:300] + "..."
        print(f"     -> {obs}")

# Show the result
print(f"\n{'='*40}")
if os.path.exists("test_calc/calc.py"):
    with open("test_calc/calc.py", "r") as f:
        print(f.read())
