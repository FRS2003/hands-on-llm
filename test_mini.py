"""Quick test: MiniCode reads a file and reports structure."""
import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from mini_code_core import MiniCodeAgent, ToolRegistry
from tools import register_all_tools

client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
registry = ToolRegistry()
register_all_tools(registry)
agent = MiniCodeAgent(tools=registry, llm_client=client)

# Simple test: ask it to explore its own source code
task = "Read mini_code_core.py and tell me what the main class does and what methods it has."

print(f"Task: {task}\n")
state = agent.run(task)

print(f"Steps: {len(state.steps)}")
for i, step in enumerate(state.steps):
    action = step.action or "(think)"
    print(f"  {i+1}. [{action}] {step.thought[:80]}")
    if step.observation:
        obs = step.observation
        if len(obs) > 500:
            print(f"     {obs[:500]}")
            print(f"     ... ({len(obs)} chars total)")
        else:
            print(f"     {obs}")

print(f"\nFiles read: {state.files_read}")
print(f"Files edited: {state.files_edited}")
