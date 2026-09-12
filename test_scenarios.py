"""3 realistic coding scenarios: bug fix, feature add, skill routing."""
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

# Initialize memory + skill router
memory = MemorySystem()
router = SkillRouter(skills_dir="./skills")
print(f"Skills loaded: {[s.name for s in router._index]}")
agent = MiniCodeAgent(tools=registry, llm_client=client, memory_system=memory, skill_router=router)

# -- Setup: create a small test project --
os.makedirs("test_project", exist_ok=True)
with open("test_project/utils.py", "w") as f:
    f.write('''
def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)

def greet(name):
    """Return a greeting message."""
    return "Hello " + name
''')

# -- Scenario 1: Bug Fix --
print("=" * 60)
print("SCENARIO 1: Fix a bug in utils.py")
print("=" * 60)
task1 = "There is a bug in test_project/utils.py. When calculate_average is called with an empty list [], it crashes with ZeroDivisionError. Find and fix the bug."
start = time.time()
state = agent.run(task1)
elapsed = time.time() - start

# Memory: learn from this task
memory.learn_from_task(task1, state, success="edit_file" in [s.action for s in state.steps])
print(f"Steps: {len(state.steps)}  Time: {elapsed:.0f}s")
for i, step in enumerate(state.steps):
    action = step.action or "think"
    thought_preview = step.thought[:80]
    print(f"  {i+1}. [{action}] {thought_preview}")
    if step.observation and len(step.observation) > 50 and step.action != "read_file":
        print(f"     -> {step.observation[:200]}")
print(f"  Files edited: {state.files_edited}")

# -- Scenario 2: Feature Addition --
print(f"\n{'=' * 60}")
print("SCENARIO 2: Add a feature to utils.py")
print("=" * 60)
task2 = "Add a function called 'median' to test_project/utils.py that calculates the median of a list of numbers. It should handle empty lists by returning None."
start = time.time()
state = agent.run(task2)
elapsed = time.time() - start
memory.learn_from_task(task2, state, success="edit_file" in [s.action for s in state.steps])
print(f"Steps: {len(state.steps)}  Time: {elapsed:.0f}s")
for i, step in enumerate(state.steps):
    action = step.action or "think"
    print(f"  {i+1}. [{action}] {step.thought[:80]}")
    if step.observation and len(step.observation) > 50 and step.action != "read_file":
        print(f"     -> {step.observation[:200]}")
print(f"  Files edited: {state.files_edited}")

# -- Scenario 3: Code Review (Skill route) --
print(f"\n{'=' * 60}")
print("SCENARIO 3: Code review (should use load_skill)")
print("=" * 60)
task3 = "Do a code review of test_project/utils.py. Check for bugs, style issues, and edge cases."
start = time.time()
state = agent.run(task3)
elapsed = time.time() - start
memory.learn_from_task(task3, state, success="load_skill" in [s.action for s in state.steps])
print(f"Steps: {len(state.steps)}  Time: {elapsed:.0f}s")
skills_used = False
for i, step in enumerate(state.steps):
    action = step.action or "think"
    print(f"  {i+1}. [{action}] {step.thought[:80]}")
    if step.action == "load_skill":
        skills_used = True
        print(f"     *** SKILL LOADED: {step.observation[:100]}")
    elif step.observation and len(step.observation) > 50 and step.action != "read_file":
        print(f"     -> {step.observation[:200]}")
print(f"  Skill used: {skills_used}")

# -- Summary --
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")

# Memory stats
mem_types = {"procedural": 0, "episodic": 0, "preference": 0}
for m in memory._index:
    mem_types[m.type] = mem_types.get(m.type, 0) + 1
print(f"Memories stored: {len(memory._index)} ({mem_types})")

# Test memory retrieval for a similar task
similar_task = "There is a bug in utils.py, can you fix it?"
relevant = memory.retrieve(similar_task, max_results=3)
print(f"Memory recall for '{similar_task[:50]}...': {len(relevant)} matches")
for m in relevant:
    print(f"  [{m.type}] {m.content[:100]}")
print(f"\nAll 3 scenarios completed. Check test_project/utils.py for results.")
