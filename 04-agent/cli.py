"""MiniCode CLI — interactive coding assistant."""
import os, sys, json, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from mini_code_core import MiniCodeAgent, ToolRegistry
from tools import register_all_tools, set_memory_system, configure_ssh
from memory import MemorySystem
from skill_router import SkillRouter

# Init
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")
memory = MemorySystem()
set_memory_system(memory)  # Wire memory so load_memory tool can access it

# Configure remote server (GPU: RTX 3080 Ti, 12GB)
configure_ssh(host=os.environ.get("REMOTE_HOST","localhost"),
              port=int(os.environ.get("REMOTE_PORT","22")),
              user=os.environ.get("REMOTE_USER","root"),
              password=os.environ.get("REMOTE_SSH_PASSWORD",""))

registry = ToolRegistry()
register_all_tools(registry)
router = SkillRouter(skills_dir="./skills")
agent = MiniCodeAgent(tools=registry, llm_client=client, model="deepseek-v4-pro",
                      memory_system=memory, skill_router=router)

print("=" * 60)
print("  MiniCode — AI Coding Assistant")
print("  Type your task, or 'quit' to exit, 'help' for examples")
print("=" * 60)

def show_help():
    print("""
Examples:
  Read code:    "Read app.py and explain what it does"
  Find code:    "Find all functions that handle user login"
  Fix bug:      "There's a crash in utils.py when list is empty"
  Add feature:  "Add a median() function to stats.py"
  Code review:  "Review auth.py for security issues"
  Build:        "Create a simple web server in server.py"

Tips:
  - Be specific about which file to work on
  - Describe the bug/feature clearly
  - The agent will read before editing
""")

def log_step(step, idx):
    """Display one agent step."""
    action = step.action or "think"
    icons = {
        "read_file": "[R]", "edit_file": "[E]", "write_file": "[W]",
        "run_command": "[CMD]", "grep": "[G]", "glob_search": "[GLOB]",
        "submit_result": "[OK]", "load_skill": "[SK]", "expand_context": "[EXP]",
        "rejected_text": "[...]",
    }
    icon = icons.get(action, "[?]")
    thought = step.thought[:60] if step.thought else ""
    print(f"  {idx}. {icon} {action}: {thought}")
    if step.observation and action not in ("read_file", "rejected_text"):
        obs = step.observation
        if len(obs) > 200:
            obs = obs[:200] + "..."
        print(f"     {obs}")

# Main loop
session_count = 0
while True:
    try:
        task = input("\nYou> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if not task:
        continue
    if task.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break
    if task.lower() in ("help", "h", "?"):
        show_help()
        continue

    session_count += 1
    print(f"\n[Working on it...]\n")

    start = time.time()
    try:
        state = agent.run(task)
    except Exception as e:
        print(f"\n  [ERROR] Agent crashed: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Handle ask_user pause/resume cycle
    while state.waiting_for_user:
        elapsed = time.time() - start
        for i, step in enumerate(state.steps, 1):
            log_step(step, i)
        print(f"\n{'─' * 50}")
        print(f"  [Agent needs your input]")
        print(f"  {state.user_question}")
        print(f"{'─' * 50}")
        try:
            answer = input("\n  Your answer > ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "skip"
        if not answer:
            answer = "skip"
        print()
        try:
            state = agent.resume(state, answer)
        except Exception as e:
            print(f"\n  [ERROR] Resume crashed: {e}")
            import traceback
            traceback.print_exc()
            break

    elapsed = time.time() - start

    # Show steps
    for i, step in enumerate(state.steps, 1):
        log_step(step, i)

    # Show result
    last_step = state.steps[-1] if state.steps else None
    if last_step:
        result_text = ""
        # 1. submit_result: prefer thought (rich markdown from msg.content)
        if last_step.action == "submit_result":
            t = (last_step.thought or "").strip()
            if t and len(t) > 50 and t not in ("JSON salvage: submit_result", "Calling submit_result"):
                result_text = t
        # 2. submit_result: extract summary from action_input JSON
        if not result_text and last_step.action == "submit_result" and last_step.action_input:
            try:
                args = json.loads(last_step.action_input)
                summary = args.get("summary", "")
                if summary and len(summary) > 10:
                    result_text = summary
            except Exception:
                # JSON parse failed — try regex extraction
                raw = last_step.action_input
                for key in ['"summary": "', "'summary': '", '"summary":"', "'summary':'"]:
                    if key in raw:
                        idx = raw.find(key) + len(key)
                        extracted = raw[idx:].rstrip('"}').rstrip('\'}"')
                        if len(extracted) > 10:
                            result_text = extracted
                        break
        # 3. submit_result: try observation (salvage path stores content here)
        if not result_text and last_step.action == "submit_result" and last_step.observation:
            obs = str(last_step.observation)
            if obs and obs != "Task complete." and len(obs) > 10:
                result_text = obs
        # 4. Plain text output (action=None, max-steps fallback)
        if not result_text and last_step.action is None and last_step.observation:
            result_text = str(last_step.observation)
        # 5. Last resort: scan rejected_text fragments
        if not result_text:
            fragments = []
            for s in state.steps:
                if s.action == "rejected_text" and s.thought:
                    fragments.append(s.thought)
            if fragments:
                result_text = "Partial analysis:\n" + "\n".join(f"  - {f}" for f in fragments)

        # Display
        if result_text and len(result_text) > 20 and "Call a tool" not in result_text:
            print(f"\n{'─' * 50}")
            try:
                # Safe print: handle encoding issues on Windows terminals
                safe = result_text[:2000].encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
                print(safe)
            except Exception:
                print(result_text[:2000])

    # Save memory (only if task actually completed, not paused mid-way)
    success = any(s.action == "submit_result" for s in state.steps)
    if success:
        memory.learn_from_task(task, state, success)

    print(f"\n{'─' * 50}")
    print(f"  {len(state.steps)} steps, {elapsed:.0f}s  "
          f"Files: read={len(state.files_read)} edited={len(state.files_edited)}  "
          f"Memories: {len(memory._index)} stored")
