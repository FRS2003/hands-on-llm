"""Test all 3 models with 2000 max_tokens for task decomposition."""
import os, sys, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")

task = "Fix the crash in utils.py when list is empty, and add unit tests"

for model in ["deepseek-chat", "deepseek-v4-pro", "deepseek-reasoner"]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": (
                f"Break this task into 2-4 sub-tasks:\n\n"
                f"Task: {task}\n\n"
                f'Format as JSON: [{{"description": "...", "worker_type": "reader/editor/tester"}}]'
            )}],
            temperature=0.2, max_tokens=2000,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)
        is_json = content.strip().startswith("[")
        print(f"{model:25s} content={len(content):4d} chars  json={is_json}  reasoning={len(reasoning or ''):4d} chars")
        if is_json:
            parsed = json.loads(content)
            print(f"  -> {len(parsed)} sub-tasks: {[s['description'][:50] for s in parsed]}")
        else:
            print(f"  -> content preview: {repr(content[:120])}")
    except Exception as e:
        print(f"{model:25s} ERROR: {e}")
    print()
