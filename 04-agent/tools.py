"""
MiniCode Tools -- coding-specific tools for file ops, search, and command execution.

Each tool is a standalone function. ToolRegistry dispatches them.
All tools operate within the project workspace directory for safety.
"""
import os
import re
import subprocess
import glob as glob_module
import json as _json

# Module-level memory reference — set by cli.py at startup
_memory_system = None

# SSH server config (set by cli.py or defaults)
_ssh_host = "localhost"
_ssh_port = 27006
_ssh_user = "root"
_ssh_password = ""


def configure_ssh(host=os.environ.get("REMOTE_HOST","localhost"),
              port=int(os.environ.get("REMOTE_PORT","22")),
              user=os.environ.get("REMOTE_USER","root"),
              password=os.environ.get("REMOTE_SSH_PASSWORD",""))


def _safe_path(path: str) -> str:
    """Resolve path relative to workspace. Block paths that escape workspace."""
    abs_path = os.path.normpath(os.path.join(WORKSPACE, path))
    if not abs_path.startswith(os.path.normpath(WORKSPACE)):
        raise ValueError(f"Path escapes workspace: {path}")
    return abs_path


# -- Tool Implementations --

def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> dict:
    """
    Read a file. Returns content with line numbers.
    Args:
        path: File path relative to workspace.
        start_line: First line to read (1-based).
        end_line: Last line to read (None = read to end).
    """
    try:
        abs_path = _safe_path(path)
        if not os.path.exists(abs_path):
            return {"error": f"File not found: {path}"}
        if os.path.isdir(abs_path):
            files = os.listdir(abs_path)[:30]
            return {"type": "directory", "path": path, "contents": files}

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        total = len(lines)
        start = max(1, start_line) - 1
        end = min(end_line or total, total)
        selected = lines[start:end]

        output = []
        for i, line in enumerate(selected, start=start + 1):
            output.append(f"{i:4d}| {line.rstrip()}")

        return {
            "path": path, "total_lines": total,
            "showing": f"{start+1}-{end}",
            "content": "\n".join(output),
        }
    except ValueError as e:
        return {"error": str(e)}


def grep(pattern: str, path: str = ".", max_results: int = 20) -> dict:
    """
    Search for a regex pattern in files. Returns matching lines with file paths.
    Args:
        pattern: Regex pattern to search for.
        path: File or directory to search (default: entire workspace).
        max_results: Max number of matches to return.
    """
    try:
        abs_path = _safe_path(path)
        results = []

        if os.path.isfile(abs_path):
            files_to_search = [abs_path]
        else:
            files_to_search = []
            for root, dirs, files in os.walk(abs_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"
                           and d != "node_modules" and d != ".git"]
                for fname in files:
                    if fname.endswith((".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml",
                                       ".html", ".css", ".txt", ".sh", ".toml", ".cfg")):
                        files_to_search.append(os.path.join(root, fname))

        for fpath in files_to_search[:100]:  # Limit total files searched
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            rel_path = os.path.relpath(fpath, WORKSPACE)
                            results.append({
                                "file": rel_path, "line": i,
                                "content": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                break
            except Exception:
                pass
            if len(results) >= max_results:
                break

        return {"pattern": pattern, "matches": len(results), "results": results[:max_results]}
    except ValueError as e:
        return {"error": str(e)}


def glob_search(file_pattern: str, path: str = ".") -> dict:
    """
    Find files matching a glob pattern.
    Args:
        file_pattern: Glob pattern (e.g., "*.py", "src/**/*.ts").
        path: Directory to search in.
    """
    try:
        abs_path = _safe_path(path)
        pattern = os.path.join(abs_path, file_pattern)
        matches = glob_module.glob(pattern, recursive=True)
        rel_matches = [os.path.relpath(m, WORKSPACE) for m in matches[:50]]
        return {"pattern": file_pattern, "count": len(rel_matches), "files": rel_matches}
    except ValueError as e:
        return {"error": str(e)}


def run_command(cmd: str = "", command: str = "") -> dict:
    """
    Execute a shell command. Use sparingly -- prefer edit_file for code changes.
    Args:
        cmd: Shell command to execute.
        command: Alias for cmd (LLMs frequently use this name).
    """
    # Accept both parameter names (LLMs often mix these up)
    cmd = cmd or command
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=30, cwd=WORKSPACE,
        )
        return {
            "command": cmd,
            "exit_code": result.returncode,
            "stdout": result.stdout[:2000] if result.stdout else "",
            "stderr": result.stderr[:500] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out (30s): {cmd[:80]}"}
    except Exception as e:
        return {"error": str(e)}


def remote_run(cmd: str, host: str = "", port: int = 0, user: str = "",
               password: str = "", timeout: int = 60) -> dict:
    """
    Execute a command on a remote GPU server via SSH.
    Use the host/port/user/password the user provides. If omitted,
    uses the default server configured at startup.
    Args:
        cmd: Shell command to run on the remote server.
        host: Server hostname (uses default if empty).
        port: SSH port (uses default if 0).
        user: SSH username (uses default if empty).
        password: SSH password (uses default if empty).
        timeout: Max seconds to wait (default 60).
    """
    h = host or _ssh_host
    p = port or _ssh_port
    u = user or _ssh_user
    pw = password or _ssh_password
    if not pw:
        return {"error": "No SSH password available. Ask the user for server credentials."}
    try:
        import paramiko
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(h, port=p, username=u, password=pw, timeout=5)
        # Ensure conda Python is on PATH (server default: /root/miniconda3/bin)
        full_cmd = f"export PATH=/root/miniconda3/bin:$PATH; {cmd}"
        stdin, stdout, stderr = client.exec_command(full_cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        client.close()
        return {
            "host": f"{h}:{p}",
            "command": cmd,
            "exit_code": exit_code,
            "stdout": out[-3000:] if len(out) > 3000 else out,
            "stderr": err[-500:] if len(err) > 500 else err,
        }
    except ImportError:
        return {"error": "paramiko not installed. Run: pip install paramiko"}
    except Exception as e:
        return {"error": f"SSH failed: {str(e)[:200]}"}


def write_file(path: str, content: str) -> dict:
    """
    Create a new file. Fails if file already exists.
    Args:
        path: File path relative to workspace.
        content: File contents.
    """
    try:
        abs_path = _safe_path(path)
        if os.path.exists(abs_path):
            return {"error": f"File already exists: {path}. Use edit_file to modify."}
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)
        lines = content.count("\n") + 1
        return {"path": path, "status": "created", "lines": lines}
    except ValueError as e:
        return {"error": str(e)}


def edit_file(path: str, old_string: str, new_string: str) -> dict:
    """
    Replace text in an existing file. old_string must match exactly.
    Args:
        path: File path relative to workspace.
        old_string: Exact text to replace.
        new_string: Replacement text.
    """
    try:
        abs_path = _safe_path(path)
        if not os.path.exists(abs_path):
            return {"error": f"File not found: {path}"}
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
        if old_string not in content:
            return {"error": "old_string not found in file. Read the file first to get exact text."}
        new_content = content.replace(old_string, new_string, 1)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"path": path, "status": "edited", "replacements": 1}
    except ValueError as e:
        return {"error": str(e)}


def submit_result(summary: str) -> dict:
    """
    Submit the final result. This terminates the agent loop.
    Args:
        summary: A clear summary of what was done, files changed, and why.
    """
    return {"status": "submitted", "summary": summary}


def expand_context(cache_id: str) -> dict:
    """
    Retrieve a previously compressed/truncated tool output.
    Args:
        cache_id: The cache ID from an [EXPAND:xxx] placeholder.
    """
    # This is handled by the ContextManager, not here
    return {"cache_id": cache_id, "note": "Use the expand_context tool from the agent."}


def ask_user(question: str) -> dict:
    """
    Ask the user a question when you need clarification, missing information,
    or want approval before proceeding. Use this when:
    - You can't find a file and need to know where it is
    - You need credentials, paths, or configuration details
    - You're stuck and need guidance on what to do next
    - You want to confirm a potentially risky operation
    Args:
        question: The question to ask the user.
    """
    return {"status": "waiting", "question": question, "note": "CLI will pause for user input."}


def load_skill(skill_name: str) -> dict:
    """
    Load a specialized Skill prompt. Skills are in ./skills/<skill_name>/SKILL.md
    Args:
        skill_name: Skill name (e.g., "code-review", "fix-bug").
    """
    skill_path = os.path.join(WORKSPACE, "skills", skill_name, "SKILL.md")
    if not os.path.exists(skill_path):
        return {"error": f"Skill '{skill_name}' not found at {skill_path}"}
    try:
        with open(skill_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"skill": skill_name, "content": content[:2000]}
    except Exception as e:
        return {"error": str(e)}


def load_memory(topic: str) -> dict:
    """
    On-demand memory search. Call this when you need to recall past experiences
    about a specific topic (e.g., "sql injection fix", "auth refactor pattern").
    Args:
        topic: What you want to recall — be specific (error type, module name, task type).
    """
    if _memory_system is None:
        return {"results": [], "hint": "Memory system not available. Proceed without it."}
    memories = _memory_system.search(topic, max_results=3)
    if not memories:
        return {"results": [], "hint": "No relevant past experiences found."}
    return {
        "results": [
            {
                "type": m.type,
                "content": m.content,
                "tags": m.tags,
                "success": m.success,
            }
            for m in memories
        ]
    }


# -- Tool Schemas for OpenAI Function Calling --

TOOL_SCHEMAS = {
    "read_file": {
        "description": "Read a file. Returns content with line numbers. Use BEFORE editing any file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace."},
                "start_line": {"type": "integer", "description": "First line to read (1-based, default 1)."},
                "end_line": {"type": "integer", "description": "Last line to read (default: end of file)."},
            },
            "required": ["path"],
        },
    },
    "grep": {
        "description": "Search for a regex pattern in files. Returns matching lines with file paths and line numbers.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for."},
                "path": {"type": "string", "description": "File or directory to search (default: workspace root)."},
                "max_results": {"type": "integer", "description": "Max results (default 20)."},
            },
            "required": ["pattern"],
        },
    },
    "glob_search": {
        "description": "Find files matching a glob pattern. Useful for discovering project structure.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' or 'src/**/*.ts'."},
                "path": {"type": "string", "description": "Directory to search (default: workspace root)."},
            },
            "required": ["file_pattern"],
        },
    },
    "run_command": {
        "description": "Execute a shell command locally. Use sparingly -- prefer code tools for changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["cmd"],
        },
    },
    "remote_run": {
        "description": "Execute a command on a remote server via SSH. If the user provides server details (host, port, username, password), use them. Otherwise uses the default server.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Shell command to run on the remote server."},
                "host": {"type": "string", "description": "Server hostname (use what the user told you)."},
                "port": {"type": "integer", "description": "SSH port (use what the user told you)."},
                "user": {"type": "string", "description": "SSH username (usually root)."},
                "password": {"type": "string", "description": "SSH password (use what the user told you)."},
                "timeout": {"type": "integer", "description": "Max seconds to wait (default 60)."},
            },
            "required": ["cmd"],
        },
    },
    "write_file": {
        "description": "Create a NEW file. Fails if file exists. Use edit_file to modify existing files.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace."},
                "content": {"type": "string", "description": "Full file content."},
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "description": "Replace exact text in a file. Read the file FIRST to get the exact old_string.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to workspace."},
                "old_string": {"type": "string", "description": "Exact text to replace (must match character-for-character)."},
                "new_string": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    "submit_result": {
        "description": "Submit the final result. Call this when the task is complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of what was done and why."},
            },
            "required": ["summary"],
        },
    },
    "ask_user": {
        "description": "Pause and ask the user for clarification, missing info, or approval. Use when stuck or when you need input to continue.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "What to ask the user."},
            },
            "required": ["question"],
        },
    },
    "load_skill": {
        "description": "Load a specialized Skill prompt for complex tasks (code-review, fix-bug, add-feature, refactor).",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {"type": "string", "description": "Skill name: code-review, fix-bug, add-feature, or refactor."},
            },
            "required": ["skill_name"],
        },
    },
    "load_memory": {
        "description": "Search past experiences for relevant knowledge. Use when you need to recall how you solved a similar problem before (e.g. 'sql injection fix', 'auth refactor pattern'). Call this BEFORE attempting a complex edit you're unsure about.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "What to recall — be specific: error type, module name, task type, bug pattern."},
            },
            "required": ["topic"],
        },
    },
}


def register_all_tools(registry) -> None:
    """Register all tools into a ToolRegistry instance."""
    registry.register("read_file", read_file, TOOL_SCHEMAS["read_file"]["description"], TOOL_SCHEMAS["read_file"]["parameters"])
    registry.register("grep", grep, TOOL_SCHEMAS["grep"]["description"], TOOL_SCHEMAS["grep"]["parameters"])
    registry.register("glob_search", glob_search, TOOL_SCHEMAS["glob_search"]["description"], TOOL_SCHEMAS["glob_search"]["parameters"])
    registry.register("run_command", run_command, TOOL_SCHEMAS["run_command"]["description"], TOOL_SCHEMAS["run_command"]["parameters"])
    registry.register("remote_run", remote_run, TOOL_SCHEMAS["remote_run"]["description"], TOOL_SCHEMAS["remote_run"]["parameters"])
    registry.register("write_file", write_file, TOOL_SCHEMAS["write_file"]["description"], TOOL_SCHEMAS["write_file"]["parameters"])
    registry.register("edit_file", edit_file, TOOL_SCHEMAS["edit_file"]["description"], TOOL_SCHEMAS["edit_file"]["parameters"])
    registry.register("submit_result", submit_result, TOOL_SCHEMAS["submit_result"]["description"], TOOL_SCHEMAS["submit_result"]["parameters"])
    registry.register("ask_user", ask_user, TOOL_SCHEMAS["ask_user"]["description"], TOOL_SCHEMAS["ask_user"]["parameters"])
    registry.register("load_skill", load_skill, TOOL_SCHEMAS["load_skill"]["description"], TOOL_SCHEMAS["load_skill"]["parameters"])
    registry.register("load_memory", load_memory, TOOL_SCHEMAS["load_memory"]["description"], TOOL_SCHEMAS["load_memory"]["parameters"])
