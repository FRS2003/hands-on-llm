"""
MiniCode Agent Core -- Harness Engineered Coding Agent.

Architecture (from v4 Agent project):
  Query Loop: Thought -> Tool Call -> Observation -> repeat
  Harness: Tool Curfew, submit_result, no-progress detection
  Skills: Lazy-load specialized prompts via load_skill tool

Key difference from research Agent: tools are coding tools (read_file, grep,
edit_file, etc.) instead of academic tools (search_papers, read_section, etc.)
"""
import json
import os
import time
import threading
import re
from dataclasses import dataclass, field
from typing import Any, Callable


def _sanitize(text: str) -> str:
    """Remove surrogate characters and invalid Unicode that break JSON/API/UTF-8."""
    if not isinstance(text, str):
        return str(text) if text else ""
    # surrogateescape → replace: handles all invalid Unicode safely
    return text.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
from openai import OpenAI


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


@dataclass
class AgentState:
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 15
    files_read: set = field(default_factory=set)
    files_edited: set = field(default_factory=set)
    consecutive_no_progress: int = 0
    consecutive_errors: int = 0       # track repeated failures
    last_action_count: int = 0
    force_finish: bool = False
    loaded_skills: set = field(default_factory=set)
    waiting_for_user: bool = False
    user_question: str = ""


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, description: str, parameters: dict):
        self._tools[name] = {"fn": fn, "description": description, "parameters": parameters}

    def get_tool_definitions(self) -> list[dict]:
        return [
            {"type": "function", "function": {"name": n, "description": i["description"], "parameters": i["parameters"]}}
            for n, i in self._tools.items()
        ]

    def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Tool '{name}' not found"})
        try:
            return json.dumps(self._tools[name]["fn"](**kwargs), ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


SYSTEM_PROMPT = """You are MiniCode, a terminal coding assistant. You help with reading, understanding, and modifying code.

Workflow: STOP and THINK first. Do NOT jump to run_command.
1. CHECK: Does the required file/script exist? Use glob_search first.
2. CREATE: If not found, write it with write_file. You CAN write full scripts.
3. RUN: Only after the file exists, use run_command to test it.
4. FIX: If it errors, read the file, edit it, then run again.
5. DONE: Call submit_result with what you did and the output.

Available tools:
- read_file(path): Read a file's contents
- grep(pattern, path): Search for text patterns in files
- glob(pattern): Find files matching a pattern
- run_command(cmd): Execute a shell command locally
- remote_run(cmd): Execute on the remote GPU server via SSH
- write_file(path, content): Create a NEW file with any content (full scripts, configs, etc.)
- edit_file(path, old_string, new_string): Replace exact text in an existing file
- load_skill(name): Load a specialized Skill prompt for complex tasks
- load_memory(topic): Search past experiences — use when unsure how to approach a problem
- ask_user(question): Pause and ask the user for clarification when stuck
- submit_result(summary): Deliver your final result (call this when done)

Rules:
1. ALWAYS read a file before editing it -- never edit blind.
2. Use grep/glob to find relevant code before making changes.
3. If a required script or file doesn't exist, CREATE it with write_file.
4. Make ONE change at a time, verify with run_command, then proceed.
5. After making changes, explain what you did and why.
6. When done, call submit_result with a clear summary.

Current task: {task}

Session history:
{history}

{progress_warning}
What should I do next?"""


class MiniCodeAgent:
    """Coding Agent with Harness Engineering."""

    def __init__(self, tools: ToolRegistry, llm_client: OpenAI, model: str = "deepseek-chat",
                 skills_dir: str = "./skills", memory_system=None, skill_router=None,
                 context_manager=None):
        self.tools = tools
        self.client = llm_client
        self.model = model
        self.skills_dir = skills_dir
        self.memory = memory_system
        self.skill_router = skill_router
        self.ctx_manager = context_manager

    def run(self, task: str, stream_callback: Callable | None = None) -> AgentState:
        task = _sanitize(task)
        state = AgentState(task=task)

        # Skill routing: auto-select best skill for this task
        # (Memory is NOT pre-injected — Agent calls load_memory tool on demand)
        skill_context = ""
        if self.skill_router:
            selected_skill, skill_content = self.skill_router.route(task, self.client, self.model)
            if selected_skill and skill_content:
                skill_context = f"## Active Skill: {selected_skill.name}\n\n{skill_content}"
                state.loaded_skills.add(selected_skill.name)

        while state.step_count < state.max_steps:
            progress_warning = self._get_progress_warning(state)
            history_text = self._format_history(state)
            prompt = SYSTEM_PROMPT.format(task=task, history=history_text, progress_warning=progress_warning)

            # Inject skill into prompt (memory is on-demand via load_memory tool)
            if skill_context:
                prompt = prompt.replace("Current task:", f"{skill_context}\n\nCurrent task:")

            available_tools = self._get_tools_for_step(state)
            tc = "auto"

            # ── API call with timeout + retry ──
            max_tok = 2000 if "pro" in self.model or "reasoner" in self.model else 500
            api_kwargs = {
                "model": self.model, "messages": [{"role": "system", "content": _sanitize(prompt)}],
                "tools": available_tools, "tool_choice": tc,
                "temperature": 0.3, "max_tokens": max_tok,
                "timeout": 30,  # API-level timeout
            }
            # Disable thinking for v4-pro too — reasoning consumes token budget,
            # leaving truncated/empty tool call content
            if self.model in ("deepseek-v4-pro", "deepseek-v4-flash"):
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            for attempt in range(2):  # retry once on network errors
                try:
                    response = self.client.chat.completions.create(**api_kwargs)
                    break  # success
                except Exception as api_err:
                    if attempt == 0 and self._is_retryable(api_err):
                        time.sleep(1.5)  # brief backoff
                        continue
                    # Final failure: log and force finish
                    state.steps.append(AgentStep(
                        f"API failed (attempt {attempt+1}/2)", "api_error", None,
                        json.dumps({"error": str(api_err)[:200]})
                    ))
                    state.step_count += 1
                    state.consecutive_errors += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break
            else:
                # exhausted retries — skip to consecutive error check
                pass

            # If API call failed after retries, check if we should continue
            if state.steps and state.steps[-1].action == "api_error":
                if state.consecutive_errors >= 3:
                    state.force_finish = True
                continue

            msg = response.choices[0].message
            state.consecutive_errors = 0  # reset on success

            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    # submit_result with long text often fails JSON parsing.
                    # Salvage the raw content as the final answer.
                    if tool_name == "submit_result":
                        raw = tool_call.function.arguments
                        content = raw
                        # Try to extract just the summary value from broken JSON
                        for key in ['"summary": "', "'summary': '", '"summary":"', "'summary':'"]:
                            if key in raw:
                                idx = raw.find(key) + len(key)
                                content = raw[idx:].rstrip('"}').rstrip('\'}"')
                                break
                        # Store extracted content as valid JSON so CLI can parse it
                        salvaged_json = json.dumps({"summary": content}, ensure_ascii=False)
                        state.steps.append(AgentStep(
                            "JSON salvage: submit_result", "submit_result",
                            salvaged_json, content[:2000]))
                        state.step_count += 1
                        if stream_callback: stream_callback(state.steps[-1])
                        break  # exit while loop — Agent is done
                    # Non-submit_result JSON failure: log and continue
                    state.steps.append(AgentStep(
                        f"JSON error: {e}", tool_name,
                        tool_call.function.arguments[:100],
                        json.dumps({"error": f"Malformed JSON: {str(e)}"})
                    ))
                    state.step_count += 1
                    state.consecutive_errors += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    continue

                thought = msg.content or f"Calling {tool_name}"

                if tool_name == "ask_user":
                    question = tool_args.get("question", "What should I do next?")
                    state.user_question = question
                    state.waiting_for_user = True
                    state.steps.append(AgentStep(thought, "ask_user", json.dumps(tool_args), "Waiting for user response..."))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break

                if tool_name == "submit_result":
                    state.steps.append(AgentStep(thought, "submit_result", json.dumps(tool_args), "Task complete."))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break

                # ── Tool execution with timeout ──
                observation = self._execute_tool_safe(state, tool_name, tool_args)
                state.steps.append(AgentStep(thought, tool_name, json.dumps(tool_args), observation))
                state.step_count += 1

                # Track consecutive tool execution errors (separate from API errors)
                try:
                    obs_data = json.loads(observation)
                    if "error" in obs_data:
                        state.consecutive_errors += 1
                    else:
                        state.consecutive_errors = 0
                except Exception:
                    state.consecutive_errors = 0

                self._check_progress(state)
                if stream_callback: stream_callback(state.steps[-1])

            else:
                content = msg.content or ""
                # Accept any substantial text as the answer (not just tool-call thinking)
                if len(content) > 100:
                    state.steps.append(AgentStep("Analysis complete.", None, None, content))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break
                else:
                    # Short text — likely transitional thinking, not final answer.
                    # Save the text as thought so we can recover it if max_steps is hit.
                    state.steps.append(AgentStep(
                        content[:200], "rejected_text", None,
                        "Call a tool or use submit_result to deliver your analysis."
                    ))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])

        if state.step_count >= state.max_steps:
            parts = []
            # 1. Collect rejected short text fragments
            fragments = []
            for s in state.steps:
                if s.action == "rejected_text" and s.thought:
                    fragments.append(s.thought)
            if fragments:
                parts.append("──── Agent's partial analysis ────")
                for i, f in enumerate(fragments):
                    parts.append(f"  [{i+1}] {f}")
            # 2. Collect last 3 tool actions + observation summaries
            tool_steps = [s for s in state.steps if s.action and s.action not in ("rejected_text", "submit_result")]
            if tool_steps:
                parts.append("\n──── Last tool calls ────")
                for s in tool_steps[-3:]:
                    obs_summary = ""
                    if s.observation:
                        try:
                            obs_data = json.loads(s.observation)
                            obs_summary = obs_data.get("stdout", "") or obs_data.get("error", "")
                            if not obs_summary:
                                obs_summary = json.dumps(obs_data, ensure_ascii=False)
                        except Exception:
                            obs_summary = s.observation
                    parts.append(f"  [{s.action}] {obs_summary[:200]}")
            # 3. Files touched
            parts.append(f"\nFiles read: {sorted(state.files_read) or '(none)'}")
            parts.append(f"Files edited: {sorted(state.files_edited) or '(none)'}")

            summary = "\n".join(parts) if len(parts) > 2 else (
                "⚠️ Max steps reached. Agent produced no analyzable output.\n"
                + f"Files read: {sorted(state.files_read) or '(none)'}. "
                + f"Files edited: {sorted(state.files_edited) or '(none)'}."
            )
            state.steps.append(AgentStep("Max steps reached.", None, None, summary))
            if stream_callback: stream_callback(state.steps[-1])

        return state

    def resume(self, state: AgentState, user_answer: str,
               stream_callback: Callable | None = None) -> AgentState:
        """Continue a paused Agent session after user provides input to ask_user."""
        state.waiting_for_user = False
        # Feed user's answer as an observation to the last ask_user step
        if state.steps and state.steps[-1].action == "ask_user":
            state.steps[-1].observation = f"User answered: {user_answer}"
        state.step_count += 1  # count the user interaction as a step

        task = state.task
        skill_context = ""
        if self.skill_router:
            for skill_name in list(state.loaded_skills):
                skill_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
                if os.path.exists(skill_path):
                    with open(skill_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        content = parts[2].strip() if len(parts) >= 3 else content
                    skill_context = f"## Active Skill: {skill_name}\n\n{content[:2000]}"
                    break

        # Continue the same while loop
        while state.step_count < state.max_steps:
            progress_warning = self._get_progress_warning(state)
            history_text = self._format_history(state)
            prompt = SYSTEM_PROMPT.format(task=task, history=history_text, progress_warning=progress_warning)

            if skill_context:
                prompt = prompt.replace("Current task:", f"{skill_context}\n\nCurrent task:")

            available_tools = self._get_tools_for_step(state)
            tc = "auto"

            max_tok = 2000 if "pro" in self.model or "reasoner" in self.model else 500
            api_kwargs = {
                "model": self.model, "messages": [{"role": "system", "content": _sanitize(prompt)}],
                "tools": available_tools, "tool_choice": tc,
                "temperature": 0.3, "max_tokens": max_tok,
                "timeout": 30,
            }
            if self.model in ("deepseek-v4-pro", "deepseek-v4-flash"):
                api_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            for attempt in range(2):
                try:
                    response = self.client.chat.completions.create(**api_kwargs)
                    break
                except Exception as api_err:
                    if attempt == 0 and self._is_retryable(api_err):
                        time.sleep(1.5)
                        continue
                    state.steps.append(AgentStep(
                        f"API failed (attempt {attempt+1}/2)", "api_error", None,
                        json.dumps({"error": str(api_err)[:200]})
                    ))
                    state.step_count += 1
                    state.consecutive_errors += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break
            else:
                pass

            if state.steps and state.steps[-1].action == "api_error":
                if state.consecutive_errors >= 3:
                    state.force_finish = True
                continue

            msg = response.choices[0].message
            state.consecutive_errors = 0

            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    if tool_name == "submit_result":
                        raw = tool_call.function.arguments
                        content = raw
                        for key in ['"summary": "', "'summary': '", '"summary":"', "'summary':'"]:
                            if key in raw:
                                idx = raw.find(key) + len(key)
                                content = raw[idx:].rstrip('"}').rstrip('\'}"')
                                break
                        state.steps.append(AgentStep(
                            "JSON salvage: submit_result", "submit_result",
                            raw[:100], content[:2000]))
                        state.step_count += 1
                        if stream_callback: stream_callback(state.steps[-1])
                        break
                    state.steps.append(AgentStep(
                        f"JSON error: {e}", tool_name,
                        tool_call.function.arguments[:100],
                        json.dumps({"error": f"Malformed JSON: {str(e)}"})
                    ))
                    state.step_count += 1
                    state.consecutive_errors += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    continue

                thought = msg.content or f"Calling {tool_name}"

                if tool_name == "ask_user":
                    question = tool_args.get("question", "What should I do next?")
                    state.user_question = question
                    state.waiting_for_user = True
                    state.steps.append(AgentStep(thought, "ask_user", json.dumps(tool_args), "Waiting for user response..."))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break

                if tool_name == "submit_result":
                    state.steps.append(AgentStep(thought, "submit_result", json.dumps(tool_args), "Task complete."))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break

                observation = self._execute_tool_safe(state, tool_name, tool_args)
                state.steps.append(AgentStep(thought, tool_name, json.dumps(tool_args), observation))
                state.step_count += 1
                try:
                    obs_data = json.loads(observation)
                    if "error" in obs_data:
                        state.consecutive_errors += 1
                    else:
                        state.consecutive_errors = 0
                except Exception:
                    state.consecutive_errors = 0
                self._check_progress(state)
                if stream_callback: stream_callback(state.steps[-1])
            else:
                content = msg.content or ""
                if len(content) > 100:
                    state.steps.append(AgentStep("Analysis complete.", None, None, content))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])
                    break
                else:
                    state.steps.append(AgentStep(
                        content[:200], "rejected_text", None,
                        "Call a tool or use submit_result to deliver your analysis."
                    ))
                    state.step_count += 1
                    if stream_callback: stream_callback(state.steps[-1])

        if state.step_count >= state.max_steps:
            parts = []
            fragments = []
            for s in state.steps:
                if s.action == "rejected_text" and s.thought:
                    fragments.append(s.thought)
            if fragments:
                parts.append("Agent's partial analysis:")
                for i, f in enumerate(fragments):
                    parts.append(f"  [{i+1}] {f}")
            tool_steps = [s for s in state.steps if s.action and s.action not in ("rejected_text", "submit_result", "ask_user")]
            if tool_steps:
                parts.append("Last tool calls:")
                for s in tool_steps[-3:]:
                    obs_summary = ""
                    if s.observation:
                        try:
                            obs_data = json.loads(s.observation)
                            obs_summary = obs_data.get("stdout", "") or obs_data.get("error", "")
                        except Exception:
                            obs_summary = str(s.observation)
                    parts.append(f"  [{s.action}] {obs_summary[:200]}")
            parts.append(f"Files read: {sorted(state.files_read) or '(none)'}")
            parts.append(f"Files edited: {sorted(state.files_edited) or '(none)'}")
            summary = "\n".join(parts) if len(parts) > 2 else (
                "Max steps reached with no analyzable output."
            )
            state.steps.append(AgentStep("Max steps reached.", None, None, summary))
            if stream_callback: stream_callback(state.steps[-1])

        return state

    def _get_tools_for_step(self, state):
        remaining = state.max_steps - state.step_count
        all_tools = self.tools.get_tool_definitions()
        if remaining <= 2:
            return [t for t in all_tools if t["function"]["name"] in ("submit_result", "read_file", "load_memory")]
        elif remaining <= 4:
            keep = {"read_file", "grep", "glob", "run_command", "remote_run", "edit_file", "submit_result", "load_memory"}
            return [t for t in all_tools if t["function"]["name"] in keep]
        return all_tools

    def _get_progress_warning(self, state):
        remaining = state.max_steps - state.step_count
        # Read-only task detection: read files but no edits → synthesize earlier
        has_read = len(state.files_read) > 0
        no_edits = len(state.files_edited) == 0
        if has_read and no_edits and state.consecutive_no_progress >= 2:
            return "You have read the file(s). This is a read-only task — no edits needed. Call submit_result NOW with your analysis.\n"
        if state.consecutive_errors >= 3:
            return "CRITICAL: 3+ consecutive errors. Stop using the failing tool. Call submit_result with what you have NOW.\n"
        if state.force_finish:
            return "URGENT: No progress in 3+ steps. Call submit_result NOW.\n"
        if remaining <= 2:
            return f"FINAL ({state.step_count}/{state.max_steps}): Only submit_result available. Wrap up NOW.\n"
        if remaining <= 4:
            return f"WARNING ({state.step_count}/{state.max_steps}): Running out of steps. Finish soon.\n"
        if remaining <= 6:
            return f"Note ({state.step_count}/{state.max_steps}): Organize, prepare to wrap up.\n"
        return ""

    def _check_progress(self, state):
        current = len(state.files_read) + len(state.files_edited)
        if current > state.last_action_count:
            state.consecutive_no_progress = 0
        else:
            state.consecutive_no_progress += 1
        state.last_action_count = current
        if state.consecutive_no_progress >= 3:
            state.force_finish = True

    def _execute_tool(self, state, tool_name, tool_args):
        if tool_name == "read_file":
            path = tool_args.get("path", "")
            # Normalize path: absolute + lowercase (Windows drive letter)
            try:
                normalized = os.path.normpath(os.path.abspath(path)).lower()
            except Exception:
                normalized = path.lower() if path else path
            path_lower = path.lower()
            # Check raw, lowercased, and normalized variants
            if (path in state.files_read or path_lower in state.files_read or
                normalized in state.files_read):
                return json.dumps({
                    "warning": f"Already read '{path}'. You have the content. Move on.",
                    "path": path,
                })
            # Store the normalized (lowercased) version
            if normalized: state.files_read.add(normalized)
        elif tool_name in ("edit_file", "write_file"):
            path = tool_args.get("path", "")
            if path: state.files_edited.add(path)
            # Allow one re-read after edit for verification
            state.files_read.discard(path)

        if tool_name == "run_command":
            cmd = tool_args.get("cmd", "")
            dangerous = any(kw in cmd for kw in ["rm ", "sudo", "chmod", "> /dev", "mkfs", "dd "])
            if dangerous:
                return json.dumps({
                    "warning": f"Dangerous command blocked: '{cmd[:80]}'. Use a safer approach.",
                    "suggestion": "Use edit_file to make file changes instead of shell commands."
                })

        # expand_context: retrieve from context manager cache
        if tool_name == "expand_context":
            if self.ctx_manager:
                cache_id = tool_args.get("cache_id", "")
                content = self.ctx_manager.expand_context(cache_id)
                return json.dumps({"cache_id": cache_id, "content": content[:2000]})
            return json.dumps({"error": "Context manager not available"})

        return self.tools.execute(tool_name, **tool_args)

    def _execute_tool_safe(self, state, tool_name, tool_args):
        """Execute a tool with timeout and error wrapping."""
        try:
            # Execute with thread-based timeout for all tools
            result_holder = [None]
            error_holder = [None]

            def _run():
                try:
                    result_holder[0] = self._execute_tool(state, tool_name, tool_args)
                except Exception as e:
                    error_holder[0] = e

            t = threading.Thread(target=_run)
            t.start()
            t.join(timeout=45)  # 45s total tool execution limit

            if t.is_alive():
                return json.dumps({
                    "error": f"Tool '{tool_name}' timed out after 45s. Try a smaller scope.",
                    "suggestion": "Read fewer files, search with narrower patterns, or use shorter commands."
                })

            if error_holder[0]:
                return json.dumps({
                    "error": f"Tool execution failed: {str(error_holder[0])[:200]}",
                    "tool": tool_name,
                })

            return result_holder[0]

        except Exception as e:
            return json.dumps({
                "error": f"Tool execution failed: {str(e)[:200]}",
                "tool": tool_name,
            })

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        """Determine if an API error is transient (worth retrying)."""
        err_str = str(error).lower()
        retryable = ["timeout", "connection", "rate limit", "server error", "503", "502", "504", "429"]
        return any(kw in err_str for kw in retryable)

    def _format_history(self, state):
        if not state.steps:
            return "(Starting new task.)"
        lines = []
        for i, step in enumerate(state.steps[-8:]):
            lines.append(f"Step {i+1}: {step.thought[:100]}")
            if step.action:
                lines.append(f"  Action: {step.action}")
                obs = step.observation or ""
                # read_file: keep full content for files < 500 lines
                if step.action == "read_file":
                    try:
                        data = json.loads(obs)
                        total_ln = data.get("total_lines", 0)
                        if total_ln > 500:
                            content_lines = data["content"].split("\n")
                            data["content"] = "\n".join(content_lines[:80])
                            data["content"] += f"\n... ({total_ln - 80} more lines)"
                            obs = json.dumps(data, ensure_ascii=False)
                    except:
                        pass
                elif len(obs) > 800:
                    obs = obs[:800] + "...[truncated]"
                lines.append(f"  Result: {obs}")
            lines.append("")
        lines.append(f"Files read: {sorted(state.files_read)}")
        lines.append(f"Files edited: {sorted(state.files_edited)}")
        return "\n".join(lines)
