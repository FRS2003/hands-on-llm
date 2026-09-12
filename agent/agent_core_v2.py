"""
Autonomous Research Agent v2 — minimal prompt, self-planning, self-reflecting.

Key difference from v1: The agent decides its OWN strategy, not following a checklist.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None


@dataclass
class AgentState:
    question: str
    plan: list[str] = field(default_factory=list)   # Agent's own plan
    steps: list[AgentStep] = field(default_factory=list)
    gathered_papers: dict[int, dict] = field(default_factory=dict)
    searched_queries: set = field(default_factory=set)
    searched_papers: set = field(default_factory=set)
    step_count: int = 0
    max_steps: int = 12  # Safety net only, agent shouldn't need this many


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, description: str, parameters: dict):
        self._tools[name] = {"fn": fn, "description": description, "parameters": parameters}

    def get_tool_definitions(self) -> list[dict]:
        return [{"type": "function", "function": {"name": n, "description": i["description"], "parameters": i["parameters"]}} for n, i in self._tools.items()]

    def execute(self, name: str, **kwargs) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Tool '{name}' not found"})
        try:
            return json.dumps(self._tools[name]["fn"](**kwargs), ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})


# ── Minimal system prompt — agent decides its own process ──

PLAN_PROMPT = """You are an autonomous research agent with access to a knowledge base of 200+ medical AI papers.

Your task: "{question}"

Before taking any action, write a concise PLAN. Break complex questions into sub-questions.
A simple fact-check needs just 1-2 searches. A comparison needs separate searches for each side.
A synthesis needs to gather evidence from multiple angles.

Output your plan as a numbered list, then on the LAST line write exactly: END_PLAN"""


EXECUTE_PROMPT = """You are an autonomous research agent. You have access to these tools:
- search_papers(query): Search for papers. Returns ranked results with abstracts.
- read_section(paper_id, section): Read abstract/methods/results/discussion of a paper.
- extract_claims(paper_id): Extract key factual claims from a paper.
- compare_papers(paper_ids, dimension): Compare papers on methodology/results/limitations/data.
- verify_claim(claim, paper_id): Check if a claim is supported by a paper.

Your task: {question}

Your plan:
{plan}

What you've done so far:
{history}

Decide your next action. You may:
- Call a tool to gather more information
- Output FINAL_ANSWER: if you have enough evidence to answer convincingly

Before outputting FINAL_ANSWER, ask yourself:
- Did I find evidence for ALL parts of the question?
- Are my claims backed by specific papers I read?
- If comparing, did I examine BOTH sides?

Be concise in your thoughts. If a search found nothing useful, try a fundamentally different query.
If you've searched 3+ times for the same thing and failed, admit it and work with what you have."""


REFLECT_PROMPT = """You are about to deliver your final answer. Before you do, reflect:

Your answer:
{draft}

Check:
1. Is every factual claim backed by a paper you actually read (not just saw in search results)?
2. Are there any contradictions between papers? If so, flag them.
3. Did you address ALL parts of the original question: "{question}"?
4. Is there anything important you missed?

If you find issues, state them and revise. Otherwise, output: ANSWER_OK"""


class AutonomousAgent:
    """Agent that plans, executes, and reflects autonomously."""

    def __init__(self, tools: ToolRegistry, llm_client: OpenAI, model: str = "deepseek-chat"):
        self.tools = tools
        self.client = llm_client
        self.model = model

    def _call_llm(self, system_prompt: str, tools_enabled: bool = True) -> tuple[str, dict | None]:
        """Call LLM and return (content, tool_call_or_None)."""
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}],
            "temperature": 0.3,
            "max_tokens": 600,
        }
        if tools_enabled:
            kwargs["tools"] = self.tools.get_tool_definitions()
            kwargs["tool_choice"] = "auto"

        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message

        tool_call = None
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]

        return msg.content or "", tool_call

    def run(self, question: str, stream_callback: Callable | None = None) -> AgentState:
        state = AgentState(question=question)

        # ── Phase 1: Plan ──
        plan_content, _ = self._call_llm(
            PLAN_PROMPT.format(question=question),
            tools_enabled=False,
        )
        # Extract plan lines
        plan_lines = []
        for line in plan_content.split("\n"):
            line = line.strip()
            if "END_PLAN" in line:
                break
            if line and (line[0].isdigit() or line.startswith("-")):
                plan_lines.append(line.lstrip("0123456789.-) "))
        state.plan = plan_lines or [f"Search for information about: {question}"]
        if stream_callback:
            stream_callback(AgentStep(thought=f"Plan: {'; '.join(state.plan[:3])}", action=None, action_input=None, observation=None))

        # ── Phase 2: Execute ──
        while state.step_count < state.max_steps:
            history = self._format_history(state)
            content, tool_call = self._call_llm(
                EXECUTE_PROMPT.format(question=question, plan="\n".join(f"- {p}" for p in state.plan), history=history),
                tools_enabled=True,
            )

            # Check for FINAL_ANSWER
            if "FINAL_ANSWER:" in content:
                draft = content.split("FINAL_ANSWER:", 1)[1].strip()
                # ── Phase 3: Reflect ──
                reflect_content, _ = self._call_llm(
                    REFLECT_PROMPT.format(draft=draft, question=question),
                    tools_enabled=False,
                )
                if "ANSWER_OK" in reflect_content:
                    step = AgentStep(thought="Reflection passed. Delivering answer.", action=None, action_input=None, observation=draft)
                else:
                    step = AgentStep(thought=f"Reflection found issues: {reflect_content[:200]}", action=None, action_input=None, observation=draft)
                state.steps.append(step)
                if stream_callback:
                    stream_callback(step)
                break

            # Tool call
            if tool_call:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                observation = self._execute_with_guards(state, tool_name, tool_args)
                step = AgentStep(thought=content[:200], action=tool_name, action_input=json.dumps(tool_args), observation=observation)
                state.steps.append(step)
                state.step_count += 1

                if stream_callback:
                    stream_callback(step)
            else:
                # No tool call and no FINAL_ANSWER — agent is thinking out loud, prompt it to act
                state.steps.append(AgentStep(thought=content[:200], action=None, action_input=None, observation="(Thinking — call a tool or output FINAL_ANSWER)"))
                state.step_count += 1

        return state

    def _execute_with_guards(self, state: AgentState, tool_name: str, tool_args: dict) -> str:
        """Execute tool with safety guards (code-level, not prompt-level)."""
        if tool_name == "search_papers":
            query = tool_args.get("query", "")
            if query in state.searched_queries:
                return json.dumps({"warning": "Duplicate query. Try completely different keywords.", "already_searched": sorted(state.searched_queries)})
            state.searched_queries.add(query)
            result = self.tools.execute(tool_name, **tool_args)
            # Track papers found
            try:
                for r in json.loads(result).get("results", []):
                    state.gathered_papers[r["id"]] = r
            except:
                pass
            return result

        elif tool_name in ("read_section", "extract_claims", "verify_claim"):
            paper_id = int(tool_args.get("paper_id", 0))
            if paper_id in state.searched_papers:
                return json.dumps({"warning": f"Already examined paper {paper_id}. Move on."})
            state.searched_papers.add(paper_id)
            return self.tools.execute(tool_name, **tool_args)

        elif tool_name == "compare_papers":
            return self.tools.execute(tool_name, **tool_args)

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _format_history(self, state: AgentState) -> str:
        if not state.steps:
            return "(No actions taken yet.)"
        lines = []
        for i, step in enumerate(state.steps):
            lines.append(f"Step {i+1}:")
            if step.action:
                lines.append(f"  Action: {step.action}")
                obs = (step.observation or "")
                if len(obs) > 800:
                    obs = obs[:800] + "...[truncated]"
                lines.append(f"  Result: {obs}")
            else:
                lines.append(f"  Thought: {step.thought[:200]}")
            lines.append("")
        lines.append(f"Papers examined so far: {sorted(state.searched_papers)}")
        lines.append(f"Queries searched: {sorted(state.searched_queries)}")
        return "\n".join(lines)
