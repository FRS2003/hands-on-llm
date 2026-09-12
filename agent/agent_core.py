"""
Academic Research Agent — ReAct-based agent loop for multi-step literature research.

Architecture:
  User asks a complex research question
  → Agent enters ReAct loop (Thought → Action → Observation → repeat)
  → Agent decides when enough information is gathered
  → Agent generates final answer with verified citations

Key difference from RAG:
  RAG = search once → answer
  Agent = search → evaluate → reformulate → search → compare → verify → answer
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI


# ── Agent State ──

@dataclass
class AgentStep:
    """One step in the agent's reasoning trace."""
    thought: str          # Agent's internal reasoning
    action: str | None    # Tool name (None if final answer)
    action_input: str | None  # Tool input
    observation: str | None   # Tool output


@dataclass
class AgentState:
    """Full state of the agent during execution."""
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    search_history: list[dict] = field(default_factory=list)
    gathered_papers: dict[int, dict] = field(default_factory=dict)
    verified_claims: list[dict] = field(default_factory=list)
    step_count: int = 0
    max_steps: int = 10
    searched_queries: set = field(default_factory=set)    # Avoid re-searching same query
    searched_papers: set = field(default_factory=set)     # Avoid re-reading same paper


# ── Tool Registry ──

class ToolRegistry:
    """Registry of tools the agent can call."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, description: str, parameters: dict):
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "parameters": parameters,
        }

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": info["description"],
                    "parameters": info["parameters"],
                },
            }
            for name, info in self._tools.items()
        ]

    def execute(self, name: str, **kwargs) -> str:
        """Execute a tool and return its output as a string."""
        if name not in self._tools:
            return f"Error: tool '{name}' not found. Available: {list(self._tools.keys())}"
        try:
            result = self._tools[name]["fn"](**kwargs)
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Tool execution error: {e}"


# ── System Prompt ──

SYSTEM_PROMPT = """You are an academic research agent. Your job is to answer research questions
by searching through a knowledge base of 200+ medical AI papers.

You work in a Thought → Action → Observation loop. Be EFFICIENT:

1. PLAN first (for L2/L3 questions): Break the question into sub-questions.
2. SEARCH: Use search_papers with specific English keywords.
3. READ: After searching, read at most 3 papers. Don't read every result.
4. SYNTHESIZE: Once you have evidence from 2-3 papers, form your answer.
5. VERIFY only critical claims with verify_claim.

CRITICAL RULES:
- If search finds papers with score > 0.5, those are likely relevant. Read them and move on.
- Do NOT read more than 3 papers total. After 3 reads, you MUST synthesize.
- Do NOT call extract_claims more than once. It's slow.
- For simple questions, answer in 2-3 steps total.
- L2 comparison questions should take 4-6 steps.
- L3 synthesis questions should take 6-8 steps.
- If you can't find a specific paper after 2 searches, admit it and use what you have.
- Cite papers using [ID] notation.
- End with FINAL_ANSWER: followed by your complete answer.

Current question: {question}

Previous steps:
{history}

What is your next thought? Be concise."""


# ── Agent Loop ──

class ResearchAgent:
    """ReAct-based agent for academic literature research."""

    def __init__(self, tools: ToolRegistry, llm_client: OpenAI, model: str = "deepseek-chat"):
        self.tools = tools
        self.client = llm_client
        self.model = model

    def run(self, question: str, stream_callback: Callable | None = None) -> AgentState:
        """
        Run the agent on a research question.

        Args:
            question: The research question to answer.
            stream_callback: Optional callback(step: AgentStep) for real-time UI updates.

        Returns:
            AgentState with the full reasoning trace and final answer.
        """
        state = AgentState(question=question)

        while state.step_count < state.max_steps:
            # Force-stop after 6 steps: append warning to last observation
            if state.step_count >= 6 and state.steps:
                last_step = state.steps[-1]
                if last_step.observation and "FORCE_STOP" not in (last_step.observation or ""):
                    last_step.observation = "[FORCE_STOP: You have taken enough steps. Output FINAL_ANSWER now.]\n" + (last_step.observation or "")

            # Build message history
            history_text = self._format_history(state)
            prompt = SYSTEM_PROMPT.format(question=question, history=history_text)

            # Get agent's next action
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt}],
                tools=self.tools.get_tool_definitions(),
                tool_choice="auto",
                temperature=0.3,
                max_tokens=500,
            )

            msg = response.choices[0].message

            # Case 1: Agent wants to call a tool
            if msg.tool_calls:
                tool_call = msg.tool_calls[0]
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                # Extract thought from message content
                thought = msg.content or f"Calling {tool_name}"

                # Guard: prevent re-searching same query or re-reading same paper
                if tool_name == "search_papers":
                    query_text = tool_args.get("query", "")
                    # Block exact duplicate searches
                    if query_text in state.searched_queries:
                        observation = json.dumps({
                            "warning": f"Already searched: '{query_text[:50]}...'. Try completely different keywords or give up and answer with what you have.",
                        })
                    # Block after 3 failed searches (no new papers found)
                    elif len(state.searched_queries) >= 3:
                        observation = json.dumps({
                            "warning": "You have already searched 3 times. STOP searching and answer with the evidence you have, even if incomplete.",
                            "papers_found_so_far": sorted(state.gathered_papers.keys()),
                        })
                    else:
                        observation = self.tools.execute(tool_name, **tool_args)
                        state.searched_queries.add(query_text)
                        # Track papers found
                        try:
                            data = json.loads(observation)
                            for r in data.get("results", []):
                                state.gathered_papers[r["id"]] = r
                        except:
                            pass

                elif tool_name in ("read_section", "extract_claims", "verify_claim"):
                    paper_id = int(tool_args.get("paper_id", 0))
                    # Paper-level dedup
                    if paper_id in state.searched_papers:
                        observation = json.dumps({
                            "warning": f"Already examined paper {paper_id}. STOP reading. Synthesize what you have NOW.",
                        })
                    # MAX 3 papers total
                    elif len(state.searched_papers) >= 5:
                        observation = json.dumps({
                            "warning": f"You have read {len(state.searched_papers)} papers. That is ENOUGH. Output FINAL_ANSWER now.",
                            "papers_read": sorted(state.searched_papers, key=str),
                        })
                    else:
                        observation = self.tools.execute(tool_name, **tool_args)
                        state.searched_papers.add(paper_id)

                else:
                    observation = self.tools.execute(tool_name, **tool_args)

                step = AgentStep(
                    thought=thought,
                    action=tool_name,
                    action_input=json.dumps(tool_args),
                    observation=observation,
                )
                state.steps.append(step)
                state.step_count += 1

                # Track searches and gathered papers
                if tool_name == "search_papers":
                    state.search_history.append({
                        "query": tool_args.get("query", ""),
                        "result": observation,
                    })

                if stream_callback:
                    stream_callback(step)

            # Case 2: Agent is giving final answer (no tool call)
            else:
                content = msg.content or ""

                # Check if FINAL_ANSWER marker is present
                if "FINAL_ANSWER:" in content:
                    final_answer = content.split("FINAL_ANSWER:", 1)[1].strip()
                else:
                    final_answer = content

                step = AgentStep(
                    thought="Generating final answer based on gathered evidence.",
                    action=None,
                    action_input=None,
                    observation=final_answer,
                )
                state.steps.append(step)

                if stream_callback:
                    stream_callback(step)

                break

        return state

    def _format_history(self, state: AgentState) -> str:
        """Format previous steps for the prompt."""
        if not state.steps:
            return "(No previous steps — this is the first action.)"

        lines = []
        for i, step in enumerate(state.steps):
            lines.append(f"Step {i + 1}:")
            lines.append(f"  Thought: {step.thought}")
            if step.action:
                lines.append(f"  Action: {step.action}({step.action_input})")
                # Truncate only very long observations (preserve evidence)
                obs = step.observation or ""
                if len(obs) > 2000:
                    obs = obs[:2000] + "...[truncated]"
                lines.append(f"  Observation: {obs}")
            lines.append("")
        return "\n".join(lines)
