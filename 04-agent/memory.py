"""
Self-Evolving Memory System — 自进化记忆沉淀

Pipeline: Execute -> Reflect -> Extract -> Classify -> Store -> Index -> Reuse

Three memory types:
  - procedural: HOW to do something (tool sequences, patterns, workarounds)
  - episodic: WHAT happened (task, result, errors encountered, files touched)
  - preference: USER preferences (coding style, naming conventions, approval patterns)
"""
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

MEMORY_DIR = ".mini-code/memories"


@dataclass
class Memory:
    """A single memory unit."""
    id: str
    type: str          # "procedural", "episodic", "preference"
    intent: str        # task intent keywords (for retrieval matching)
    content: str       # the distilled lesson
    tags: list[str]    # searchable tags
    source_task: str   # what task produced this
    success: bool      # was the task successful?
    tools_used: list[str]
    files_touched: list[str]
    created_at: float
    reused_count: int = 0
    last_reused_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id, "type": self.type, "intent": self.intent,
            "content": self.content, "tags": self.tags, "source_task": self.source_task,
            "success": self.success, "tools_used": self.tools_used,
            "files_touched": self.files_touched, "created_at": self.created_at,
            "reused_count": self.reused_count, "last_reused_at": self.last_reused_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Memory":
        return cls(**d)


class MemorySystem:
    """
    Manages the full memory lifecycle: store, search, retrieve, evolve.

    Storage: JSON files in .mini-code/memories/
    Index: in-memory keyword index for fast lookup
    Retrieval: intent matching + tag overlap + recency boost
    """

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        os.makedirs(os.path.join(memory_dir, "procedural"), exist_ok=True)
        os.makedirs(os.path.join(memory_dir, "episodic"), exist_ok=True)
        os.makedirs(os.path.join(memory_dir, "preference"), exist_ok=True)
        self._index: list[Memory] = []
        self._load_index()

    def _load_index(self):
        """Load all memories into in-memory index."""
        self._index = []
        for mem_type in ["procedural", "episodic", "preference"]:
            type_dir = os.path.join(self.memory_dir, mem_type)
            if not os.path.exists(type_dir):
                continue
            for fname in os.listdir(type_dir):
                if fname.endswith(".json"):
                    try:
                        with open(os.path.join(type_dir, fname), "r", encoding="utf-8") as f:
                            self._index.append(Memory.from_dict(json.load(f)))
                    except Exception:
                        pass
        self._index.sort(key=lambda m: m.created_at, reverse=True)

    # -- Step 1: Execute (called by agent after task completion) --

    def learn_from_task(self, task: str, state: Any, success: bool) -> list[Memory]:
        """
        Extract lessons from a completed task. Called automatically after agent.run().
        Returns newly created memories.
        """
        new_memories = []

        # Classify task intent from the task description
        intent = self._classify_intent(task)

        # Extract tool usage pattern (procedural)
        tools_used = []
        for step in state.steps:
            if step.action and step.action not in ("submit_result", "think"):
                tools_used.append(step.action)

        success_tools = [t for t in tools_used if t != "load_skill"]
        if len(success_tools) >= 2 and success:
            pattern = " -> ".join(success_tools[:6])
            mem = self._create_memory(
                mem_type="procedural",
                intent=intent,
                content=f"For task '{intent}', the successful tool pattern was: {pattern}.",
                tags=["tool-pattern", intent] + list(set(success_tools[:4])),
                source_task=task,
                success=True,
                tools_used=success_tools,
                files_touched=list(state.files_read | state.files_edited),
            )
            new_memories.append(mem)

        # Extract error pattern (procedural)
        errors = []
        for step in state.steps:
            if step.observation:
                try:
                    data = json.loads(step.observation)
                    if "error" in data or "warning" in data:
                        errors.append(f"{step.action}: {str(data.get('error', data.get('warning', '')))[:100]}")
                except:
                    pass

        if errors:
            mem = self._create_memory(
                mem_type="procedural",
                intent=intent,
                content=f"Common errors for '{intent}': {'; '.join(errors[:3])}. Avoid by: checking file content before editing, using exact string matching.",
                tags=["error-pattern", intent] + list(set(success_tools[:3])),
                source_task=task,
                success=False,
                tools_used=tools_used,
                files_touched=list(state.files_read | state.files_edited),
            )
            new_memories.append(mem)

        # Extract episode (episodic)
        files_touched = list(state.files_read | state.files_edited)[:5]
        mem = self._create_memory(
            mem_type="episodic",
            intent=intent,
            content=f"Task '{task[:80]}...' completed (success={success}). Touched: {files_touched}. Steps: {len(state.steps)}.",
            tags=["episode", intent],
            source_task=task,
            success=success,
            tools_used=tools_used,
            files_touched=files_touched,
        )
        new_memories.append(mem)

        return new_memories

    # -- Step 2: Reflect (CRITICAL: transforms raw experience into reusable knowledge) --

    def reflect(self, llm_client, model: str = "deepseek-chat") -> list[Memory]:
        """
        Periodically consolidate recent episodic memories into refined procedural ones.
        This is the "reflect" step — turning raw episodes into reusable patterns.
        Only called when there are >= 5 new unreflected episodes.
        """
        recent = [m for m in self._index if m.type == "episodic" and m.created_at > time.time() - 86400]
        if len(recent) < 5:
            return []

        # Group by intent
        by_intent = {}
        for m in recent:
            if m.intent not in by_intent:
                by_intent[m.intent] = []
            by_intent[m.intent].append(m)

        new_memories = []
        for intent, episodes in by_intent.items():
            if len(episodes) < 3:
                continue

            success_rate = sum(1 for e in episodes if e.success) / len(episodes)
            all_tools = list(set(t for e in episodes for t in e.tools_used))
            all_files = list(set(f for e in episodes for f in e.files_touched))

            # Use LLM to distill patterns
            episodes_text = "\n".join(
                f"- Success={e.success}: {e.content[:200]}" for e in episodes[:5]
            )
            prompt = (
                f"You are analyzing coding patterns to extract reusable knowledge.\n\n"
                f"Task type: {intent}\n"
                f"Success rate: {success_rate:.0%}\n"
                f"Recent episodes:\n{episodes_text}\n\n"
                f"Based on these episodes, write ONE concise procedural memory (1-2 sentences) "
                f"that captures: the best approach, common pitfalls to avoid, and recommended tool sequence.\n"
                f"Format: plain text, max 200 chars."
            )

            try:
                resp = llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3, max_tokens=150,
                )
                refined = resp.choices[0].message.content.strip()

                mem = self._create_memory(
                    mem_type="procedural",
                    intent=intent,
                    content=refined,
                    tags=["refined", intent, f"success-rate-{int(success_rate*100)}"],
                    source_task=f"Reflection on {len(episodes)} episodes of '{intent}'",
                    success=success_rate > 0.5,
                    tools_used=all_tools,
                    files_touched=all_files,
                )
                new_memories.append(mem)
            except Exception:
                pass

        return new_memories

    # -- Step 3: Classify (intent extraction) --

    def _classify_intent(self, task: str) -> str:
        """Extract a short intent label from a task description."""
        task_lower = task.lower()
        if any(kw in task_lower for kw in ["bug", "fix", "error", "crash", "broken"]):
            return "fix-bug"
        if any(kw in task_lower for kw in ["add", "new feature", "implement", "create"]):
            return "add-feature"
        if any(kw in task_lower for kw in ["review", "check", "inspect", "audit"]):
            return "code-review"
        if any(kw in task_lower for kw in ["refactor", "restructure", "clean", "simplify"]):
            return "refactor"
        if any(kw in task_lower for kw in ["read", "explain", "what", "how", "tell"]):
            return "understand"
        if any(kw in task_lower for kw in ["test", "verify", "validate"]):
            return "test"
        return "general"

    # -- Step 4: Store + Index --

    def _create_memory(self, mem_type, intent, content, tags, source_task,
                       success, tools_used, files_touched) -> Memory:
        """Create, store, and index a new memory."""
        mem_id = f"{mem_type[:4]}_{int(time.time()*1000)}"
        mem = Memory(
            id=mem_id, type=mem_type, intent=intent, content=content,
            tags=list(set(tags)), source_task=source_task[:200],
            success=success, tools_used=tools_used, files_touched=files_touched,
            created_at=time.time(),
        )

        # Persist
        type_dir = os.path.join(self.memory_dir, mem_type)
        fpath = os.path.join(type_dir, f"{mem_id}.json")
        with open(fpath, "w", encoding="utf-8") as f:
            data = mem.to_dict()
            # Sanitize surrogates from all string values
            for k, v in data.items():
                if isinstance(v, str):
                    data[k] = v.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
                elif isinstance(v, list):
                    data[k] = [x.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace") if isinstance(x, str) else x for x in v]
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Index + auto-cleanup: keep max 200 memories, remove lowest-score old ones
        self._index.insert(0, mem)
        if len(self._index) > 200:
            # Remove oldest episodic memories first (procedural are more valuable)
            episodic = [m for m in self._index if m.type == "episodic"]
            if episodic:
                oldest = min(episodic, key=lambda m: m.created_at)
                self._index.remove(oldest)
                try:
                    os.remove(os.path.join(self.memory_dir, "episodic", f"{oldest.id}.json"))
                except Exception:
                    pass
        return mem

    # -- Step 5: Retrieve (for injection into new tasks) --

    def retrieve(self, task: str, max_results: int = 3) -> list[Memory]:
        """Find relevant memories for a new task. Uses intent classification + keyword overlap."""
        intent = self._classify_intent(task)
        task_terms = set(re.findall(r"[a-zA-Z]{4,}", task.lower()))
        return self._search(intent, task_terms, max_results)

    def search(self, query: str, max_results: int = 3) -> list[Memory]:
        """
        On-demand search invoked by the Agent via load_memory tool.
        More flexible than retrieve() — Agent can ask specific questions like
        "sql injection fix" or "auth module refactor pattern".
        """
        query_lower = query.lower()
        query_terms = set(re.findall(r"[a-zA-Z]{3,}", query_lower))

        # Also try to match against intent labels
        intent = self._classify_intent(query)

        # Score: keyword overlap in content + tags + intent bonus + recency
        scored = []
        for mem in self._index:
            score = 0.0
            mem_content_lower = mem.content.lower()
            # Direct keyword match in content
            kw_hits = sum(1 for t in query_terms if t in mem_content_lower)
            score += kw_hits * 3.0
            # Tag match
            tag_hits = sum(1 for t in query_terms for tag in mem.tags if t in tag.lower())
            score += tag_hits * 2.0
            # Intent match (bonus, not primary)
            if mem.intent == intent:
                score += 2.0
            # Content substring match for multi-word queries
            if len(query) > 10 and query_lower in mem_content_lower:
                score += 5.0
            # Recency boost
            if mem.created_at > time.time() - 3600:
                score += 0.5
            # Reuse bonus
            score += min(mem.reused_count * 0.2, 1.0)

            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, mem in scored[:max_results]:
            mem.reused_count += 1
            mem.last_reused_at = time.time()
            results.append(mem)

        return results

    def _search(self, intent: str, terms: set, max_results: int = 3) -> list[Memory]:
        """Shared scoring engine between retrieve() and search()."""
        scored = []
        for mem in self._index:
            score = 0.0
            # Intent match
            if mem.intent == intent:
                score += 3.0
            # Tag overlap with task terms
            tag_overlap = len(set(mem.tags) & terms)
            score += tag_overlap * 1.0
            # Keyword overlap in content
            content_overlap = len(set(re.findall(r"[a-zA-Z]{4,}", mem.content.lower())) & terms)
            score += content_overlap * 0.5
            # Recency boost
            if mem.created_at > time.time() - 3600:
                score += 0.5
            # Reuse bonus
            score += min(mem.reused_count * 0.2, 1.0)

            if score > 0:
                scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, mem in scored[:max_results]:
            mem.reused_count += 1
            mem.last_reused_at = time.time()
            results.append(mem)

        return results

    # -- Step 6: Format for prompt injection --

    def format_for_prompt(self, task: str) -> str:
        """Get formatted memory context for system prompt injection."""
        memories = self.retrieve(task, max_results=3)
        if not memories:
            return ""

        lines = ["## Relevant Past Experiences (from memory)"]
        for i, mem in enumerate(memories, 1):
            tag = {"procedural": "[HOW-TO]", "episodic": "[PAST-TASK]", "preference": "[PREF]"}
            label = tag.get(mem.type, "[MEM]")
            lines.append(f"{i}. {label} {mem.content}")
        return "\n".join(lines)
