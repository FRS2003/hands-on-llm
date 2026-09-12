"""
Layered Context Compression — 分层上下文压缩

Pipeline: Large output -> Externalize -> Placeholder -> On-demand expand -> Overflow fallback

Layers:
  L1 (Normal):    Full context, no compression needed
  L2 (Summary):   Observations > 2000 chars -> structured summary with placeholders
  L3 (External):  Full outputs saved to disk, replaced with short references
  L4 (Emergency): Aggressive truncation, keep only recent steps + key decisions

Design: cache-friendly placeholder compression + structured note summaries.
"""
import json
import os
import time
from dataclasses import dataclass, field

CACHE_DIR = ".mini-code/context-cache"


@dataclass
class ContextStats:
    """Track context usage across the agent session."""
    total_tokens_estimate: int = 0
    steps_compressed: int = 0
    externalized_count: int = 0
    emergency_triggers: int = 0


class ContextManager:
    """
    Manages context window budget.

    Token budget: 30000 chars (~7500 tokens for prompt)
    When approaching limit:
      1. Compress old read_file observations (>2000 chars -> summary)
      2. Externalize large tool outputs to disk cache
      3. Replace with [EXPAND:id] placeholders
      4. If still over budget, aggressive truncation
    """

    def __init__(self, cache_dir: str = CACHE_DIR, budget_chars: int = 30000):
        self.cache_dir = cache_dir
        self.budget_chars = budget_chars
        self.stats = ContextStats()
        os.makedirs(cache_dir, exist_ok=True)
        self._cache: dict[str, str] = {}  # in-memory cache for fast lookup

    def compress_history(self, steps: list, state) -> list:
        """
        Compress step history to stay within token budget.
        Returns compressed list of step display dicts.
        """
        # Build uncompressed display first
        display_steps = []
        total_chars = 0

        for step in steps:
            entry = self._step_to_display(step)
            display_steps.append(entry)
            total_chars += len(str(entry))

        # If under budget, no compression needed
        if total_chars < self.budget_chars:
            return display_steps

        # L2: Compress old observations (>2000 chars -> summary)
        for entry in reversed(display_steps):
            if total_chars < self.budget_chars:
                break
            if "result" in entry and len(str(entry["result"])) > 2000:
                old_len = len(str(entry["result"]))
                entry["result"] = self._summarize_observation(step, entry.get("action", ""))
                total_chars -= (old_len - len(str(entry["result"])))
                self.stats.steps_compressed += 1

        # L3: Externalize largest remaining observations
        for entry in reversed(display_steps):
            if total_chars < self.budget_chars:
                break
            result = entry.get("result", "")
            if len(str(result)) > 1000:
                cache_id = self._externalize(result, entry.get("action", "unknown"))
                entry["result"] = f"[EXPAND:{cache_id}] Call expand_context('{cache_id}') to retrieve full output."
                entry["result_preview"] = str(result)[:100]
                total_chars -= (len(str(result)) - len(entry["result"]))
                self.stats.externalized_count += 1

        # L4: Emergency — aggressive truncation, keep only last 5 steps + key decisions
        if total_chars > self.budget_chars:
            self.stats.emergency_triggers += 1
            # Keep first step (task context) + last 5 steps
            if len(display_steps) > 6:
                display_steps = display_steps[:1] + display_steps[-5:]
            # Further compress all observations
            for entry in display_steps:
                if "result" in entry and len(str(entry["result"])) > 300:
                    entry["result"] = str(entry["result"])[:300] + "...[emergency truncated]"

        return display_steps

    def _step_to_display(self, step) -> dict:
        """Convert an AgentStep to a compact display dict."""
        entry = {
            "thought": step.thought[:150] if step.thought else "",
            "action": step.action or "",
        }
        if step.observation:
            obs = step.observation
            # read_file: keep structure, truncate content
            if step.action == "read_file":
                try:
                    data = json.loads(obs)
                    total_ln = data.get("total_lines", 0)
                    content = data.get("content", "")
                    if total_ln <= 100:
                        entry["result"] = content
                    else:
                        lines = content.split("\n")
                        summary = "\n".join(lines[:40])
                        summary += f"\n... ({total_ln - 40} more lines)"
                        entry["result"] = summary
                        # Cache full content
                        cache_id = self._externalize(content, step.action)
                        entry["result"] += f"\n[Full file cached: expand_context('{cache_id}')]"
                except:
                    entry["result"] = obs[:1500]
            # Other tools: truncate if too long
            elif len(obs) > 2000:
                entry["result"] = obs[:1500] + "...[truncated]"
                entry["result_preview_only"] = True
            else:
                entry["result"] = obs
        return entry

    def _summarize_observation(self, step, tool_name: str) -> str:
        """Create a structured summary of a large observation."""
        obs = step.observation if hasattr(step, 'observation') else str(step)
        if tool_name == "read_file":
            try:
                data = json.loads(obs)
                total = data.get("total_lines", 0)
                path = data.get("path", "")
                # Extract key structural elements
                content = data.get("content", "")
                imports = [l for l in content.split("\n") if l.strip().startswith(("import ", "from "))]
                classes = [l for l in content.split("\n") if l.strip().startswith("class ")]
                funcs = [l for l in content.split("\n") if l.strip().startswith("def ")]
                return (
                    f"[Summary] File: {path} ({total} lines)\n"
                    f"  Imports: {len(imports)}\n"
                    f"  Classes: {len(classes)}\n"
                    f"  Functions: {len(funcs)}"
                )
            except:
                pass
        return f"[Summary] {tool_name} output ({len(str(obs))} chars)"

    def _externalize(self, content: str, context: str = "") -> str:
        """Save content to disk cache, return cache ID."""
        cache_id = f"{context}_{int(time.time()*1000)}_{hash(content) % 10000}"
        self._cache[cache_id] = content[:10000]  # Max 10K chars in memory
        # Also persist to disk
        try:
            fpath = os.path.join(self.cache_dir, f"{cache_id}.json")
            with open(fpath, "w") as f:
                json.dump({"content": content[:10000], "context": context}, f)
        except Exception:
            pass
        self.stats.externalized_count += 1
        return cache_id

    def expand_context(self, cache_id: str) -> str:
        """Retrieve externalized content by cache ID."""
        if cache_id in self._cache:
            return self._cache[cache_id]
        try:
            fpath = os.path.join(self.cache_dir, f"{cache_id}.json")
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    data = json.load(f)
                    return data.get("content", "Cache empty.")
        except Exception:
            pass
        return f"Cache '{cache_id}' not found or expired."

    def token_estimate(self, text: str) -> int:
        """Rough token count (4 chars ~= 1 token)."""
        return len(text) // 4
