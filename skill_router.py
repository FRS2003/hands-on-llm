"""
Skill Router — Two-stage recall + rerank for automatic skill selection.

Problem: As skills grow, injecting all skill descriptions wastes tokens and
creates noise (agent picks wrong skill or none at all).

Solution:
  Stage 1 (Coarse recall): Keyword + tag matching of task against skill metadata.
  Stage 2 (Rerank): LLM selects the single best skill from top-K candidates.

This saves tokens (only 1 skill's full prompt injected) and improves accuracy
(less noise in the system prompt).
"""
import os
import json
import re
import yaml  # for SKILL.md frontmatter parsing
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillMeta:
    """Metadata extracted from SKILL.md frontmatter."""
    name: str
    description: str           # What this skill does (1 line)
    triggers: list[str]        # Keywords that trigger this skill
    boundaries: list[str]      # When NOT to use this skill
    examples: list[str]        # Example tasks that match this skill
    tags: list[str]            # Searchable tags
    skill_path: str            # Path to SKILL.md file
    priority: int = 0          # Higher = preferred when multiple skills match

    def to_index_entry(self) -> str:
        """Compact representation for injection into system prompt."""
        triggers_str = ", ".join(self.triggers[:5])
        return f"- **{self.name}**: {self.description} (triggers: {triggers_str})"


class SkillRouter:
    """
    Two-stage skill selection:
    1. Coarse recall: match task against skill metadata (keywords, triggers, tags)
    2. LLM rerank: pick single best skill from top-3 candidates
    """

    def __init__(self, skills_dir: str = "./skills"):
        self.skills_dir = skills_dir
        self._index: list[SkillMeta] = []
        self._load_skills()

    def _load_skills(self):
        """Scan skills directory and parse SKILL.md frontmatter."""
        if not os.path.exists(self.skills_dir):
            return

        for skill_name in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_name, "SKILL.md")
            if not os.path.exists(skill_path):
                continue

            try:
                meta = self._parse_skill(skill_name, skill_path)
                self._index.append(meta)
            except Exception as e:
                print(f"  [SkillRouter] Failed to parse {skill_name}: {e}")

        # Sort by priority (higher first)
        self._index.sort(key=lambda m: -m.priority)

    def _parse_skill(self, name: str, path: str) -> SkillMeta:
        """Parse SKILL.md — extract frontmatter YAML + first heading as description."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Default values
        description = ""
        triggers = []
        boundaries = []
        examples = []
        tags = []
        priority = 0

        # Parse YAML frontmatter if present (between --- markers)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    fm = yaml.safe_load(parts[1])
                    if fm:
                        description = fm.get("description", "")
                        triggers = fm.get("triggers", [])
                        boundaries = fm.get("boundaries", [])
                        examples = fm.get("examples", [])
                        tags = fm.get("tags", [])
                        priority = fm.get("priority", 0)
                except Exception:
                    pass  # Fall through to content-based extraction

        # Fallback: extract from markdown content
        if not description:
            lines = content.split("\n")
            for line in lines:
                if line.startswith("# ") and "Skill" not in line:
                    description = line[2:].strip()
                    break
            if not description:
                description = lines[0].lstrip("# ").strip() if lines else name

        if not triggers:
            triggers = [name.lower().replace("-", " ")]

        return SkillMeta(
            name=name, description=description,
            triggers=triggers, boundaries=boundaries,
            examples=examples, tags=tags,
            skill_path=path, priority=priority,
        )

    def build_index_text(self) -> str:
        """Build compact skill index for system prompt injection (Stage 1 context)."""
        if not self._index:
            return ""
        lines = ["## Available Skills (auto-selected by router)"]
        for skill in self._index:
            lines.append(skill.to_index_entry())
        return "\n".join(lines)

    # -- Stage 1: Coarse Recall --

    def coarse_recall(self, task: str, top_k: int = 3) -> list[SkillMeta]:
        """
        Keyword-based matching of task against skill metadata.
        Returns top-K candidate skills.
        """
        task_lower = task.lower()
        task_terms = set(re.findall(r"[a-zA-Z]{4,}", task_lower))

        scored = []
        for skill in self._index:
            score = 0.0

            # Trigger match (highest weight)
            for trigger in skill.triggers:
                if trigger.lower() in task_lower:
                    score += 5.0
                    break

            # Tag overlap
            tag_hits = sum(1 for tag in skill.tags if tag.lower() in task_lower)
            score += tag_hits * 3.0

            # Description keyword overlap
            desc_terms = set(re.findall(r"[a-zA-Z]{4,}", skill.description.lower()))
            desc_overlap = len(desc_terms & task_terms)
            score += desc_overlap * 0.5

            # Boundary check (penalize)
            for boundary in skill.boundaries:
                if boundary.lower() in task_lower:
                    score -= 10.0
                    break

            # Example match
            for example in skill.examples:
                if any(term.lower() in task_lower for term in example.split()[:3]):
                    score += 2.0
                    break

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: -x[0])
        return [s for _, s in scored[:top_k]]

    # -- Stage 2: LLM Rerank --

    def rerank(self, task: str, candidates: list[SkillMeta], llm_client, model: str = "deepseek-chat") -> SkillMeta | None:
        """
        Use LLM to pick the single best skill from candidates.
        Only called when multiple skills are plausible (avoids LLM cost for obvious cases).
        """
        if len(candidates) == 0:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # Build candidate descriptions
        candidate_text = "\n".join(
            f"{i+1}. {s.name}: {s.description}\n   Triggers: {', '.join(s.triggers[:3])}\n   Boundaries: {', '.join(s.boundaries[:3])}"
            for i, s in enumerate(candidates)
        )

        try:
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    f"Task: {task}\n\n"
                    f"Available skills:\n{candidate_text}\n\n"
                    f"Select the SINGLE best skill for this task. If the task is not a good "
                    f"match for any skill, say NONE.\n"
                    f"Answer with just the skill name or 'NONE'."
                )}],
                temperature=0.0, max_tokens=20,
            )
            choice = resp.choices[0].message.content.strip().lower()
            for s in candidates:
                if s.name.lower() in choice:
                    return s
        except Exception:
            pass

        # Fallback: return highest-scored candidate
        return candidates[0]

    # -- Full Pipeline --

    def route(self, task: str, llm_client, model: str = "deepseek-chat") -> tuple[SkillMeta | None, str]:
        """
        Complete routing pipeline.
        Returns (selected_skill, skill_content_or_empty_string).
        """
        # Stage 1: Coarse recall
        candidates = self.coarse_recall(task, top_k=3)

        if not candidates:
            return None, ""

        # Stage 2: Rerank (only when ambiguous)
        selected = self.rerank(task, candidates, llm_client, model)

        if selected:
            # Load skill content
            try:
                with open(selected.skill_path, "r", encoding="utf-8") as f:
                    content = f.read()
                # Strip frontmatter for cleaner injection
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    content = parts[2].strip() if len(parts) >= 3 else content
                return selected, content[:2000]
            except Exception:
                pass

        return None, ""
