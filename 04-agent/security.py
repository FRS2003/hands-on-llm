"""
Multi-Layer Security Review — 权限与安全审查

Layers (from reference project):
  L1: Rule-based filtering — block dangerous patterns
  L2: Tool self-check — each tool validates its own safety
  L3: AI risk classification — LLM-based prompt injection detection
  L4: Human confirmation — critical operations require user approval

Architecture: filter chain, each layer can reject or escalate.
"""
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class SecurityResult:
    """Result of a security check."""
    allowed: bool
    risk_level: str     # "safe", "warning", "dangerous", "blocked"
    reason: str
    requires_approval: bool = False


class SecurityFilter:
    """Chain of security filters. Each can block, warn, or pass."""

    def __init__(self):
        self._stats = {"blocked": 0, "warned": 0, "passed": 0}

    # -- L1: Rule-based pattern filtering --

    def check_tool_call(self, tool_name: str, tool_args: dict) -> SecurityResult:
        """Check a tool call before execution."""

        # L1: Dangerous command patterns
        if tool_name == "run_command":
            cmd = tool_args.get("cmd", "")
            return self._check_command(cmd)

        # L1: File operations outside workspace
        if tool_name in ("read_file", "edit_file", "write_file"):
            path = tool_args.get("path", "")
            return self._check_path(path, tool_name)

        # L1: Prompt injection detection in edit strings
        if tool_name == "edit_file":
            old = tool_args.get("old_string", "")
            new = tool_args.get("new_string", "")
            injection = self._check_prompt_injection(old + " " + new)
            if injection:
                return SecurityResult(False, "dangerous",
                    "Potential prompt injection detected in edit content.", True)

        self._stats["passed"] += 1
        return SecurityResult(True, "safe", "OK")

    def _check_command(self, cmd: str) -> SecurityResult:
        """L1: Block dangerous shell commands."""
        blocked_patterns = [
            (r"rm\s+(-rf?\s+)?[/~]", "File deletion"),
            (r"sudo\s", "Privilege escalation"),
            (r"chmod\s+777", "World-writable permissions"),
            (r">\s*/dev/[sh]d[a-z]", "Raw device write"),
            (r"mkfs\.", "Filesystem formatting"),
            (r"dd\s+if=", "Raw disk operations"),
            (r"git\s+push\s+.*--force", "Force push"),
            (r"git\s+reset\s+--hard", "Hard reset"),
            (r"curl.*\|\s*(ba)?sh", "Pipe to shell"),
            (r"wget.*\|\s*(ba)?sh", "Pipe to shell"),
        ]
        for pattern, desc in blocked_patterns:
            if re.search(pattern, cmd):
                self._stats["blocked"] += 1
                return SecurityResult(False, "blocked",
                    f"Blocked: {desc} (pattern: {pattern[:30]}...)", True)

        # Warning patterns
        warning_patterns = [
            (r"pip\s+install", "Package installation"),
            (r"npm\s+install", "Package installation"),
            (r"docker\s", "Docker operations"),
            (r"kubectl\s", "Kubernetes operations"),
        ]
        for pattern, desc in warning_patterns:
            if re.search(pattern, cmd):
                self._stats["warned"] += 1
                return SecurityResult(True, "warning",
                    f"Warning: {desc} — proceed with caution.", True)

        self._stats["passed"] += 1
        return SecurityResult(True, "safe", "OK")

    def _check_path(self, path: str, tool_name: str) -> SecurityResult:
        """L1: Check file path safety."""
        # Path traversal attack
        if "../" in path or "..\\" in path:
            self._stats["blocked"] += 1
            return SecurityResult(False, "blocked",
                "Path traversal blocked. Use relative paths within workspace.", True)

        # Sensitive files
        sensitive_patterns = [
            r"\.env$", r"\.secret", r"\.pem$", r"\.key$",
            r"/etc/passwd", r"/etc/shadow", r"\.ssh/", r"\.aws/",
        ]
        for pattern in sensitive_patterns:
            if re.search(pattern, path):
                self._stats["warned"] += 1
                return SecurityResult(True, "warning",
                    f"Accessing sensitive file: {path} — proceed with caution.", True)

        # Editing critical config files
        if tool_name == "edit_file":
            critical = [r"setup\.py", r"requirements\.txt", r"package\.json",
                        r"Dockerfile", r"\.github/", r"\.gitlab-ci"]
            for pattern in critical:
                if re.search(pattern, path):
                    self._stats["warned"] += 1
                    return SecurityResult(True, "warning",
                        f"Editing critical file: {path} — verify changes carefully.", True)

        self._stats["passed"] += 1
        return SecurityResult(True, "safe", "OK")

    # -- L2: AI-based risk classification for prompt injection --

    def _check_prompt_injection(self, text: str) -> bool:
        """L2: Detect prompt injection patterns in text."""
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?",
            r"you\s+are\s+now\s+.*?(assistant|agent|bot)",
            r"system\s*:\s*you\s+(must|should|will)",
            r"<\|im_start\|>",
            r"<\|system\|>",
            r"\[INST\].*\[/INST\]",
        ]
        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    # -- L3: Human approval gate --

    def request_approval(self, tool_name: str, tool_args: dict, reason: str) -> bool:
        """
        Request user approval for a dangerous/critical operation.
        In CLI mode, this would prompt the user. In API mode, returns False.
        """
        print(f"\n  [SECURITY] {reason}")
        print(f"  Tool: {tool_name}")
        args_preview = json.dumps(tool_args, ensure_ascii=False)[:200]
        print(f"  Args: {args_preview}")
        # In non-interactive mode, auto-deny
        return False

    def get_stats(self) -> dict:
        return dict(self._stats)
