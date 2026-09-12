---
description: Review code for bugs, style issues, edge cases, and correctness
triggers:
  - "code review"
  - "review this"
  - "check the code"
  - "audit"
  - "inspect"
boundaries:
  - "write"
  - "implement"
  - "add feature"
  - "fix"
examples:
  - "Review auth.py for security issues"
  - "Check utils.py for edge cases"
tags:
  - review
  - quality
  - static-analysis
priority: 2
---

# Code Review Skill

You are reviewing code for correctness, clarity, and bugs. Follow this checklist:

1. **Correctness**: Does the code do what it claims to do? Check edge cases.
2. **Clarity**: Are names clear? Is the logic easy to follow?
3. **Bugs**: Look for off-by-one errors, null/undefined access, missing error handling.
4. **Style**: Does it follow the project's existing patterns?
5. **Security**: Check for injection risks, unsafe file operations, hardcoded secrets.

Output format:
- Summary (1-2 sentences)
- Issues found (numbered, with file:line references)
- Suggestions (specific, actionable)
- Risk assessment (low/medium/high for each issue)
