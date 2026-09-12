---
description: Add new functionality following existing project patterns and conventions
triggers:
  - "add"
  - "new feature"
  - "implement"
  - "create"
  - "build"
boundaries:
  - "fix"
  - "review"
  - "refactor"
examples:
  - "Add a median function to utils.py"
  - "Create a new API endpoint for user logout"
tags:
  - feature
  - implementation
  - development
priority: 2
---

# Feature Addition Skill

You are adding a new feature to the codebase. Follow this process:

1. **Explore**: Read existing code to understand the project structure and patterns.
2. **Plan**: Identify which files need to change. Explain your plan before writing code.
3. **Implement**: Add the feature following existing code patterns and conventions.
4. **Test**: Run existing tests if available. Add a simple test or verification.
5. **Document**: Update any relevant comments or docs.

Rules:
- Reuse existing patterns -- don't invent new ones unless necessary
- Keep changes minimal -- add only what the feature needs
- After implementation, verify with run_command
- If the project has tests, run them to check nothing broke
