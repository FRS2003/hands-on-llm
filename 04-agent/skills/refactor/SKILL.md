---
description: Improve code structure without changing behavior — extract, simplify, rename
triggers:
  - "refactor"
  - "restructure"
  - "clean up"
  - "simplify"
  - "improve structure"
  - "extract"
boundaries:
  - "add feature"
  - "fix bug"
  - "new behavior"
examples:
  - "Refactor the auth module to reduce duplication"
  - "Extract validation logic into a separate function"
tags:
  - refactoring
  - cleanup
  - architecture
priority: 1
---

# Refactoring Skill

You are improving code structure without changing behavior. Follow this process:

1. **Identify**: Find code that is duplicated, overly complex, or poorly named.
2. **Check tests**: If tests exist, use them as a safety net. If not, note that.
3. **Extract**: Pull out reusable logic into functions or modules.
4. **Simplify**: Remove dead code, flatten deep nesting, clarify names.
5. **Verify**: Ensure behavior is unchanged after each refactoring step.

Rules:
- NEVER change behavior during refactoring -- only structure
- Make ONE refactoring at a time, verify, then continue
- Prefer small, clear functions over large, clever ones
- If you are not sure whether a change preserves behavior, stop
