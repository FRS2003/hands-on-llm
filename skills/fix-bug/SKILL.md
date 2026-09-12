---
description: Diagnose and fix bugs using a systematic hypothesis-test-verify loop
triggers:
  - "fix"
  - "bug"
  - "error"
  - "crash"
  - "broken"
  - "not working"
  - "debug"
  - "issue"
boundaries:
  - "new feature"
  - "refactor"
  - "review"
examples:
  - "There is a crash when calculate_average is called with empty list"
  - "The login endpoint returns 500 when password is empty"
  - "Fix the race condition in cache.py"
tags:
  - debug
  - bug-fix
  - troubleshooting
priority: 3
---

# Bug Fixing Skill

You are debugging a reported issue. Follow this disciplined process:

1. **Reproduce**: Read the error message or bug description carefully. Find the exact file and line.
2. **Understand**: Read the surrounding code. What is it supposed to do? Why might it fail?
3. **Hypothesize**: Form ONE hypothesis about the root cause.
4. **Fix**: Make the smallest possible change to test that hypothesis.
5. **Verify**: After fixing, check that the fix doesn't break anything else.
6. **Test**: Add or run a test to prevent regression.

Rules:
- Read a file BEFORE editing it
- Make ONE change at a time
- If your hypothesis is wrong, try a different approach
- Explain your reasoning at each step
- After fixing, verify with run_command
