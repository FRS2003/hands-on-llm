"""
Centralized Multi-Agent Orchestration — 中心化多 Agent 协作

Architecture: Supervisor-Worker with async execution + failure handling
  Supervisor: Plans, decomposes tasks, assigns to workers, monitors progress
  Workers:    Execute sub-tasks via tool calls, return structured results
  Async:      Independent sub-tasks run concurrently via ThreadPoolExecutor
  Failure:    Retry (same worker) → Replan (new decomposition) → Skip + escalate

Key design decisions (from reference project):
  - Supervisor maintains control (never transfers authority)
  - Workers execute as tool calls (minimize coordination overhead)
  - Results are minimal structured summaries (reduce context pollution)
  - Path boundaries restrict workers to safe directories
  - Timeout protection: API-level (15s) + Worker-level (60s)
  - Progress callback: Supervisor receives real-time step updates
"""
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class WorkerTask:
    """A sub-task assigned to a worker agent."""
    task_id: str
    description: str
    context: str
    constraints: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # task_ids this depends on
    result: str = ""
    success: bool = False
    tools_used: list[str] = field(default_factory=list)
    retries: int = 0
    status: str = "pending"  # pending | running | done | failed | skipped


class WorkerAgent:
    """
    A lightweight sub-agent that executes ONE specific task.
    Minimal state, no memory, no routing — just execute and return.
    Supports progress callback for real-time Supervisor monitoring.
    """

    def __init__(self, name: str, tools: Any, llm_client, model: str = "deepseek-chat"):
        self.name = name
        self.tools = tools
        self.client = llm_client
        self.model = model

    def execute(self, task: WorkerTask, max_steps: int = 5,
                progress_callback: Callable | None = None) -> WorkerTask:
        """Execute a single sub-task with progress reporting."""
        task.status = "running"
        system_prompt = (
            f"You are a specialized worker agent: {self.name}.\n"
            f"Execute this specific sub-task and return the result.\n"
            f"Constraints: {', '.join(task.constraints)}\n\n"
            f"Context:\n{task.context}\n\n"
            f"Sub-task: {task.description}\n\n"
            f"Rules: Read before editing. Make minimal changes. Use exact string matching for edits.\n"
            f"When done, call submit_result with a clear summary."
        )

        messages = [{"role": "system", "content": system_prompt}]
        step = 0
        tools_used = []
        consecutive_no_progress = 0
        last_tool_count = 0

        while step < max_steps:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    tools=self.tools.get_tool_definitions(),
                    tool_choice="auto", temperature=0.3, max_tokens=500,
                    timeout=15,  # API-level timeout
                )
            except Exception as e:
                task.result = f"API call failed: {str(e)[:200]}"
                task.success = False
                task.status = "failed"
                break

            msg = resp.choices[0].message

            if msg.tool_calls:
                tc = msg.tool_calls[0]
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except Exception:
                    continue

                if tool_name == "submit_result":
                    task.result = tool_args.get("summary", "Done.")
                    task.success = True
                    task.status = "done"
                    break

                result = self.tools.execute(tool_name, **tool_args)
                tools_used.append(tool_name)
                messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)[:2000]})

                # Progress check: detect looping
                if len(tools_used) > last_tool_count:
                    consecutive_no_progress = 0
                    last_tool_count = len(tools_used)
                else:
                    consecutive_no_progress += 1

                # Report progress to Supervisor
                if progress_callback:
                    progress_callback({
                        "task_id": task.task_id,
                        "worker": self.name,
                        "step": step + 1,
                        "max_steps": max_steps,
                        "action": tool_name,
                        "tools_used": len(tools_used),
                        "no_progress": consecutive_no_progress,
                        "status": "running",
                    })

                # Worker-level stall detection
                if consecutive_no_progress >= 2:
                    task.result = f"Stalled: no progress in {consecutive_no_progress} steps."
                    task.success = False
                    task.status = "failed"
                    break
            else:
                task.result = msg.content or "Done."
                task.success = True
                task.status = "done"
                break

            step += 1

        task.tools_used = tools_used
        if not task.result:
            task.result = f"Max steps ({max_steps}) reached."
            task.success = False
            task.status = "failed"
        if task.status == "running":
            task.status = "done" if task.success else "failed"
        return task


class Supervisor:
    """
    Central orchestrator: decomposes task → assigns to workers → monitors progress → handles failures.

    Execution modes:
      - execute_fork():        Sequential, synchronous (backward compatible)
      - execute_fork_async():  Concurrent with real-time failure handling

    Failure escalation: Retry → Replan → Skip & escalate
    """

    def __init__(self, tools: Any, llm_client, model: str = "deepseek-chat"):
        self.tools = tools
        self.client = llm_client
        self.model = model
        self.workers: dict[str, WorkerAgent] = {}
        self._progress_log: list[dict] = []  # real-time progress events

    def register_worker(self, name: str, worker: WorkerAgent):
        self.workers[name] = worker

    # ── Task decomposition ──

    def decompose_task(self, task: str) -> list[WorkerTask]:
        """Use LLM to decompose a complex task into sub-tasks with dependency info."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": (
                    f"Break this task into 2-4 sub-tasks. Include dependencies.\n\n"
                    f"Task: {task}\n\n"
                    f"Format as JSON list with 'description', 'worker_type' (reader/editor/tester/reviewer), "
                    f"and 'depends_on' (list of sub-task indices this depends on, empty list if independent).\n\n"
                    f'Example: [{{"description": "Read and analyze auth.py", "worker_type": "reader", "depends_on": []}}, '
                    f'{{"description": "Fix auth.py bug", "worker_type": "editor", "depends_on": [0]}}]'
                )}],
                temperature=0.2, max_tokens=2000,
            )
            text = resp.choices[0].message.content or ""
            text = text.strip()
            if not text:
                return [WorkerTask(task_id="main", description=task, context=task, constraints=["read-write"])]
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            sub_tasks = json.loads(text)
            result = []
            for i, st in enumerate(sub_tasks):
                raw_deps = st.get("depends_on", [])
                # Convert numeric indices to task_id strings
                dep_ids = [f"sub_{d}" if isinstance(d, int) else str(d) for d in raw_deps]
                result.append(WorkerTask(
                    task_id=f"sub_{i}",
                    description=st["description"],
                    context=task,
                    constraints=self._get_constraints(st.get("worker_type", "reader")),
                    depends_on=dep_ids,
                ))
            return result
        except Exception:
            return [WorkerTask(task_id="main", description=task, context=task, constraints=["read-write"])]

    def _get_constraints(self, worker_type: str) -> list[str]:
        constraints = {
            "reader": ["read-only", "no edits", "report findings only"],
            "editor": ["read-write", "edit_file only", "verify after edit"],
            "tester": ["read-only", "run_command allowed", "report test results"],
            "reviewer": ["read-only", "inspect and report", "no changes"],
        }
        return constraints.get(worker_type, ["read-write"])

    # ── Decomposition Router ──

    def _should_decompose(self, task: str) -> bool:
        """
        LLM Router: decide whether this task benefits from decomposition.

        Simple tasks ("read a file and explain") → single Worker, no overhead.
        Complex tasks ("fix bug + add tests + refactor") → decompose.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": (
                    "Decide if this coding task should be split into sub-tasks (DECOMPOSE) "
                    "or can be done as one action (SINGLE).\n\n"
                    "Choose DECOMPOSE if: multiple independent file operations, or distinct "
                    "phases (read → fix → test), or complexity requiring different approaches.\n"
                    "Choose SINGLE if: one clear action (read, explain, find), or simple fix "
                    "in a single file.\n\n"
                    f"Task: {task}\n\nAnswer ONLY 'DECOMPOSE' or 'SINGLE'."
                )}],
                temperature=0.0, max_tokens=5,
            )
            return "DECOMPOSE" in resp.choices[0].message.content.strip().upper()
        except Exception:
            # On failure, default to decompose (safe side — overhead is just one extra LLM call)
            return True

    # ── Worker assignment ──

    def _assign_worker(self, sub_task: WorkerTask) -> WorkerAgent | None:
        """Pick the best worker for a sub-task. Returns None if no worker available."""
        desc = sub_task.description.lower()
        worker_type = "editor" if any(kw in desc for kw in ["edit", "fix", "write", "add", "change"]) else "reader"
        return self.workers.get(worker_type, list(self.workers.values())[0] if self.workers else None)

    # ── Progress monitoring ──

    def _on_progress(self, event: dict):
        """Real-time progress callback from workers."""
        self._progress_log.append(event)
        worker = event.get("worker", "?")
        tid = event.get("task_id", "?")
        step = event.get("step", "?")
        action = event.get("action", "?")
        stall = event.get("no_progress", 0)
        status = event.get("status", "?")
        print(f"  [{status}] {worker}/{tid} step={step} {action}", end="")
        if stall >= 2:
            print(f" STALLED(stall={stall})", end="")
        print()

    # ── Sequential execution (backward compatible) ──

    def execute_fork(self, task: str, parallel: bool = False) -> dict:
        """Sequential execution. Kept for backward compatibility."""
        sub_tasks = self.decompose_task(task)
        results = []
        for sub_task in sub_tasks:
            worker = self._assign_worker(sub_task)
            if worker:
                sub_task = worker.execute(sub_task)
            else:
                sub_task.result = "No worker available."
            results.append(sub_task)
        return self._synthesize(results)

    # ── Async execution with failure handling ──

    def execute_fork_async(self, task: str, max_retries: int = 1,
                           worker_timeout: int = 60) -> dict:
        """
        Concurrent execution with real-time monitoring and failure escalation.

        First checks if the task needs decomposition (LLM Router).
        Simple tasks run as a single Worker, complex tasks get decomposed.

        Failure handling per sub-task:
          1. Retry: same worker, same task
          2. Replan: re-decompose this sub-task
          3. Skip: skip dependent tasks, escalate to main Agent
        """
        self._progress_log = []

        # Router: does this task even need decomposition?
        if not self._should_decompose(task):
            st = WorkerTask(task_id="main", description=task, context=task,
                          constraints=["read-write"])
            worker = self._assign_worker(st)
            if worker:
                st = worker.execute(st, progress_callback=self._on_progress)
            else:
                st.result = "No worker available."
            return self._synthesize([st])

        sub_tasks = self.decompose_task(task)

        if not sub_tasks:
            return {"total_subtasks": 0, "successful": 0, "results": [], "overall_success": False}

        # For single-task fallback, run inline
        if len(sub_tasks) == 1:
            return self.execute_fork(task)

        results: dict[str, WorkerTask] = {}  # task_id → result
        executor = ThreadPoolExecutor(max_workers=min(4, len(sub_tasks)))

        def _run_worker(st: WorkerTask) -> WorkerTask:
            """Run a single worker with retry logic, in a thread."""
            worker = self._assign_worker(st)
            if not worker:
                st.result = "No worker available."
                st.status = "failed"
                return st

            # Attempt 1: execute
            st = worker.execute(st, progress_callback=self._on_progress)

            # Retry on failure (strategy 1)
            while not st.success and st.retries < max_retries:
                st.retries += 1
                st.result = ""
                st = worker.execute(st, progress_callback=self._on_progress)

            # Replan on persistent failure (strategy 2)
            if not st.success:
                alt_tasks = self.decompose_task(
                    f"Alternative approach to: {st.description}. "
                    f"Previous attempt failed: {st.result[:200]}"
                )
                if alt_tasks and len(alt_tasks) > 0:
                    alt = alt_tasks[0]
                    alt.task_id = st.task_id
                    alt.depends_on = st.depends_on
                    alt.retries = st.retries + 1
                    st = worker.execute(alt, progress_callback=self._on_progress)

            return st

        # Track which tasks are ready to run (all dependencies satisfied)
        pending = {st.task_id: st for st in sub_tasks}
        futures: dict = {}

        def _submit_ready():
            """Submit all pending tasks whose dependencies are satisfied."""
            for tid, st in list(pending.items()):
                deps_met = all(
                    d not in results or results[d].success
                    for d in st.depends_on
                )
                if deps_met and tid not in futures:
                    futures[tid] = executor.submit(_run_worker, st)

        # Main event loop
        while pending:
            _submit_ready()

            if not futures:
                # Remaining tasks have unsatisfied dependencies — skip them
                for tid, st in list(pending.items()):
                    st.result = f"Skipped: prerequisite task(s) failed ({st.depends_on})"
                    st.success = False
                    st.status = "skipped"
                    results[tid] = st
                    del pending[tid]
                break

            # Wait for next completion (with timeout so we can report progress)
            try:
                for future in as_completed(futures, timeout=worker_timeout):
                    st = future.result()
                    results[st.task_id] = st
                    del pending[st.task_id]
                    del futures[st.task_id]

                    if not st.success:
                        # Strategy 3: skip dependent tasks
                        print(f"  [FAIL] {st.task_id}: {st.result[:120]} — "
                              f"dependent tasks will be skipped")
                        for tid, dep_st in list(pending.items()):
                            if st.task_id in dep_st.depends_on:
                                dep_st.result = f"Skipped: {st.task_id} failed"
                                dep_st.success = False
                                dep_st.status = "skipped"
                                results[tid] = dep_st
                                del pending[tid]
                    else:
                        print(f"  [OK] {st.task_id}: {st.description[:80]}")

                    break  # Re-enter loop to submit newly unblocked tasks
            except FutureTimeout:
                # Worker-level timeout — mark remaining futures as failed
                print(f"  [TIMEOUT] Some workers exceeded {worker_timeout}s limit")
                for tid, f in list(futures.items()):
                    if not f.done():
                        st = pending[tid]
                        st.result = f"Timed out after {worker_timeout}s"
                        st.success = False
                        st.status = "failed"
                        results[tid] = st
                        del pending[tid]
                        del futures[tid]
                break

        executor.shutdown(wait=False)
        return self._synthesize(list(results.values()))

    # ── Result synthesis ──

    def _synthesize(self, sub_tasks: list[WorkerTask]) -> dict:
        """Aggregate worker results into a structured summary."""
        summary_parts = []
        for st in sorted(sub_tasks, key=lambda s: s.task_id):
            icon = {None: "?", "done": "+", "failed": "-", "skipped": "~"}.get(st.status, "?")
            summary_parts.append(f"[{icon}] {st.task_id}: {st.description[:80]}"
                                 f" — {st.result[:150]}")
        return {
            "total_subtasks": len(sub_tasks),
            "successful": sum(1 for st in sub_tasks if st.success),
            "skipped": sum(1 for st in sub_tasks if st.status == "skipped"),
            "failed": sum(1 for st in sub_tasks if st.status == "failed"),
            "results": summary_parts,
            "overall_success": all(st.success for st in sub_tasks),
            "progress_log": self._progress_log,
        }
