from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .store import CampaignStore


SAFE_GIT_REF = re.compile(r"[0-9A-Za-z._/~-]{1,200}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_catalog(repository_root: Path | None) -> list[dict[str, Any]]:
    if repository_root is None:
        return []
    resources: list[Path] = []
    prompt_root = repository_root / "orchestrator" / "prompts"
    if prompt_root.is_dir():
        resources.extend(sorted(prompt_root.glob("*.md")))
    resources.extend(
        path
        for path in (
            repository_root / "orchestrator" / "optimization_policy.py",
            repository_root / "reference" / "CLAUDE.md",
        )
        if path.is_file()
    )
    return [
        {
            "resource": "@runtime/" + str(path.relative_to(repository_root)),
            "chars": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in resources
    ]


def _memory_digest(workspace: Path, limit: int = 100) -> list[dict[str, Any]]:
    memory_dir = workspace / "memory"
    versions: list[tuple[int, Path]] = []
    if memory_dir.is_dir():
        for path in memory_dir.glob("v*.json"):
            match = re.fullmatch(r"v(\d+)\.json", path.name)
            if match:
                versions.append((int(match.group(1)), path))
    result: list[dict[str, Any]] = []
    for version, path in sorted(versions)[-max(1, min(limit, 500)) :]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        optimization = data.get("optimization") or {}
        performance = data.get("performance") or {}
        correctness = data.get("correctness") or {}
        quality_gate = data.get("quality_gate") or {}
        directions = data.get("open_directions") or []
        result.append(
            {
                "version": f"v{version}",
                "correctness": correctness.get("status"),
                "quality_gate": quality_gate.get("result"),
                "failure_reason": quality_gate.get("failure_reason"),
                "latency_us": performance.get("latency_us"),
                "action_category": optimization.get("action_category"),
                "action_description": optimization.get("action_description"),
                "git_commit_hash": data.get("git_commit_hash"),
                "open_directions": [
                    {
                        "direction": item.get("direction"),
                        "rationale": item.get("rationale"),
                    }
                    for item in directions[:3]
                    if isinstance(item, dict)
                ],
            }
        )
    return result


def campaign_digest(store: CampaignStore) -> dict[str, Any]:
    memories = _memory_digest(store.workspace)
    event_kinds: dict[str, int] = {}
    run_outcomes: list[dict[str, Any]] = []
    if store.events_path.exists():
        with store.events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = str(event.get("kind") or "unknown")
                event_kinds[kind] = event_kinds.get(kind, 0) + 1
                if kind == "session_finished":
                    metadata = event.get("metadata") or {}
                    run_outcomes.append(
                        {
                            "cursor": event.get("cursor"),
                            "timestamp": event.get("timestamp"),
                            "run_id": event.get("run_id"),
                            "status": metadata.get("status"),
                            "exit_status": metadata.get("exit_status"),
                            "timed_out": metadata.get("timed_out"),
                            "prompt_kind": metadata.get("prompt_kind"),
                        }
                    )
    return {
        "schedule": store.schedule_state(),
        "event_kinds": event_kinds,
        "recent_run_outcomes": run_outcomes[-20:],
        "version_trajectory": memories,
        "captured_executor_prompts": store.list_session_prompts(limit=20),
        "supervisor_guidance_history": store.guidance_history(limit=20),
        "interpretation_note": (
            "This is a navigation digest, not ground truth. Verify material conclusions against "
            "Git, immutable evaluator output, profiles, and raw events. Compare each prior guidance "
            "message with later Executor behavior to assess follow-through."
        ),
    }


def _git(workspace: Path, args: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"exit_status": 127, "stdout": "", "stderr": str(exc)}
    return {
        "exit_status": result.returncode,
        "stdout": result.stdout[-100_000:],
        "stderr": result.stderr[-20_000:],
    }


def latest_memory_version(workspace: Path) -> int:
    highest = -1
    memory = workspace / "memory"
    if not memory.is_dir():
        return highest
    for path in memory.glob("v*.json"):
        match = re.fullmatch(r"v(\d+)\.json", path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


def campaign_facts(
    store: CampaignStore,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    workspace = store.workspace
    head = _git(workspace, ["rev-parse", "HEAD"])
    status = _git(workspace, ["status", "--short", "--branch"])
    diff_stat = _git(workspace, ["diff", "--stat"])
    stall_path = workspace / ".orchestrator_state.json"
    try:
        stall = json.loads(stall_path.read_text(encoding="utf-8")).get("stall")
    except (OSError, json.JSONDecodeError):
        stall = None
    return {
        "campaign_id": store.campaign_id,
        "workspace": str(workspace),
        "event_cursor": store.cursor,
        "current_run": store.current_run(),
        "git": {
            "head": head["stdout"].strip() if head["exit_status"] == 0 else "",
            "status": status,
            "diff_stat": diff_stat,
        },
        "stall": stall,
        "latest_memory_version": latest_memory_version(workspace),
        "execution_contract": {
            **store.execution_context(),
            "current_prompt_kind": store.current_run().get("prompt_kind"),
            "latest_prompt_snapshot": (
                store.list_session_prompts(limit=1)[0]
                if store.list_session_prompts(limit=1)
                else None
            ),
            "prompt_catalog": _prompt_catalog(repository_root),
            "prompt_access": (
                "Read @executor/latest/base and @executor/latest/effective for the exact AKA and "
                "delivered prompts. Read @runtime/... resources from prompt_catalog for the "
                "framework-wide execution rules."
            ),
        },
        "campaign_digest": campaign_digest(store),
    }


def inspect_git(
    workspace: Path,
    operation: str,
    revision: str = "HEAD",
    path: str = "",
    max_count: int = 20,
) -> dict[str, Any]:
    operation = operation.strip().lower()
    if revision and not SAFE_GIT_REF.fullmatch(revision):
        raise ValueError("revision contains unsupported characters")
    if path:
        resolved = (workspace / path).resolve()
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError as exc:
            raise ValueError("path escapes the campaign workspace") from exc

    if operation == "status":
        args = ["status", "--short", "--branch"]
    elif operation == "log":
        args = ["log", f"-{max(1, min(max_count, 100))}", "--oneline", "--decorate"]
    elif operation == "diff":
        args = ["diff", revision]
        if path:
            args += ["--", path]
    elif operation == "show":
        args = ["show", "--stat", "--summary", revision]
        if path:
            args += ["--", path]
    else:
        raise ValueError("operation must be status, log, diff, or show")
    return _git(workspace, args)


def read_workspace_file(
    workspace: Path,
    path: str,
    start_line: int = 1,
    max_lines: int = 300,
) -> dict[str, Any]:
    requested = (workspace / path).resolve()
    try:
        requested.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError("path escapes the campaign workspace") from exc
    if not requested.is_file():
        raise ValueError(f"workspace file does not exist: {path}")
    if requested.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("file is larger than the 5 MiB supervisor read limit")
    first = max(1, int(start_line))
    count = max(1, min(int(max_lines), 2000))
    selected: list[str] = []
    with requested.open("r", encoding="utf-8", errors="replace") as handle:
        for number, line in enumerate(handle, start=1):
            if number < first:
                continue
            selected.append(line.rstrip("\n"))
            if len(selected) >= count:
                break
    return {
        "path": path,
        "start_line": first,
        "line_count": len(selected),
        "text": "\n".join(selected),
    }


def read_supervisor_resource(
    store: CampaignStore,
    repository_root: Path,
    resource: str,
    start_line: int = 1,
    max_lines: int = 300,
) -> dict[str, Any]:
    """Read captured Executor prompts or allowlisted AKA execution-contract sources."""
    if resource.startswith("@executor/"):
        parts = resource.split("/")
        if len(parts) != 3:
            raise ValueError("Executor resource must be @executor/<session|latest>/<base|effective>")
        text = store.read_session_prompt(parts[1], parts[2])
    elif resource.startswith("@runtime/"):
        relative = resource.removeprefix("@runtime/")
        candidate = (repository_root / relative).resolve()
        allowed_files = {
            (repository_root / "orchestrator" / "optimization_policy.py").resolve(),
            (repository_root / "reference" / "CLAUDE.md").resolve(),
        }
        prompt_root = (repository_root / "orchestrator" / "prompts").resolve()
        allowed = candidate in allowed_files
        try:
            candidate.relative_to(prompt_root)
            allowed = allowed or candidate.suffix == ".md"
        except ValueError:
            pass
        if not allowed or not candidate.is_file():
            raise ValueError("runtime resource is outside the allowlisted AKA execution contract")
        if candidate.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("runtime resource is larger than the 5 MiB read limit")
        text = candidate.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError("virtual resource must start with @executor/ or @runtime/")
    first = max(1, int(start_line))
    count = max(1, min(int(max_lines), 2000))
    lines = text.splitlines()[first - 1 : first - 1 + count]
    return {
        "path": resource,
        "start_line": first,
        "line_count": len(lines),
        "text": "\n".join(lines),
    }


def inspect_environment(
    store: CampaignStore,
    diagnostic: str,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    """Run fixed, non-mutating environment probes; syntax checks execute only in scratch."""
    diagnostic = diagnostic.strip().lower()
    if diagnostic == "toolchain":
        tools = [sys.executable, "git", "codex", "agate", "ncu"]
        result: list[dict[str, Any]] = []
        for tool in tools:
            executable = tool if Path(tool).is_file() else shutil.which(tool)
            item: dict[str, Any] = {"tool": tool, "path": executable or ""}
            if executable:
                try:
                    probe = subprocess.run(
                        [executable, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    item.update(
                        {
                            "exit_status": probe.returncode,
                            "version_output": (probe.stdout + probe.stderr).strip()[:2000],
                        }
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    item["error"] = str(exc)
            result.append(item)
        return {"diagnostic": diagnostic, "tools": result}
    if diagnostic in {"workspace_layout", "syntax"}:
        requested = paths or []
        if not requested or len(requested) > 50:
            raise ValueError("diagnostic paths must contain between 1 and 50 workspace paths")
        inspected: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="atrex-supervisor-diagnostic-") as temp_dir:
            scratch = Path(temp_dir)
            for raw_path in requested:
                source = (store.workspace / raw_path).resolve()
                try:
                    source.relative_to(store.workspace.resolve())
                except ValueError as exc:
                    raise ValueError(f"diagnostic path escapes workspace: {raw_path}") from exc
                item: dict[str, Any] = {
                    "path": raw_path,
                    "exists": source.exists(),
                    "resolved_path": str(source),
                    "is_symlink": (store.workspace / raw_path).is_symlink(),
                }
                if source.is_file():
                    item.update({"size": source.stat().st_size, "sha256": _sha256(source)})
                    if diagnostic == "syntax" and source.suffix in {".py", ".sh"}:
                        target = scratch / source.name
                        shutil.copy2(source, target)
                        command = (
                            [sys.executable, "-m", "py_compile", str(target)]
                            if source.suffix == ".py"
                            else ["bash", "-n", str(target)]
                        )
                        check = subprocess.run(
                            command,
                            cwd=str(scratch),
                            capture_output=True,
                            text=True,
                            timeout=20,
                        )
                        item["syntax"] = {
                            "exit_status": check.returncode,
                            "output": (check.stdout + check.stderr)[-4000:],
                        }
                    elif diagnostic == "workspace_layout" and source.suffix == ".py":
                        try:
                            tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
                            item["static_imports"] = sorted(
                                {
                                    node.module or ""
                                    for node in ast.walk(tree)
                                    if isinstance(node, ast.ImportFrom)
                                }
                                | {
                                    alias.name
                                    for node in ast.walk(tree)
                                    if isinstance(node, ast.Import)
                                    for alias in node.names
                                }
                            )[:200]
                        except SyntaxError as exc:
                            item["parse_error"] = str(exc)
                inspected.append(item)
        return {"diagnostic": diagnostic, "paths": inspected}
    raise ValueError("diagnostic must be toolchain, workspace_layout, or syntax")


def inspect_gateway_job(store: CampaignStore, job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{3,200}", job_id):
        raise ValueError("job_id contains unsupported characters")
    matches = []
    if store.events_path.exists():
        with store.events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = str(event.get("raw") or "")
                if job_id in raw:
                    matches.append(event)
    return {
        "job_id": job_id,
        "source": "captured executor events",
        "matches": matches[-100:],
        "note": (
            "V1 independently returns the raw captured gateway evidence. Direct read-only "
            "gateway API lookup can be added without changing the executor integration."
        ),
    }
