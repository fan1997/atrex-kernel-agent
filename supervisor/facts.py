from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .store import CampaignStore


SAFE_GIT_REF = re.compile(r"[0-9A-Za-z._/~-]{1,200}")


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


def campaign_facts(store: CampaignStore) -> dict[str, Any]:
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


def inspect_gateway_job(store: CampaignStore, job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{3,200}", job_id):
        raise ValueError("job_id contains unsupported characters")
    matches = []
    for event in store.read_events(after_cursor=0, limit=1000):
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
