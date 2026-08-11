from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ABBA_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed", "cancelled", "timed_out"})
_LOCAL_PROCESSES: dict[str, subprocess.Popen] = {}


@dataclass(frozen=True)
class PendingAgateJob:
    job_id: str
    command: tuple[str, ...]
    stdout: str
    stderr: str


def submit_local_dev(
    stage: Path,
    *,
    job_timeout: int,
    remote_command: tuple[str, ...] | None = None,
) -> PendingAgateJob:
    if remote_command is None:
        # Local evaluation is part of the current Repository Horizon runtime.
        # Prefer the launcher-pinned runtime so an Agent invoking dev_eval with
        # a sandbox-provided system Python cannot silently change GPU packages.
        # Fall back to the current interpreter for ordinary installations.
        local_python = os.environ.get("ATREX_LOCAL_PYTHON", sys.executable)
        remote_command = (local_python, "repo_abba.py")
    command = [
        sys.executable,
        "-m",
        "repository_horizon.local_worker",
        "--stage",
        str(stage),
        "--timeout",
        str(job_timeout),
        "--command-json",
        json.dumps(list(remote_command), separators=(",", ":")),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parent.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=dict(os.environ),
    )
    job_id = f"local-{process.pid}"
    _LOCAL_PROCESSES[job_id] = process
    return PendingAgateJob(
        job_id=job_id,
        command=tuple(command),
        stdout="",
        stderr="",
    )


def get_local_job(stage: Path, job_id: str) -> AgateJobSnapshot:
    status_path = stage.parent / "local_job_result.json"
    if status_path.is_file():
        response = json.loads(status_path.read_text(encoding="utf-8"))
        status = str(response.get("status") or "failed")
        process = _LOCAL_PROCESSES.pop(job_id, None)
        if process is not None:
            process.wait(timeout=1)
    else:
        process = _LOCAL_PROCESSES.get(job_id)
        alive = process is not None and process.poll() is None
        if process is None and job_id.startswith("local-"):
            try:
                os.kill(int(job_id.removeprefix("local-")), 0)
                alive = True
            except (OSError, ValueError):
                alive = False
        if alive:
            response = {"job_id": job_id, "status": "running"}
            status = "running"
        else:
            response = {
                "job_id": job_id,
                "status": "failed",
                "command_ok": False,
                "error": "detached local worker exited without writing its terminal result",
                "result": {"exit_code": -1},
            }
            status = "failed"
    response["job_id"] = job_id
    stdout_path = stage.parent / "worker.stdout.log"
    stderr_path = stage.parent / "worker.stderr.log"
    return AgateJobSnapshot(
        job_id=job_id,
        status=status,
        response=response,
        stdout=(stdout_path.read_text(encoding="utf-8") if stdout_path.is_file() else ""),
        stderr=(stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""),
        command=("local-worker", job_id),
    )


@dataclass(frozen=True)
class AgateJobSnapshot:
    job_id: str
    status: str
    response: dict[str, Any]
    stdout: str
    stderr: str
    command: tuple[str, ...]

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES


@dataclass(frozen=True)
class AgateDevResult:
    payload: dict[str, Any]
    stdout: str
    stderr: str
    job_id: str
    command: tuple[str, ...]


def _all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _json_objects(output: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    values: list[dict[str, Any]] = []
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values


def _payload(output: str) -> dict[str, Any]:
    candidates = [output]
    for value in _json_objects(output):
        candidates.extend(_all_strings(value))
    for line in output.splitlines():
        try:
            candidates.extend(_all_strings(json.loads(line)))
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        for line in candidate.splitlines():
            index = line.find(ABBA_PREFIX)
            if index < 0:
                continue
            raw = line[index + len(ABBA_PREFIX) :].strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("Agate dev output has no repository ABBA result sentinel")


def _job_id(output: str) -> str:
    patterns = (
        r'"job_id"\s*:\s*"([^"]+)"',
        r'"id"\s*:\s*"([0-9a-fA-F-]{16,})"',
        r"\bjob[_ -]?id[=: ]+([0-9a-zA-Z_.-]{12,})",
    )
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def _job_response(output: str, expected_job_id: str = "") -> dict[str, Any]:
    candidates = []
    for value in _json_objects(output):
        job_id = value.get("job_id") or value.get("id")
        if not isinstance(job_id, str):
            continue
        if expected_job_id and job_id != expected_job_id:
            continue
        candidates.append(value)
    if candidates:
        return candidates[-1]
    job_id = _job_id(output)
    if job_id and (not expected_job_id or job_id == expected_job_id):
        return {"job_id": job_id, "status": "submitted"}
    raise ValueError("Agate output has no job object or job id")


def _endpoint_args(*, profile: str, url: str) -> list[str]:
    if url:
        return ["--url", url]
    if profile:
        return ["--profile", profile]
    return []


def submit_agate_dev(
    stage: Path,
    *,
    hardware: str,
    profile: str,
    url: str,
    job_timeout: int,
    submit_timeout: int = 120,
    remote_command: str = "python3 repo_abba.py",
    intent: str = "custom_harness",
    note: str = "repository horizon same-allocation ABBA verification",
) -> PendingAgateJob:
    command = ["agate", "dev", *_endpoint_args(profile=profile, url=url)]
    command += [
        "--gpu",
        hardware,
        "--working-dir",
        str(stage),
        "--job-timeout",
        str(job_timeout),
        "--intent",
        intent,
        "--note",
        note,
        "--no-wait",
        remote_command,
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=submit_timeout
    )
    combined = process.stdout + "\n" + process.stderr
    if process.returncode != 0:
        raise RuntimeError(f"agate dev submit exited {process.returncode}: {combined[-5000:]}")
    response = _job_response(combined)
    job_id = str(response.get("job_id") or response.get("id") or _job_id(combined))
    if not job_id:
        raise RuntimeError("agate dev --no-wait returned no job id")
    return PendingAgateJob(
        job_id=job_id,
        command=tuple(command),
        stdout=process.stdout,
        stderr=process.stderr,
    )


def submit_agate_profile(
    *,
    candidate: Path,
    reference_dir: Path,
    hardware: str,
    profile: str,
    url: str,
    level: str,
    kernel_name: str = "",
    kernel_regex: str = "",
    source: bool = False,
    launch_skip: int = 0,
    launch_count: int = 0,
    job_timeout: int = 600,
) -> PendingAgateJob:
    command = ["agate", "profile", *_endpoint_args(profile=profile, url=url)]
    command += [
        "--gpu", hardware,
        "--candidate", str(candidate),
        "--reference-dir", str(reference_dir),
        "--level", level,
        "--job-timeout", str(job_timeout),
    ]
    if kernel_name:
        command += ["--kernel-name", kernel_name]
    if kernel_regex:
        command += ["--kernel-regex", kernel_regex]
    if source:
        command.append("--source")
    if launch_skip:
        command += ["--launch-skip", str(launch_skip)]
    if launch_count:
        command += ["--launch-count", str(launch_count)]
    command.append("--no-wait")
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    combined = process.stdout + "\n" + process.stderr
    if process.returncode:
        raise RuntimeError(f"agate profile submit exited {process.returncode}: {combined[-5000:]}")
    response = _job_response(combined)
    job_id = str(response.get("job_id") or response.get("id") or _job_id(combined))
    if not job_id:
        raise RuntimeError("agate profile --no-wait returned no job id")
    return PendingAgateJob(job_id, tuple(command), process.stdout, process.stderr)


def get_agate_job(
    job_id: str,
    *,
    profile: str,
    url: str,
    http_timeout: int = 600,
) -> AgateJobSnapshot:
    command = ["agate", "get", *_endpoint_args(profile=profile, url=url)]
    command += ["--http-timeout", str(http_timeout), "--spec", job_id]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=http_timeout + 30
    )
    combined = process.stdout + "\n" + process.stderr
    try:
        response = _job_response(combined, expected_job_id=job_id)
    except ValueError:
        if process.returncode != 0:
            raise RuntimeError(
                f"agate get exited {process.returncode}: {combined[-5000:]}"
            )
        raise
    snapshot = AgateJobSnapshot(
        job_id=job_id,
        status=str(response.get("status") or "unknown"),
        response=response,
        stdout=process.stdout,
        stderr=process.stderr,
        command=tuple(command),
    )
    if process.returncode != 0 and not snapshot.terminal:
        raise RuntimeError(f"agate get exited {process.returncode}: {combined[-5000:]}")
    return snapshot


def collect_agate_dev(snapshot: AgateJobSnapshot) -> AgateDevResult:
    if not snapshot.terminal:
        raise RuntimeError(
            f"Agate job {snapshot.job_id} is not terminal: {snapshot.status}"
        )
    response = snapshot.response
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    if (
        response.get("status") != "succeeded"
        or response.get("error")
        or response.get("command_ok") is not True
        or result.get("exit_code") != 0
    ):
        raise RuntimeError(
            "Agate dev terminal status rejected: "
            + json.dumps(
                {
                    "job_id": snapshot.job_id,
                    "status": response.get("status"),
                    "error": response.get("error"),
                    "command_ok": response.get("command_ok"),
                    "exit_code": result.get("exit_code"),
                    "stderr": result.get("stderr"),
                },
                ensure_ascii=False,
            )
        )
    rendered = json.dumps(response, ensure_ascii=False)
    payload = _payload(rendered + "\n" + snapshot.stdout + "\n" + snapshot.stderr)
    return AgateDevResult(
        payload=payload,
        stdout=snapshot.stdout,
        stderr=snapshot.stderr,
        job_id=snapshot.job_id,
        command=snapshot.command,
    )


def profile_payload(snapshot: AgateJobSnapshot) -> dict[str, Any]:
    prefix = "__REPOSITORY_HORIZON_PROFILE_RESULT__="
    rendered = json.dumps(snapshot.response, ensure_ascii=False)
    candidates = [rendered, snapshot.stdout, snapshot.stderr]
    candidates.extend(_all_strings(snapshot.response))
    for candidate in candidates:
        for line in str(candidate).splitlines():
            index = line.find(prefix)
            if index < 0:
                continue
            try:
                value = json.loads(line[index + len(prefix) :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("repository profile output has no result sentinel")
