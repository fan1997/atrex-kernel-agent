from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from long_horizon.protocol import atomic_write_json

from .transport import AgateJobSnapshot, get_agate_job


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PendingVerification:
    schema_version: int
    evaluation_id: str
    job_id: str
    hardware: str
    profile: str
    url: str
    job_timeout: int
    wait_timeout: int
    stage: str
    base_commit: str
    candidate_commit: str
    changed_paths: tuple[str, ...]
    schedule: tuple[dict[str, Any], ...]
    repeats: int
    min_improvement_pct: float
    candidate_only: bool
    packed_bytes: int
    submit_command: tuple[str, ...]
    submitted_at: str

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["changed_paths"] = list(self.changed_paths)
        value["schedule"] = list(self.schedule)
        value["submit_command"] = list(self.submit_command)
        return value

    @classmethod
    def from_dict(cls, value: object) -> "PendingVerification":
        if not isinstance(value, dict):
            raise ValueError("pending verification must be a JSON object")
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            evaluation_id=str(value["evaluation_id"]),
            job_id=str(value["job_id"]),
            hardware=str(value["hardware"]),
            profile=str(value.get("profile", "")),
            url=str(value.get("url", "")),
            job_timeout=int(value["job_timeout"]),
            wait_timeout=int(value["wait_timeout"]),
            stage=str(value["stage"]),
            base_commit=str(value["base_commit"]),
            candidate_commit=str(value["candidate_commit"]),
            changed_paths=tuple(str(item) for item in value.get("changed_paths", [])),
            schedule=tuple(dict(item) for item in value["schedule"]),
            repeats=int(value["repeats"]),
            min_improvement_pct=float(value["min_improvement_pct"]),
            candidate_only=bool(value.get("candidate_only", False)),
            packed_bytes=int(value["packed_bytes"]),
            submit_command=tuple(str(item) for item in value.get("submit_command", [])),
            submitted_at=str(value["submitted_at"]),
        )

    @property
    def directory(self) -> Path:
        return Path(self.stage).parent

    @property
    def pending_path(self) -> Path:
        return self.directory / "pending.json"

    @property
    def result_path(self) -> Path:
        return self.directory / "result.json"


def save_pending(pending: PendingVerification) -> Path:
    atomic_write_json(pending.pending_path, pending.as_dict())
    return pending.pending_path


def load_pending(path: Path) -> PendingVerification:
    return PendingVerification.from_dict(json.loads(path.read_text(encoding="utf-8")))


def wait_for_terminal_job(
    pending: PendingVerification,
    *,
    initial_poll_seconds: float = 30.0,
    max_poll_seconds: float = 120.0,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AgateJobSnapshot:
    deadline = time.monotonic() + pending.wait_timeout
    delay = max(1.0, initial_poll_seconds)
    failures = 0
    last_error = ""
    while True:
        try:
            snapshot = get_agate_job(
                pending.job_id,
                profile=pending.profile,
                url=pending.url,
            )
            failures = 0
            last_error = ""
            status = {
                "schema_version": 1,
                "evaluation_id": pending.evaluation_id,
                "job_id": pending.job_id,
                "status": snapshot.status,
                "checked_at": utc_now(),
                "response": snapshot.response,
            }
            atomic_write_json(pending.directory / "status.json", status)
            if status_callback:
                status_callback(status)
            if snapshot.terminal:
                return snapshot
        except Exception as exc:
            failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            status = {
                "schema_version": 1,
                "evaluation_id": pending.evaluation_id,
                "job_id": pending.job_id,
                "status": "poll_retry",
                "checked_at": utc_now(),
                "consecutive_failures": failures,
                "error": last_error,
            }
            atomic_write_json(pending.directory / "status.json", status)
            if status_callback:
                status_callback(status)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detail = f"; last polling error: {last_error}" if last_error else ""
            raise TimeoutError(
                f"Agate job {pending.job_id} did not reach a terminal state within "
                f"{pending.wait_timeout} seconds{detail}"
            )
        time.sleep(min(delay, remaining))
        delay = min(max_poll_seconds, delay * 1.5)


def evaluation_handoff_path(workspace: Path) -> Path:
    return workspace / ".repository_horizon_runtime" / "evaluation_handoff.json"


def write_evaluation_handoff(workspace: Path, pending: PendingVerification) -> Path:
    path = evaluation_handoff_path(workspace)
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "status": "awaiting_evaluation",
            "evaluation_id": pending.evaluation_id,
            "job_id": pending.job_id,
            "candidate_commit": pending.candidate_commit,
            "pending_path": str(pending.pending_path),
            "submitted_at": pending.submitted_at,
        },
    )
    return path
