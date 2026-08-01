from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterable

from .models import AgentEvent, ControlRequest, utc_now


def campaign_id_for_workspace(workspace: Path) -> str:
    resolved = str(workspace.resolve())
    slug = re.sub(r"[^a-z0-9]+", "_", workspace.name.lower()).strip("_")
    digest = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{slug or 'campaign'}_{digest}"


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class CampaignStore:
    """Parent-owned durable event and control storage for one campaign."""

    def __init__(self, root: Path, workspace: Path):
        self.root = root.resolve()
        self.workspace = workspace.resolve()
        self.campaign_id = campaign_id_for_workspace(self.workspace)
        self.campaign_dir = self.root / "campaigns" / self.campaign_id
        self.events_path = self.campaign_dir / "raw-events.jsonl"
        self.trusted_facts_path = self.campaign_dir / "trusted-facts.jsonl"
        self.guidance_pending = self.campaign_dir / "guidance" / "pending"
        self.guidance_consumed = self.campaign_dir / "guidance" / "consumed"
        self.guidance_superseded = self.campaign_dir / "guidance" / "superseded"
        self.control_pending = self.campaign_dir / "control" / "pending"
        self.control_consumed = self.campaign_dir / "control" / "consumed"
        self.sessions_dir = self.campaign_dir / "sessions"
        self.activations_dir = self.campaign_dir / "activations"
        self.state_dir = self.campaign_dir / "state"
        for path in (
            self.campaign_dir,
            self.guidance_pending,
            self.guidance_consumed,
            self.guidance_superseded,
            self.control_pending,
            self.control_consumed,
            self.sessions_dir,
            self.activations_dir,
            self.state_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.campaign_dir, 0o700)
        except OSError:
            pass
        self._lock = threading.Lock()
        self._cursor = self._read_last_cursor()

    def _read_last_cursor(self) -> int:
        if not self.events_path.exists():
            return 0
        last = ""
        try:
            with self.events_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.strip():
                        last = line
            return int(json.loads(last).get("cursor") or 0) if last else 0
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def append_event(self, event: AgentEvent) -> AgentEvent:
        with self._lock:
            self._cursor += 1
            recorded = AgentEvent(
                campaign_id=event.campaign_id,
                run_id=event.run_id,
                kind=event.kind,
                timestamp=event.timestamp,
                stream=event.stream,
                raw=event.raw,
                metadata=event.metadata,
                cursor=self._cursor,
            )
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(recorded.as_dict(), ensure_ascii=False) + "\n")
            return recorded

    def append_fact(self, kind: str, data: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), "kind": kind, "data": data}
        with self._lock:
            with self.trusted_facts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_events(
        self,
        after_cursor: int = 0,
        limit: int = 100,
        event_types: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        allowed = set(event_types)
        result: list[dict[str, Any]] = []
        if not self.events_path.exists():
            return result
        with self.events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(event.get("cursor") or 0) <= after_cursor:
                    continue
                if allowed and str(event.get("kind")) not in allowed:
                    continue
                result.append(event)
                if len(result) >= max(1, min(limit, 1000)):
                    break
        return result

    def read_recent_events(
        self,
        after_cursor: int = 0,
        through_cursor: int | None = None,
        limit: int = 1000,
        event_types: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        """Return the newest bounded event window, preserving chronological order."""
        allowed = set(event_types)
        newest: deque[dict[str, Any]] = deque(maxlen=max(1, min(limit, 5000)))
        if not self.events_path.exists():
            return []
        ceiling = through_cursor if through_cursor is not None else self.cursor
        with self.events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cursor = int(event.get("cursor") or 0)
                if cursor <= after_cursor or cursor > ceiling:
                    continue
                if allowed and str(event.get("kind")) not in allowed:
                    continue
                newest.append(event)
        return list(newest)

    def count_events(self, after_cursor: int = 0, through_cursor: int | None = None) -> int:
        if not self.events_path.exists():
            return 0
        ceiling = through_cursor if through_cursor is not None else self.cursor
        count = 0
        with self.events_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    cursor = int(json.loads(line).get("cursor") or 0)
                except (ValueError, json.JSONDecodeError):
                    continue
                if after_cursor < cursor <= ceiling:
                    count += 1
        return count

    def wait_events(self, after_cursor: int, timeout_seconds: int) -> list[dict[str, Any]]:
        deadline = time.monotonic() + max(0, min(timeout_seconds, 30))
        while True:
            events = self.read_events(after_cursor=after_cursor, limit=200)
            if events or time.monotonic() >= deadline:
                return events
            time.sleep(0.25)

    def set_current_run(self, data: dict[str, Any]) -> None:
        _atomic_json(self.state_dir / "current-run.json", data)

    def current_run(self) -> dict[str, Any]:
        path = self.state_dir / "current-run.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def set_execution_context(self, data: dict[str, Any]) -> None:
        """Persist the non-secret controller contract visible to the Supervisor."""
        _atomic_json(self.state_dir / "execution-context.json", data)

    def execution_context(self) -> dict[str, Any]:
        path = self.state_dir / "execution-context.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def record_session_prompt(
        self,
        run_id: str,
        base_prompt: str,
        effective_prompt: str,
        prompt_kind: str,
        attempt: int,
        guidance_request_id: str = "",
    ) -> dict[str, Any]:
        """Capture exactly what AKA generated and what the Executor actually received."""
        session_id = run_id.rsplit(":", 1)[-1]
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        base_path = session_dir / "base-prompt.md"
        effective_path = session_dir / "effective-prompt.md"
        base_path.write_text(base_prompt, encoding="utf-8")
        effective_path.write_text(effective_prompt, encoding="utf-8")
        metadata = {
            "campaign_id": self.campaign_id,
            "run_id": run_id,
            "session_id": session_id,
            "captured_at": utc_now(),
            "prompt_kind": prompt_kind,
            "attempt": int(attempt),
            "guidance_request_id": guidance_request_id,
            "base_prompt_sha256": hashlib.sha256(base_prompt.encode("utf-8")).hexdigest(),
            "effective_prompt_sha256": hashlib.sha256(
                effective_prompt.encode("utf-8")
            ).hexdigest(),
            "base_prompt_chars": len(base_prompt),
            "effective_prompt_chars": len(effective_prompt),
            "base_prompt_path": str(base_path),
            "effective_prompt_path": str(effective_path),
        }
        _atomic_json(session_dir / "metadata.json", metadata)
        _atomic_json(self.state_dir / "latest-session-prompt.json", metadata)
        self.append_fact("session_prompt_captured", metadata)
        return metadata

    def list_session_prompts(self, limit: int = 20) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("*/metadata.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        records.sort(key=lambda item: str(item.get("captured_at") or ""), reverse=True)
        return records[: max(1, min(int(limit), 100))]

    def read_session_prompt(self, session_id: str, variant: str = "effective") -> str:
        if session_id == "latest":
            records = self.list_session_prompts(limit=1)
            if not records:
                raise ValueError("no captured Executor prompts")
            session_id = str(records[0].get("session_id") or "")
        if not re.fullmatch(r"[0-9A-Za-z._-]{1,200}", session_id):
            raise ValueError("invalid session id")
        filename = {
            "base": "base-prompt.md",
            "effective": "effective-prompt.md",
        }.get(variant)
        if filename is None:
            raise ValueError("prompt variant must be base or effective")
        path = self.sessions_dir / session_id / filename
        if not path.is_file():
            raise ValueError(f"captured prompt does not exist: {session_id}/{variant}")
        return path.read_text(encoding="utf-8", errors="replace")

    def guidance_history(self, limit: int = 20) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for status, directory in (
            ("pending", self.guidance_pending),
            ("consumed", self.guidance_consumed),
            ("superseded", self.guidance_superseded),
        ):
            for path in directory.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                records.append({**data, "lifecycle_status": status, "artifact": str(path)})
        records.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return records[: max(1, min(int(limit), 100))]

    def schedule_state(self) -> dict[str, Any]:
        path = self.state_dir / "supervisor-schedule.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        return {
            "baseline_review_attempted": bool(data.get("baseline_review_attempted", False)),
            "completed_iterations": _nonnegative_int(data.get("completed_iterations")),
            "last_periodic_review_iteration": _nonnegative_int(
                data.get("last_periodic_review_iteration")
            ),
            "manual_reviews": _nonnegative_int(data.get("manual_reviews")),
            "stop_reviews": _nonnegative_int(data.get("stop_reviews")),
        }

    def write_schedule_state(self, data: dict[str, Any]) -> None:
        _atomic_json(self.state_dir / "supervisor-schedule.json", data)

    def next_logical_iteration(self) -> int:
        return int(self.schedule_state()["completed_iterations"]) + 1

    def queue_guidance(
        self,
        message: str,
        valid_for_iterations: int = 1,
        source: str = "supervisor",
    ) -> Path:
        state = self.schedule_state()
        completed_iterations = int(state["completed_iterations"])
        current_run = self.current_run()
        try:
            controller_pid = int(current_run.get("controller_pid") or 0)
            if controller_pid <= 0:
                raise ValueError("missing controller pid")
            os.kill(controller_pid, 0)
            controller_alive = True
        except (OSError, TypeError, ValueError):
            controller_alive = False
        active_iteration = (
            current_run.get("status") == "running"
            and current_run.get("prompt_kind") != "baseline"
            and controller_alive
        )
        first_iteration = completed_iterations + (2 if active_iteration else 1)
        valid_iterations = max(1, min(int(valid_for_iterations), 20))
        request_id = uuid.uuid4().hex
        path = self.guidance_pending / f"{time.time_ns()}-{request_id}.json"
        with self._lock:
            # Strong Supervisor guidance is campaign-level strategy. A newer plan supersedes
            # older pending plans instead of forming a FIFO backlog of conflicting advice.
            for existing in sorted(self.guidance_pending.glob("*.json")):
                try:
                    os.replace(existing, self.guidance_superseded / existing.name)
                except OSError:
                    pass
            _atomic_json(
                path,
                {
                    "request_id": request_id,
                    "created_at": utc_now(),
                    "source": source,
                    "guidance_kind": "standing_campaign_strategy",
                    "message": message,
                    "first_iteration": first_iteration,
                    "expires_after_iteration": first_iteration + valid_iterations - 1,
                    "valid_for_iterations": valid_iterations,
                    "delivery_count": 0,
                },
            )
        return path

    def consume_guidance(self, logical_iteration: int) -> dict[str, Any] | None:
        candidates = sorted(self.guidance_pending.glob("*.json"), reverse=True)
        if not candidates:
            return None
        path = candidates[0]
        for stale in candidates[1:]:
            try:
                os.replace(stale, self.guidance_superseded / stale.name)
            except OSError:
                pass
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        # Legacy V1 guidance had no iteration scope and is unsafe to replay after a resume.
        if "first_iteration" not in data or "expires_after_iteration" not in data:
            try:
                os.replace(path, self.guidance_superseded / path.name)
            except OSError:
                pass
            return None
        first = int(data.get("first_iteration") or 0)
        expires = int(data.get("expires_after_iteration") or 0)
        if logical_iteration < first:
            return None
        if logical_iteration > expires:
            try:
                os.replace(path, self.guidance_superseded / path.name)
            except OSError:
                pass
            return None
        data["delivery_count"] = int(data.get("delivery_count") or 0) + 1
        data["last_delivered_iteration"] = logical_iteration
        if logical_iteration < expires:
            _atomic_json(path, data)
            return data
        destination = self.guidance_consumed / path.name
        try:
            os.replace(path, destination)
        except OSError:
            return None
        return data

    def queue_control(self, request: ControlRequest) -> Path:
        path = self.control_pending / f"{time.time_ns()}-{request.request_id}.json"
        _atomic_json(
            path,
            {
                "request_id": request.request_id,
                "action": request.action,
                "expected_run_id": request.expected_run_id,
                "reason": request.reason,
                "evidence_event_ids": list(request.evidence_event_ids),
                "guidance": request.guidance,
                "strategy": request.strategy,
                "created_at": request.created_at,
            },
        )
        return path

    def consume_control(self, expected_run_id: str) -> ControlRequest | None:
        for path in sorted(self.control_pending.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                request = ControlRequest.from_dict(data)
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                destination = self.control_consumed / f"invalid-{path.name}"
                try:
                    os.replace(path, destination)
                except OSError:
                    pass
                continue
            if request.expected_run_id != expected_run_id:
                continue
            destination = self.control_consumed / path.name
            try:
                os.replace(path, destination)
            except OSError:
                continue
            return request
        return None

    def create_activation_dir(self) -> Path:
        existing = sorted(self.activations_dir.glob("activation-*"))
        number = len(existing) + 1
        path = self.activations_dir / f"activation-{number:04d}"
        path.mkdir(parents=True, exist_ok=False)
        return path
