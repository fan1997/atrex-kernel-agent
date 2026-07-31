from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AgentEvent:
    campaign_id: str
    run_id: str
    kind: str
    timestamp: str = field(default_factory=utc_now)
    stream: str = "control"
    raw: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    cursor: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControlRequest:
    request_id: str
    action: str
    expected_run_id: str
    reason: str
    evidence_event_ids: tuple[int, ...] = ()
    guidance: str = ""
    strategy: str = "fresh_context"
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlRequest":
        return cls(
            request_id=str(data.get("request_id") or ""),
            action=str(data.get("action") or ""),
            expected_run_id=str(data.get("expected_run_id") or ""),
            reason=str(data.get("reason") or ""),
            evidence_event_ids=tuple(
                int(value) for value in data.get("evidence_event_ids") or []
            ),
            guidance=str(data.get("guidance") or ""),
            strategy=str(data.get("strategy") or "fresh_context"),
            created_at=str(data.get("created_at") or utc_now()),
        )


@dataclass(frozen=True)
class AttemptResult:
    stdout: str
    stderr: str
    exit_status: int
    timed_out: bool
    interrupted: bool = False
    control_request: ControlRequest | None = None
