from __future__ import annotations

from dataclasses import asdict, dataclass

WAIT_MODES = frozenset({"inline"})
EVALUATION_BACKENDS = frozenset({"agate", "local"})


@dataclass(frozen=True)
class EvaluationPolicy:
    """Repository evaluation policy compatible with main's session runner."""

    backend: str = "agate"
    wait_mode: str = "inline"
    wait_timeout: int = 14_400
    agent_result_max_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if self.backend not in EVALUATION_BACKENDS:
            raise ValueError(f"unsupported evaluation backend: {self.backend}")
        if self.wait_mode not in WAIT_MODES:
            raise ValueError(f"unsupported evaluation wait mode: {self.wait_mode}")
        if self.wait_timeout <= 0:
            raise ValueError("evaluation wait timeout must be positive")
        if self.agent_result_max_bytes < 1024:
            raise ValueError("agent result limit must be at least 1024 bytes")

    def resolved_wait_mode(self, agent_cli: str, *, endpoint_is_local: bool) -> str:
        del agent_cli, endpoint_is_local
        return "inline"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def endpoint_is_local(url: str, hardware: str) -> bool:
    lowered = url.lower().strip()
    return hardware.lower() == "local" or lowered.startswith(
        (
            "http://127.0.0.1",
            "http://localhost",
            "https://127.0.0.1",
            "https://localhost",
        )
    )
