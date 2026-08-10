from __future__ import annotations

from dataclasses import asdict, dataclass

from long_horizon import main_adapter


WAIT_MODES = frozenset({"auto", "inline", "suspend"})
SUSPEND_ENFORCEMENTS = frozenset({"graceful", "enforced"})
EVALUATION_BACKENDS = frozenset({"agate", "local"})


@dataclass(frozen=True)
class EvaluationPolicy:
    """User-visible policy for GPU execution and model suspension."""

    backend: str = "agate"
    wait_mode: str = "auto"
    suspend_enforcement: str = "enforced"
    suspend_grace_seconds: float = 5.0
    wait_timeout: int = 14_400
    agent_result_max_bytes: int = 16 * 1024
    resume_prompt_max_bytes: int = 2 * 1024

    def __post_init__(self) -> None:
        if self.backend not in EVALUATION_BACKENDS:
            raise ValueError(f"unsupported evaluation backend: {self.backend}")
        if self.wait_mode not in WAIT_MODES:
            raise ValueError(f"unsupported evaluation wait mode: {self.wait_mode}")
        if self.suspend_enforcement not in SUSPEND_ENFORCEMENTS:
            raise ValueError(
                f"unsupported suspend enforcement: {self.suspend_enforcement}"
            )
        if self.suspend_grace_seconds < 0:
            raise ValueError("suspend grace must be non-negative")
        if self.wait_timeout <= 0:
            raise ValueError("evaluation wait timeout must be positive")
        if self.agent_result_max_bytes < 1024:
            raise ValueError("agent result limit must be at least 1024 bytes")
        if self.resume_prompt_max_bytes < 512:
            raise ValueError("resume prompt limit must be at least 512 bytes")

    def resolved_wait_mode(self, agent_cli: str, *, endpoint_is_local: bool) -> str:
        if self.wait_mode != "auto":
            resolved = self.wait_mode
        elif self.backend == "local" or endpoint_is_local:
            resolved = "inline"
        elif main_adapter.supports_same_session_resume(agent_cli):
            resolved = "suspend"
        else:
            resolved = "inline"
        if resolved == "suspend" and not main_adapter.supports_same_session_resume(
            agent_cli
        ):
            raise ValueError(
                f"{agent_cli} cannot resume the same native session; use "
                "--evaluation-wait-mode inline"
            )
        return resolved

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def endpoint_is_local(url: str, hardware: str) -> bool:
    lowered = url.lower().strip()
    return hardware.lower() == "local" or lowered.startswith(
        ("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")
    )
