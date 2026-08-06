from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from . import main_adapter

EVIDENCE_PREFIXES = ("plans/", "profiles/")


class CandidateContract(Protocol):
    """Defines which committed files constitute a candidate.

    The protocol is deliberately independent of the verifier and transport.  A
    repository overlay can therefore admit source files without weakening the
    default immutable-evaluator boundary.
    """

    def validate_changed_paths(self, paths: list[str]) -> str: ...

    def verification_paths(self, paths: list[str]) -> list[str]: ...

    def workspace_violations(self, campaign: Any, workspace: Path) -> list[str]: ...


class KernelCandidateContract:
    """The historical long-horizon single-file contract."""

    def validate_changed_paths(self, paths: list[str]) -> str:
        from .git_episode import protected_violation

        violation = protected_violation(paths)
        if violation:
            return violation
        if "kernel.py" not in paths:
            return "candidate must change kernel.py relative to incumbent"
        return ""

    def verification_paths(self, paths: list[str]) -> list[str]:
        return [path for path in paths if not path.startswith(EVIDENCE_PREFIXES)]

    def workspace_violations(self, campaign: Any, workspace: Path) -> list[str]:
        return main_adapter.candidate_policy_violations(campaign, workspace)


class LongHorizonIntegration(Protocol):
    """Optional lifecycle overlay for source-assisted optimization campaigns."""

    def prepare_campaign(self, campaign: Any) -> None: ...

    def link_episode_runtime(self, campaign: Any, workspace: Path) -> None: ...

    def prompt_fields(
        self, campaign: Any, workspace: Path, version: int
    ) -> dict[str, object]: ...

    def candidate_contract(self) -> CandidateContract: ...

    def make_verifier(
        self, campaign: Any, options: Any, default_verifier: Any
    ) -> Any: ...

    def memory_metadata(
        self, campaign: Any, verification: Any
    ) -> dict[str, object]: ...

    def finish_campaign(self, campaign: Any, reason: str) -> bool: ...


class DefaultLongHorizonIntegration:
    """Compatibility implementation: every operation delegates to current main."""

    def __init__(self) -> None:
        self._contract = KernelCandidateContract()

    def prepare_campaign(self, campaign: Any) -> None:
        main_adapter.prepare_campaign(campaign)

    def link_episode_runtime(self, campaign: Any, workspace: Path) -> None:
        main_adapter.link_episode_runtime(campaign, workspace)

    def prompt_fields(
        self, campaign: Any, workspace: Path, version: int
    ) -> dict[str, object]:
        return {}

    def candidate_contract(self) -> CandidateContract:
        return self._contract

    def make_verifier(self, campaign: Any, options: Any, default_verifier: Any) -> Any:
        return default_verifier

    def memory_metadata(self, campaign: Any, verification: Any) -> dict[str, object]:
        return {}

    def finish_campaign(self, campaign: Any, reason: str) -> bool:
        return False


def normalize_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() in {"", "."}
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()
