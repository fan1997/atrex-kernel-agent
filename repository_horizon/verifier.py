from __future__ import annotations

import json
import uuid
from pathlib import Path

from long_horizon.models import VerificationResult
from long_horizon.protocol import atomic_write_json
from long_horizon.verifier import score_verification_payload, verification_schedule

from .evaluation import (
    PendingVerification,
    load_pending,
    save_pending,
    utc_now,
    wait_for_terminal_job,
)
from .manifest import RepositoryManifest
from .staging import build_abba_stage
from .transport import collect_agate_dev, submit_agate_dev


def _safe_command(command: tuple[str, ...]) -> list[str]:
    hidden = {"--token", "--ak", "--sk"}
    result: list[str] = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        if item in hidden:
            skip = True
            continue
        result.append(item)
    return result


def collect_pending_verification(
    pending_or_path: PendingVerification | Path,
) -> VerificationResult:
    pending = (
        pending_or_path
        if isinstance(pending_or_path, PendingVerification)
        else load_pending(pending_or_path)
    )
    evidence_path = pending.directory / "agate_result.json"
    try:
        snapshot = wait_for_terminal_job(pending)
        result = collect_agate_dev(snapshot)
        evidence = {
            "schema_version": 1,
            "evaluation_id": pending.evaluation_id,
            "job_id": result.job_id,
            "submitted_at": pending.submitted_at,
            "collected_at": utc_now(),
            "submit_command": _safe_command(pending.submit_command),
            "collect_command": _safe_command(result.command),
            "packed_bytes": pending.packed_bytes,
            "base_commit": pending.base_commit,
            "candidate_commit": pending.candidate_commit,
            "changed_paths": list(pending.changed_paths),
            "payload": result.payload,
            "gateway_response": snapshot.response,
            "stdout_tail": result.stdout[-8000:],
            "stderr_tail": result.stderr[-8000:],
        }
        atomic_write_json(evidence_path, evidence)
        verification = score_verification_payload(
            result.payload,
            schedule=list(pending.schedule),
            repeats=pending.repeats,
            min_improvement_pct=pending.min_improvement_pct,
            artifact=str(evidence_path),
            candidate_only=pending.candidate_only,
        )
    except Exception as exc:
        verification = VerificationResult(
            "ERROR",
            None,
            None,
            None,
            error=(
                "repository Agate verification failed: "
                f"{type(exc).__name__}: {exc}"
            ),
            artifact=str(pending.directory),
        )
    atomic_write_json(pending.result_path, verification.as_dict())
    return verification


class RepositoryABBAValidator:
    def __init__(
        self,
        *,
        manifest: RepositoryManifest,
        atrex_bench_root: Path,
        hardware: str,
        profile: str = "",
        url: str = "",
        timeout: int = 600,
        repeats: int | None = None,
        per_run_timeout: int | None = None,
        min_improvement_pct: float | None = None,
        wait_timeout: int = 14_400,
        max_packed_bytes: int = 900_000,
        candidate_only: bool = False,
    ):
        self.manifest = manifest
        self.atrex_bench_root = atrex_bench_root
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.timeout = timeout
        self.repeats = repeats or manifest.measurement.repeats
        self.per_run_timeout = per_run_timeout or manifest.measurement.per_run_timeout
        self.min_improvement_pct = (
            manifest.measurement.min_improvement_pct
            if min_improvement_pct is None
            else min_improvement_pct
        )
        self.wait_timeout = wait_timeout
        self.max_packed_bytes = max_packed_bytes
        self.candidate_only = candidate_only

    def submit(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> PendingVerification:
        schedule = (
            [
                {"revision": "candidate", "repeat": repeat}
                for repeat in range(self.repeats)
            ]
            if self.candidate_only
            else verification_schedule(self.repeats)
        )
        if self.per_run_timeout * len(schedule) + 30 > self.timeout:
            raise ValueError(
                "repository verification schedule cannot fit in one Agate allocation timeout"
            )
        runtime_root = workspace / ".repository_horizon_runtime" / "evaluations"
        runtime_root.mkdir(parents=True, exist_ok=True)
        verification_id = uuid.uuid4().hex
        stage = runtime_root / verification_id / "staging"
        metadata = build_abba_stage(
            workspace,
            base_commit=base_commit,
            candidate_commit=candidate_commit,
            changed_paths=changed_paths,
            manifest=self.manifest,
            atrex_bench_root=self.atrex_bench_root,
            destination=stage,
            schedule=schedule,
            per_run_timeout=self.per_run_timeout,
        )
        if metadata["packed_bytes"] > self.max_packed_bytes:
            raise ValueError(
                f"repository staging payload is {metadata['packed_bytes']} bytes; "
                f"Agate working-dir limit is {self.max_packed_bytes}"
            )
        submitted = submit_agate_dev(
            stage,
            hardware=self.hardware,
            profile=self.profile,
            url=self.url,
            job_timeout=self.timeout,
        )
        pending = PendingVerification(
            schema_version=1,
            evaluation_id=verification_id,
            job_id=submitted.job_id,
            hardware=self.hardware,
            profile=self.profile,
            url=self.url,
            job_timeout=self.timeout,
            wait_timeout=self.wait_timeout,
            stage=str(stage),
            base_commit=base_commit,
            candidate_commit=candidate_commit,
            changed_paths=tuple(changed_paths),
            schedule=tuple(schedule),
            repeats=self.repeats,
            min_improvement_pct=self.min_improvement_pct,
            candidate_only=self.candidate_only,
            packed_bytes=int(metadata["packed_bytes"]),
            submit_command=submitted.command,
            submitted_at=utc_now(),
        )
        save_pending(pending)
        atomic_write_json(
            pending.directory / "submission.json",
            {
                "schema_version": 1,
                "evaluation_id": verification_id,
                "job_id": submitted.job_id,
                "submitted_at": pending.submitted_at,
                "command": _safe_command(submitted.command),
                "stdout": submitted.stdout,
                "stderr": submitted.stderr,
            },
        )
        return pending

    def collect(self, pending_or_path: PendingVerification | Path) -> VerificationResult:
        return collect_pending_verification(pending_or_path)

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        try:
            pending = self.submit(
                workspace,
                base_commit=base_commit,
                candidate_commit=candidate_commit,
                changed_paths=changed_paths,
            )
            return self.collect(pending)
        except Exception as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=(
                    "repository Agate verification failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )


class RepositoryPhaseValidator:
    """Use correctness-only candidate validation until canonical V0 exists."""

    def __init__(
        self,
        normal: RepositoryABBAValidator,
        bringup: RepositoryABBAValidator,
    ):
        self.normal = normal
        self.bringup = bringup

    def _validator(self, workspace: Path) -> RepositoryABBAValidator:
        return (
            self.normal
            if (workspace / "memory" / "v0.json").is_file()
            else self.bringup
        )

    def submit(self, workspace: Path, **kwargs) -> PendingVerification:
        return self._validator(workspace).submit(workspace, **kwargs)

    def collect(self, pending_or_path: PendingVerification | Path) -> VerificationResult:
        return collect_pending_verification(pending_or_path)

    def verify(self, workspace: Path, **kwargs) -> VerificationResult:
        return self._validator(workspace).verify(workspace, **kwargs)
