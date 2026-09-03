from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
import time
import uuid
from pathlib import Path

from long_horizon.models import VerificationResult, VerificationRun
from long_horizon.protocol import atomic_write_json
from long_horizon.verifier import score_verification_payload, verification_schedule

from .evaluation import (
    PendingVerification,
    load_completed_verification,
    load_pending,
    save_pending,
    utc_now,
    wait_for_terminal_job,
)
from .agent_result import write_agent_result
from .manifest import RepositoryManifest
from .staging import build_abba_stage
from .transport import collect_agate_dev, submit_agate_dev, submit_local_dev


_INFRA_VERIFY_RETRY_DELAYS = (60.0, 300.0, 900.0)
_INFRA_ERROR_SIGNATURES = (
    '"error_class": "infra"',
    "oss_input_download_failed",
    "logs_unavailable",
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
    "RemoteDisconnected",
    "Connection reset",
    "Connection aborted",
    "Max retries exceeded",
    "Name or service not known",
    "Temporary failure in name resolution",
    "502 Bad Gateway",
    "503 Service",
    "504 Gateway",
)


def _infra_verification_retryable(result: VerificationResult) -> bool:
    if result.gate != "ERROR":
        return False
    message = result.error or ""
    return any(signature in message for signature in _INFRA_ERROR_SIGNATURES)


def _evaluation_runtime_root(workspace: Path) -> Path:
    """Keep verifier staging outside the Agent worktree.

    A private production stage contains exact evaluator cases.  Persisting it below
    the episode workspace would make those files ordinary Agent-visible artifacts.
    The stable digest also avoids collisions between same-named worktrees.
    """
    resolved = workspace.resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    return (
        resolved.parent
        / ".repository_horizon_evaluations"
        / f"{resolved.name}-{digest}"
    )


def has_measured_v0(workspace: Path) -> bool:
    """Return whether repository bring-up has produced canonical V0.

    Later non-promotion memories (for example an interrupted episode) must not
    put the campaign back into candidate-only bring-up verification.  V0 is the
    durable phase boundary; subsequent candidates require normal ABBA.
    """
    path = workspace / "memory" / "v0.json"
    try:
        memory = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(memory, dict)
        and (memory.get("quality_gate") or {}).get("result") == "PASS"
    )


def _remove_private_stage_inputs(pending: PendingVerification) -> None:
    runtime = Path(pending.stage) / "runtime"
    for name in ("shapes.json", "metadata.json", "roofline.json"):
        (runtime / name).unlink(missing_ok=True)


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


def _candidate_latency(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("latency_us_geomean")
    if not isinstance(value, (int, float)):
        performance = result.get("performance")
        value = performance.get("latency_us") if isinstance(performance, dict) else None
    if (
        not isinstance(value, (int, float))
        or value <= 0
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _score_candidate_only(
    payload: object,
    *,
    schedule: list[dict[str, Any]],
    repeats: int,
    artifact: str,
) -> VerificationResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return VerificationResult(
            "ERROR", None, None, None, error="unsupported result schema"
        )
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return VerificationResult(
            "ERROR", None, None, None, error="runs must be a list of objects"
        )
    runs = [
        VerificationRun(
            revision=str(row.get("revision", "")),
            repeat=int(row.get("repeat", -1)),
            exit_code=int(row.get("exit_code", -1)),
            result=(
                dict(row["result"]) if isinstance(row.get("result"), dict) else None
            ),
            stdout_tail=str(row.get("stdout_tail", "")),
            stderr_tail=str(row.get("stderr_tail", "")),
        )
        for row in rows
    ]
    if payload.get("error"):
        return VerificationResult(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error=str(payload["error"]),
            artifact=artifact,
        )
    actual = [{"revision": run.revision, "repeat": run.repeat} for run in runs]
    if actual != schedule:
        return VerificationResult(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error="remote verifier did not execute the exact candidate-only schedule",
            artifact=artifact,
        )
    values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := _candidate_latency(run.result)) is not None
    ]
    if len(values) != repeats:
        return VerificationResult(
            "FAIL",
            None,
            None,
            None,
            runs=runs,
            error="not every authoritative candidate run passed",
            artifact=artifact,
        )
    candidate = (
        values[0]
        if len(values) == 1
        else math.exp(sum(math.log(value) for value in values) / len(values))
    )
    return VerificationResult(
        "PASS", candidate, candidate, 0.0, runs=runs, artifact=artifact
    )


def _require_complete_shape_coverage(
    verification: VerificationResult, expected_shape_ids: tuple[str, ...]
) -> VerificationResult:
    """Fail a nominal pass unless every scheduled run covers every evaluator shape."""
    if not verification.passed or not expected_shape_ids:
        return verification
    expected = set(expected_shape_ids)
    incomplete: list[str] = []
    for run in verification.runs:
        by_shape = (
            run.result.get("latency_us_by_shape")
            if isinstance(run.result, dict)
            else None
        )
        measured = (
            set(str(value) for value in by_shape)
            if isinstance(by_shape, dict)
            else set()
        )
        if measured != expected:
            incomplete.append(
                f"{run.revision}[{run.repeat}]={len(measured)}/{len(expected)}"
            )
    if not incomplete:
        return verification
    return VerificationResult(
        "FAIL",
        verification.candidate_latency_us,
        verification.incumbent_latency_us,
        verification.improvement_pct,
        runs=verification.runs,
        error="incomplete evaluator shape coverage: " + ", ".join(incomplete),
        artifact=verification.artifact,
    )


def collect_pending_verification(
    pending_or_path: PendingVerification | Path,
) -> VerificationResult:
    pending = (
        pending_or_path
        if isinstance(pending_or_path, PendingVerification)
        else load_pending(pending_or_path)
    )
    completed = load_completed_verification(pending)
    if completed is not None:
        _remove_private_stage_inputs(pending)
        return completed
    evidence_path = pending.directory / "transport_result.json"
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
            "transport_response": snapshot.response,
            "stdout_tail": result.stdout[-8000:],
            "stderr_tail": result.stderr[-8000:],
        }
        atomic_write_json(evidence_path, evidence)
        verification = (
            _score_candidate_only(
                result.payload,
                schedule=list(pending.schedule),
                repeats=pending.repeats,
                artifact=str(evidence_path),
            )
            if pending.candidate_only
            else score_verification_payload(
                result.payload,
                schedule=list(pending.schedule),
                repeats=pending.repeats,
                min_improvement_pct=pending.min_improvement_pct,
                artifact=str(evidence_path),
            )
        )
        verification = _require_complete_shape_coverage(
            verification, pending.expected_shape_ids
        )
    except Exception as exc:
        verification = VerificationResult(
            "ERROR",
            None,
            None,
            None,
            error=(
                "repository Agate verification failed: " f"{type(exc).__name__}: {exc}"
            ),
            artifact=str(pending.directory),
        )
    _remove_private_stage_inputs(pending)
    atomic_write_json(pending.result_path, verification.as_dict())
    write_agent_result(
        pending.directory,
        verification,
        evaluation_id=pending.evaluation_id,
        backend=pending.backend,
        request_digest=pending.request_digest,
        max_bytes=pending.agent_result_max_bytes,
    )
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
        backend: str = "agate",
        wait_mode: str = "inline",
        agent_result_max_bytes: int = 16 * 1024,
        private_reference_dir: Path | None = None,
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
        self.backend = backend
        self.wait_mode = wait_mode
        self.agent_result_max_bytes = agent_result_max_bytes
        self.private_reference_dir = (
            private_reference_dir.resolve()
            if private_reference_dir is not None
            else None
        )

    def submit(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
        working_snapshot: bool = False,
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
        runtime_root = _evaluation_runtime_root(workspace)
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
            working_snapshot=working_snapshot,
            private_reference_dir=self.private_reference_dir,
        )
        shapes_path = (self.private_reference_dir or workspace) / "shapes.json"
        expected_shape_ids: tuple[str, ...] = ()
        if shapes_path.is_file():
            shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
            if not isinstance(shapes, dict) or not shapes:
                raise ValueError("evaluator shapes.json must be a non-empty object")
            expected_shape_ids = tuple(sorted(str(value) for value in shapes))
        if metadata["packed_bytes"] > self.max_packed_bytes:
            raise ValueError(
                f"repository staging payload is {metadata['packed_bytes']} bytes; "
                f"Agate working-dir limit is {self.max_packed_bytes}"
            )
        for existing_path in runtime_root.glob("*/pending.json"):
            existing = load_pending(existing_path)
            if (
                existing.request_digest == metadata["request_digest"]
                and not existing.result_path.is_file()
            ):
                shutil.rmtree(stage.parent)
                return existing
        submitted = (
            submit_local_dev(stage, job_timeout=self.timeout)
            if self.backend == "local"
            else submit_agate_dev(
                stage,
                hardware=self.hardware,
                profile=self.profile,
                url=self.url,
                job_timeout=self.timeout,
            )
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
            backend=self.backend,
            wait_mode=self.wait_mode,
            request_digest=str(metadata["request_digest"]),
            snapshot_kind="worktree" if working_snapshot else "commit",
            agent_result_max_bytes=self.agent_result_max_bytes,
            expected_shape_ids=expected_shape_ids,
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

    def collect(
        self, pending_or_path: PendingVerification | Path
    ) -> VerificationResult:
        return collect_pending_verification(pending_or_path)

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        attempt = 0
        while True:
            try:
                pending = self.submit(
                    workspace,
                    base_commit=base_commit,
                    candidate_commit=candidate_commit,
                    changed_paths=changed_paths,
                )
                result = self.collect(pending)
            except Exception as exc:
                result = VerificationResult(
                    "ERROR",
                    None,
                    None,
                    None,
                    error=(
                        "repository Agate verification failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            if (
                self.backend != "agate"
                or not _infra_verification_retryable(result)
                or attempt >= len(_INFRA_VERIFY_RETRY_DELAYS)
            ):
                return result
            delay = _INFRA_VERIFY_RETRY_DELAYS[attempt]
            print(
                "[repository-horizon] Agate infra failure; resubmitting "
                f"authoritative verification in {delay:.0f}s "
                f"(attempt {attempt + 1}/{len(_INFRA_VERIFY_RETRY_DELAYS)})",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            attempt += 1


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
        return self.normal if has_measured_v0(workspace) else self.bringup

    def submit(self, workspace: Path, **kwargs) -> PendingVerification:
        return self._validator(workspace).submit(workspace, **kwargs)

    def collect(
        self, pending_or_path: PendingVerification | Path
    ) -> VerificationResult:
        return collect_pending_verification(pending_or_path)

    def verify(self, workspace: Path, **kwargs) -> VerificationResult:
        return self._validator(workspace).verify(workspace, **kwargs)
