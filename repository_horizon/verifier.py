from __future__ import annotations

import json
import uuid
from pathlib import Path

from long_horizon.models import VerificationResult
from long_horizon.verifier import score_verification_payload, verification_schedule

from .manifest import RepositoryManifest
from .staging import build_abba_stage
from .transport import run_agate_dev


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

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        schedule = verification_schedule(self.repeats)
        if self.per_run_timeout * len(schedule) + 30 > self.timeout:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error="repository ABBA schedule cannot fit in one Agate allocation timeout",
            )
        runtime_root = workspace / ".repository_horizon_runtime" / "verifications"
        runtime_root.mkdir(parents=True, exist_ok=True)
        verification_id = uuid.uuid4().hex
        stage = runtime_root / verification_id / "staging"
        try:
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
                return VerificationResult(
                    "ERROR",
                    None,
                    None,
                    None,
                    error=(
                        f"repository staging payload is {metadata['packed_bytes']} bytes; "
                        f"Agate working-dir limit is {self.max_packed_bytes}"
                    ),
                    artifact=str(stage),
                )
            result = run_agate_dev(
                stage,
                hardware=self.hardware,
                profile=self.profile,
                url=self.url,
                job_timeout=self.timeout,
                wait_timeout=self.wait_timeout,
            )
            evidence = {
                "job_id": result.job_id,
                "command": [
                    item
                    for item in result.command
                    if item not in {"--token", "--ak", "--sk"}
                ],
                "packed_bytes": metadata["packed_bytes"],
                "payload": result.payload,
                "stdout_tail": result.stdout[-8000:],
                "stderr_tail": result.stderr[-8000:],
            }
            evidence_path = stage.parent / "agate_result.json"
            evidence_path.write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
            return score_verification_payload(
                result.payload,
                schedule=schedule,
                repeats=self.repeats,
                min_improvement_pct=self.min_improvement_pct,
                artifact=str(evidence_path),
            )
        except Exception as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"repository Agate verification failed: {type(exc).__name__}: {exc}",
                artifact=str(stage.parent),
            )
