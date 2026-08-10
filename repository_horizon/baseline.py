from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from long_horizon.git_episode import git_head
from long_horizon.protocol import atomic_write_json

from .config import endpoint_is_local
from .manifest import RepositoryManifest
from .seed import seed_workspace
from .verifier import RepositoryABBAValidator


class RepositoryBaselineManager:
    def __init__(
        self,
        manifest: RepositoryManifest,
        source_checkout: Path,
        support_wheels: dict[str, Path] | None = None,
    ):
        self.manifest = manifest
        self.source_checkout = source_checkout.resolve()
        self.support_wheels = support_wheels or {}

    def prepare(self, campaign: Any) -> None:
        if campaign.framework != "CuteDSL":
            raise RuntimeError("repository horizon requires --framework CuteDSL")
        if campaign.framework_baseline != "never":
            raise RuntimeError("repository horizon requires --framework-baseline never")
        seed_workspace(
            campaign,
            self.manifest,
            self.source_checkout,
            support_wheels=self.support_wheels,
        )
        self._ensure_measured_v0(campaign)

    def _ensure_measured_v0(self, campaign: Any) -> None:
        v0_path = campaign.workspace / "memory" / "v0.json"
        bringup_enabled = self.manifest.bringup.mode == "auto"
        memory_path = (
            campaign.workspace / "memory" / "r0.json"
            if bringup_enabled and not v0_path.is_file()
            else v0_path
        )
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        if v0_path.is_file() and (memory.get("correctness") or {}).get("status") == "PASS":
            return
        if (
            bringup_enabled
            and not v0_path.is_file()
            and (memory.get("probe") or {}).get("classification") == "WORKLOAD_FAIL"
        ):
            return

        revision = git_head(campaign.workspace)
        run_count = self.manifest.bringup.probe_repeats if bringup_enabled else 2
        baseline_run_timeout = max(1, (campaign.sandbox_timeout - 30) // run_count)
        policy = getattr(campaign, "repository_evaluation_policy", None)
        verifier = RepositoryABBAValidator(
            manifest=self.manifest,
            atrex_bench_root=Path(campaign.atrex_bench_root),
            hardware=campaign.sandbox_hardware,
            profile=campaign.sandbox_profile,
            url=campaign.sandbox_url,
            timeout=campaign.sandbox_timeout,
            repeats=(self.manifest.bringup.probe_repeats if bringup_enabled else 1),
            per_run_timeout=baseline_run_timeout,
            min_improvement_pct=-100.0,
            candidate_only=bringup_enabled,
            backend=getattr(policy, "backend", "agate"),
            wait_mode=(
                policy.resolved_wait_mode(
                    campaign.agent_cli,
                    endpoint_is_local=endpoint_is_local(
                        campaign.sandbox_url, campaign.sandbox_hardware
                    ),
                )
                if policy is not None
                else "suspend"
            ),
            wait_timeout=getattr(policy, "wait_timeout", 14_400),
            agent_result_max_bytes=getattr(policy, "agent_result_max_bytes", 16 * 1024),
        )
        verification = verifier.verify(
            campaign.workspace,
            base_commit=revision,
            candidate_commit=revision,
            changed_paths=[],
        )
        if not verification.passed:
            if verification.gate == "ERROR":
                raise RuntimeError(
                    "repository R0 probe had no authoritative workload outcome: "
                    f"{verification.error}; artifact={verification.artifact}"
                )
            if not bringup_enabled:
                raise RuntimeError(
                    "repository V0 failed remote validation: "
                    f"{verification.error}; artifact={verification.artifact}"
                )
            memory["correctness"] = {"status": "FAIL"}
            memory["quality_gate"] = {
                "result": "BRINGUP_REQUIRED",
                "failure_reason": verification.error,
            }
            memory["probe"] = {
                "gate": verification.gate,
                "artifact": verification.artifact,
                "classification": "WORKLOAD_FAIL",
            }
            atomic_write_json(memory_path, memory)
            self._amend(campaign.workspace, memory_path)
            print(
                "[repository-horizon] R0 does not satisfy the workload; "
                "entering correctness-only bring-up",
                flush=True,
            )
            return

        candidate_runs = [
            run
            for run in verification.runs
            if run.revision == "candidate" and isinstance(run.result, dict)
        ]
        representative = candidate_runs[-1].result if candidate_runs else {}
        v0_memory = dict(memory)
        v0_memory["version"] = "v0"
        v0_memory["performance"] = {
            "latency_us": verification.candidate_latency_us,
            "latency_us_geomean": verification.candidate_latency_us,
            "latency_us_arith_mean": representative.get("latency_us_arith_mean"),
            "latency_us_by_shape": representative.get("latency_us_by_shape", {}),
            "timer": "atrex-bench CUDA event",
            "artifact": verification.artifact,
        }
        v0_memory["correctness"] = {
            "status": "PASS",
            "max_abs_err": representative.get("max_abs_err"),
            "max_rel_err": representative.get("max_rel_err"),
        }
        v0_memory["quality_gate"] = {"result": "PASS", "failure_reason": None}
        if bringup_enabled:
            v0_memory["optimization"] = {
                "action_category": "repository_v0_fast_path",
                "action_description": (
                    "locked official source satisfies the workload without source changes"
                ),
            }
            atomic_write_json(v0_path, v0_memory)
        else:
            atomic_write_json(memory_path, v0_memory)
        self._amend(campaign.workspace, v0_path)

    @staticmethod
    def _amend(workspace: Path, memory_path: Path) -> None:
        subprocess.run(
            ["git", "add", str(memory_path.relative_to(workspace))],
            cwd=str(workspace),
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=atrex-repository-horizon",
                "-c",
                "user.email=atrex-repository-horizon@local",
                "commit",
                "--amend",
                "--no-edit",
            ],
            cwd=str(workspace),
            check=True,
            capture_output=True,
        )
