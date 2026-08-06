from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from long_horizon import main_adapter
from long_horizon.git_episode import git_head
from long_horizon.protocol import atomic_write_json

from .candidate import RepositoryCandidateContract
from .corpus import CORPUS_RELATIVE, read_catalog
from .manifest import RepositoryManifest
from .policy import install_repository_policy
from .seed import seed_workspace
from .verifier import RepositoryABBAValidator

MODULE_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK = Path(__file__).resolve().parent / "prompts" / "episode_playbook.md"


class RepositoryPhaseValidator:
    """Select correctness-only bring-up until the first canonical V0 exists."""

    def __init__(self, normal: RepositoryABBAValidator, bringup: RepositoryABBAValidator):
        self.normal = normal
        self.bringup = bringup

    def verify(self, workspace: Path, **kwargs):
        validator = (
            self.normal
            if (workspace / "memory" / "v0.json").is_file()
            else self.bringup
        )
        return validator.verify(workspace, **kwargs)


class RepositoryIntegration:
    def __init__(
        self,
        manifest: RepositoryManifest,
        source_checkout: Path,
        support_wheels: dict[str, Path] | None = None,
    ):
        self.manifest = manifest
        self.source_checkout = source_checkout.resolve()
        self.support_wheels = support_wheels or {}
        self._contract = RepositoryCandidateContract(manifest)

    def prepare_campaign(self, campaign: Any) -> None:
        if campaign.framework != "CuteDSL":
            raise RuntimeError("repository horizon v1 requires --framework CuteDSL")
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
        revision = git_head(campaign.workspace)
        # V0 compares the exact same commit twice, so give each cold library
        # import/JIT/evaluator run the largest budget that still fits in the
        # gateway allocation. Candidate verification retains four-run A-B-B-A.
        run_count = (
            self.manifest.bringup.probe_repeats if bringup_enabled else 2
        )
        baseline_run_timeout = max(1, (campaign.sandbox_timeout - 30) // run_count)
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
        )
        verification = verifier.verify(
            campaign.workspace,
            base_commit=revision,
            candidate_commit=revision,
            changed_paths=[],
        )
        if not verification.passed:
            if bringup_enabled:
                memory["correctness"] = {"status": "FAIL"}
                memory["quality_gate"] = {
                    "result": "BRINGUP_REQUIRED",
                    "failure_reason": verification.error,
                }
                memory["probe"] = {
                    "gate": verification.gate,
                    "artifact": verification.artifact,
                }
                atomic_write_json(memory_path, memory)
                subprocess.run(
                    ["git", "add", str(memory_path.relative_to(campaign.workspace))],
                    cwd=str(campaign.workspace),
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
                    cwd=str(campaign.workspace),
                    check=True,
                    capture_output=True,
                )
                print(
                    "[repository-horizon] R0 does not satisfy the workload; "
                    "entering correctness-only bring-up",
                    flush=True,
                )
                return
            raise RuntimeError(
                "repository V0 failed remote correctness/performance validation: "
                f"{verification.error}; artifact={verification.artifact}"
            )
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
                "action_description": "locked official source satisfies the workload without source changes",
            }
            atomic_write_json(v0_path, v0_memory)
        else:
            atomic_write_json(memory_path, v0_memory)
        subprocess.run(
            ["git", "add", "memory/v0.json"], cwd=str(campaign.workspace), check=True
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
            cwd=str(campaign.workspace),
            check=True,
            capture_output=True,
        )

    def link_episode_runtime(self, campaign: Any, workspace: Path) -> None:
        main_adapter.link_episode_runtime(campaign, workspace)
        install_repository_policy(workspace, self.manifest)
        source = campaign.workspace / CORPUS_RELATIVE
        if source.is_dir():
            target = workspace / CORPUS_RELATIVE
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink() or target.exists():
                if target.resolve() != source.resolve():
                    raise RuntimeError("episode source corpus points at an unexpected path")
            else:
                target.symlink_to(source, target_is_directory=True)

    def prompt_fields(
        self, campaign: Any, workspace: Path, version: int
    ) -> dict[str, object]:
        command = (
            f"PYTHONPATH={MODULE_ROOT} python -m repository_horizon.dev_eval "
            f"--workspace {workspace} --hardware {campaign.sandbox_hardware}"
        )
        if campaign.sandbox_profile:
            command += f" --profile {campaign.sandbox_profile}"
        if campaign.sandbox_url:
            command += f" --url {campaign.sandbox_url}"
        playbook = PLAYBOOK.read_text(encoding="utf-8")
        values = {
            "SOURCE_NAME": self.manifest.source_name,
            "SOURCE_REVISION": self.manifest.revision,
            "EDITABLE_ROOTS": ", ".join(
                f"`{x}`" for x in self.manifest.editable_workspace_roots
            ),
            "DEV_EVAL_COMMAND": command,
            "CAMPAIGN_PHASE": (
                "optimization"
                if (workspace / "memory" / "v0.json").is_file()
                else "bring-up"
            ),
            "SOURCE_CORPUS": (
                CORPUS_RELATIVE if read_catalog(workspace) is not None else "unavailable"
            ),
        }
        for key, value in values.items():
            playbook = playbook.replace("{{" + key + "}}", str(value))
        return {
            "INTEGRATION_PLAYBOOK": playbook,
            "MAIN_ITERATION_PLAYBOOK": (
                "The repository playbook above replaces the single-file `kernel.py` edit and "
                "ordinary sandbox commands. Retain the current-main principles of evidence-first "
                "profiling, full correctness, reproducible timing, and clean committed candidates."
            ),
            "MODE_POLICY": (
                "## Source-assisted production policy\n\nThe declared source snapshot is an "
                "intentional dependency. Only manifest-declared editable roots may change; the "
                "adapter, evaluator, lock, and workload remain immutable."
            ),
            "EVALUATOR": (
                "## Evaluation route: repository-staged Atrex-Bench\n\nThe official Atrex-Bench "
                "runtime is staged with the source snapshot and executed remotely."
            ),
            "SANDBOX": (
                "## GPU boundary\n\nAll GPU imports, compilation, correctness, timing, and profiling "
                "must use the repository_horizon Agate working-directory command from the playbook."
            ),
        }

    def candidate_contract(self):
        return self._contract

    def make_verifier(self, campaign: Any, options: Any, default_verifier: Any):
        normal = RepositoryABBAValidator(
            manifest=self.manifest,
            atrex_bench_root=Path(campaign.atrex_bench_root),
            hardware=campaign.sandbox_hardware,
            profile=campaign.sandbox_profile,
            url=campaign.sandbox_url,
            timeout=campaign.sandbox_timeout,
        )
        run_timeout = max(
            1,
            (campaign.sandbox_timeout - 30) // self.manifest.bringup.probe_repeats,
        )
        bringup = RepositoryABBAValidator(
            manifest=self.manifest,
            atrex_bench_root=Path(campaign.atrex_bench_root),
            hardware=campaign.sandbox_hardware,
            profile=campaign.sandbox_profile,
            url=campaign.sandbox_url,
            timeout=campaign.sandbox_timeout,
            repeats=self.manifest.bringup.probe_repeats,
            per_run_timeout=run_timeout,
            min_improvement_pct=-100.0,
            candidate_only=True,
        )
        return RepositoryPhaseValidator(normal, bringup)

    def memory_metadata(self, campaign: Any, verification: Any) -> dict[str, object]:
        return {
            "repository_source": {
                "name": self.manifest.source_name,
                "revision": self.manifest.revision,
                "editable_roots": list(self.manifest.editable_workspace_roots),
                "measurement": {
                    "timer": "atrex-bench CUDA event",
                    "schedule": "A-B-B-A",
                    "warmup": self.manifest.measurement.warmup,
                    "timed_runs": self.manifest.measurement.timed_runs,
                },
            },
            "repository_phase": (
                "bring-up" if verification.incumbent_latency_us is None else "optimization"
            ),
        }

    def finish_campaign(self, campaign: Any, reason: str) -> bool:
        export_dir = campaign.workspace / ".repository_horizon_runtime" / "export"
        export_dir.mkdir(parents=True, exist_ok=True)
        root_commit = subprocess.run(
            ["git", "rev-list", "--max-parents=0", "HEAD"],
            cwd=str(campaign.workspace),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0]
        v0_history = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--format=%H",
                "--reverse",
                "--",
                "memory/v0.json",
            ],
            cwd=str(campaign.workspace),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        v0_commit = v0_history[0] if v0_history else None
        export_paths = [
            "kernel.py",
            "source_manifest.json",
            "source.lock.json",
            self.manifest.vendor_root,
        ]
        if (campaign.workspace / "source_corpus.json").is_file():
            export_paths.append("source_corpus.json")
        if (campaign.workspace / "vendor_support").is_dir():
            export_paths.append("vendor_support")
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar.gz",
                f"--output={export_dir / 'repository_candidate.tar.gz'}",
                "HEAD",
                "--",
                *export_paths,
            ],
            cwd=str(campaign.workspace),
            check=True,
        )
        patch = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                root_commit,
                "HEAD",
                "--",
                self.manifest.vendor_root,
            ],
            cwd=str(campaign.workspace),
            check=True,
            capture_output=True,
        ).stdout
        (export_dir / "source_changes.patch").write_bytes(patch)
        atomic_write_json(
            export_dir / "export.json",
            {
                "schema_version": 1,
                "reason": reason,
                "head": git_head(campaign.workspace),
                "root_commit": root_commit,
                "r0_commit": (
                    root_commit if self.manifest.bringup.mode == "auto" else None
                ),
                "v0_commit": v0_commit,
                "source_name": self.manifest.source_name,
                "source_revision": self.manifest.revision,
                "archive": "repository_candidate.tar.gz",
                "patch": "source_changes.patch",
            },
        )
        print(f"\n[repository-horizon] STOP — {reason}", flush=True)
        print(f"[repository-horizon] workspace={campaign.workspace}", flush=True)
        print(f"[repository-horizon] export={export_dir}", flush=True)
        return True
