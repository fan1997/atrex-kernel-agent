from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from long_horizon import main_adapter
from long_horizon.git_episode import git_head
from long_horizon.protocol import atomic_write_json

from .candidate import RepositoryCandidateContract
from .manifest import RepositoryManifest
from .seed import seed_workspace
from .verifier import RepositoryABBAValidator

MODULE_ROOT = Path(__file__).resolve().parent.parent
PLAYBOOK = Path(__file__).resolve().parent / "prompts" / "episode_playbook.md"


class RepositoryIntegration:
    def __init__(
        self,
        manifest: RepositoryManifest,
        source_checkout: Path,
    ):
        self.manifest = manifest
        self.source_checkout = source_checkout.resolve()
        self._contract = RepositoryCandidateContract(manifest)

    def prepare_campaign(self, campaign: Any) -> None:
        if campaign.framework != "CuteDSL":
            raise RuntimeError("repository horizon v1 requires --framework CuteDSL")
        if campaign.framework_baseline != "never":
            raise RuntimeError("repository horizon requires --framework-baseline never")
        seed_workspace(campaign, self.manifest, self.source_checkout)
        self._ensure_measured_v0(campaign)

    def _ensure_measured_v0(self, campaign: Any) -> None:
        memory_path = campaign.workspace / "memory" / "v0.json"
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        if (memory.get("correctness") or {}).get("status") == "PASS":
            return
        revision = git_head(campaign.workspace)
        verifier = RepositoryABBAValidator(
            manifest=self.manifest,
            atrex_bench_root=Path(campaign.atrex_bench_root),
            hardware=campaign.sandbox_hardware,
            profile=campaign.sandbox_profile,
            url=campaign.sandbox_url,
            timeout=campaign.sandbox_timeout,
            repeats=1,
            min_improvement_pct=-100.0,
        )
        verification = verifier.verify(
            campaign.workspace,
            base_commit=revision,
            candidate_commit=revision,
            changed_paths=[],
        )
        if not verification.passed:
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
        memory["performance"] = {
            "latency_us": verification.candidate_latency_us,
            "latency_us_geomean": verification.candidate_latency_us,
            "latency_us_arith_mean": representative.get("latency_us_arith_mean"),
            "latency_us_by_shape": representative.get("latency_us_by_shape", {}),
            "timer": "atrex-bench CUDA event",
            "artifact": verification.artifact,
        }
        memory["correctness"] = {
            "status": "PASS",
            "max_abs_err": representative.get("max_abs_err"),
            "max_rel_err": representative.get("max_rel_err"),
        }
        memory["quality_gate"] = {"result": "PASS", "failure_reason": None}
        atomic_write_json(memory_path, memory)
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
        return RepositoryABBAValidator(
            manifest=self.manifest,
            atrex_bench_root=Path(campaign.atrex_bench_root),
            hardware=campaign.sandbox_hardware,
            profile=campaign.sandbox_profile,
            url=campaign.sandbox_url,
            timeout=campaign.sandbox_timeout,
        )

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
            }
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
        export_paths = [
            "kernel.py",
            "source_manifest.json",
            "source.lock.json",
            self.manifest.vendor_root,
        ]
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
                "v0_commit": root_commit,
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
