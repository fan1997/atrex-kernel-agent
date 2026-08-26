from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from long_horizon import main_adapter
from long_horizon.campaign import LongHorizonCampaign
from long_horizon.git_episode import (
    EpisodeWorktree,
    changed_paths,
    git_head,
    git_text,
    working_changes,
)
from long_horizon.models import VerificationResult
from long_horizon.journal import load as load_journal
from orchestrator.campaign import Campaign
from orchestrator.constants import ATREX_PRIVATE_REFERENCE_ENV

from .baseline import RepositoryBaselineManager
from .candidate import RepositoryCandidateContract
from .config import EvaluationPolicy
from .fixed_route import (
    FixedPreplanRouteStore,
    render_fixed_route_directive,
    validate_fixed_route_outcome,
)
from .manifest import RepositoryManifest
from .prompt import render_prompt
from .runtime import link_repository_runtime


@dataclass
class RepositoryCampaign(Campaign):
    """Current-main ``Campaign`` with only the Wiki-free runtime specialization."""

    repository_manifest: RepositoryManifest | None = field(
        default=None, repr=False, compare=False
    )

    def agent_environment(self) -> dict[str, str]:
        """Keep evaluator-owned paths out of repository coding sessions.

        Repository Horizon candidates are checked privately by the supervisor after
        handoff.  A coding session receives only the public contract and reviewer
        configuration; it must never inherit the directory that owns exact shapes.
        """
        environment = super().agent_environment()
        environment.pop(ATREX_PRIVATE_REFERENCE_ENV, None)
        return environment

    def _sandbox_directive(self) -> str:
        endpoint = (
            f" using gateway URL `{self.sandbox_url}`"
            if self.sandbox_url
            else (
                f" using gateway profile `{self.sandbox_profile}`"
                if self.sandbox_profile
                else " using agate's configured gateway"
            )
        )
        return f"""## GPU sandbox execution (mandatory)

- Target gateway hardware: **{self.sandbox_hardware}**{endpoint}. Every GPU import,
  compile, correctness check, timer, or profiler must cross this gateway.
- This coding session has no private-evaluator capability. Exact shapes, release metadata,
  private evaluator directories, `PROFILE_SHAPE_ID`, and
  `.atrex_private_profile_case.json` are forbidden. Do not invoke
  `repository_horizon.dev_eval`; the supervisor owns full hidden-shape verification and
  same-allocation ABBA promotion after handoff.
- Build development cases only from `agent_problem.json`. Put temporary public-contract
  drivers and profiler harnesses under ignored `profiles/`, and submit them with
  `python tools/sandbox.py --kind dev --hardware {self.sandbox_hardware}` plus the
  configured endpoint and explicit `--input` allowlist. Development measurements are
  evidence, never acceptance authority.
- Never run GPU/JIT packages on the host, install dependencies, mutate or poll the shared
  gateway, or inspect files outside the workspace for evaluator cases.
"""

    def _production_kernel_violations(
        self,
        workspace: Path | None = None,
        *,
        require_gluon: bool = False,
    ) -> list[str]:
        """Use the locked repository contract during main's resume validation."""
        del require_gluon
        if self.repository_manifest is None:
            return ["repository campaign has no source manifest"]
        return RepositoryCandidateContract(
            self.repository_manifest
        ).workspace_violations(self, workspace or self.workspace)

    def _link_runtime(self) -> None:
        self._assert_generalized_inputs_are_private()
        if self.repository_manifest is None:
            raise RuntimeError("repository campaign has no source manifest")
        link_repository_runtime(self, self.workspace, self.repository_manifest)


@dataclass
class RepositoryHorizonCampaign(LongHorizonCampaign):
    """Repository optimization implemented as a thin current-main extension.

    Main owns the episode loop, state machine, canonical memory, hidden-shape
    coverage, interruption recovery, telemetry, and squash promotion.  This class
    supplies only the repository-specific source contract, staging verifier,
    runtime surface, and optional-planning prompt.
    """

    manifest: RepositoryManifest | None = None
    baseline: RepositoryBaselineManager | None = None
    evaluation_policy: EvaluationPolicy = field(
        default_factory=lambda: EvaluationPolicy(wait_mode="inline")
    )
    preplan_route_id: str = ""
    total_episode_target: int = 0

    @property
    def repository_manifest(self) -> RepositoryManifest:
        if self.manifest is None:
            raise RuntimeError("repository horizon has no source manifest")
        return self.manifest

    @property
    def candidate_contract(self) -> RepositoryCandidateContract:
        return RepositoryCandidateContract(self.repository_manifest)

    @property
    def fixed_route_store(self) -> FixedPreplanRouteStore:
        return FixedPreplanRouteStore(self.workspace)

    def _configure_fixed_route(self) -> None:
        if self.preplan_route_id:
            self.fixed_route_store.configure(
                route_id=self.preplan_route_id,
                manifest=self.repository_manifest,
                total_episode_target=self.total_episode_target,
            )

    def _prepare_campaign(self) -> None:
        if self.baseline is None:
            raise RuntimeError("repository horizon has no baseline manager")
        self.baseline.prepare(self.base_campaign)
        if (
            main_adapter.latest_version(self.workspace) < 0
            and (self.workspace / "memory" / "r0.json").is_file()
        ):
            # Bring-up starts from a mechanically locked R0 that does not yet
            # satisfy the workload.  Main's episode state machine can produce
            # canonical V0, but its generic setup session must not replace this
            # repository-backed seed.
            self.base_campaign._link_runtime()
            self.base_campaign.ensure_framework_baseline()
            self._configure_fixed_route()
            return
        # The workspace now has a measured V0.  Delegate resume validation,
        # interrupted-change preservation, framework policy, and private-input
        # checks to the current main implementation.
        main_adapter.prepare_campaign(self.base_campaign)
        self._configure_fixed_route()

    def _link_episode_runtime(self, workspace: Path) -> None:
        link_repository_runtime(self.base_campaign, workspace, self.repository_manifest)

    def _prepare_episode_worktree(self, worktree: EpisodeWorktree, state) -> None:
        del state
        if not self.preplan_route_id:
            self._fixed_route_wip_applied = False
            return
        self._fixed_route_wip_applied = self.fixed_route_store.stage_episode(
            worktree.path
        )

    def _prompt(
        self,
        *,
        episode: int,
        version: int,
        worktree: EpisodeWorktree,
        journal_path: Path,
        handoff_path: Path,
        live_memory_path: Path,
        conversion_pending: bool,
    ) -> str:
        if conversion_pending:
            raise RuntimeError(
                "repository horizon does not support main's Triton-to-Gluon conversion latch"
            )
        route_directive = ""
        if self.preplan_route_id:
            state = self.fixed_route_store.load()
            route_directive = render_fixed_route_directive(
                state,
                self.fixed_route_store.selected_route(),
                wip_applied=getattr(self, "_fixed_route_wip_applied", False),
            )
        return render_prompt(
            campaign=self.base_campaign,
            manifest=self.repository_manifest,
            episode=episode,
            version=version,
            worktree=worktree,
            journal_path=journal_path,
            handoff_path=handoff_path,
            live_memory_path=live_memory_path,
            evaluation_policy=self.evaluation_policy,
            route_directive=route_directive,
        )

    def _completion_check(
        self, worktree: EpisodeWorktree, journal_path: Path, handoff
    ) -> str:
        diagnosis = super()._completion_check(worktree, journal_path, handoff)
        if diagnosis or not self.preplan_route_id:
            return diagnosis
        try:
            journal = load_journal(journal_path)
        except (OSError, UnicodeError, ValueError) as exc:
            return f"cannot validate fixed-route outcome: {exc}"
        diagnosis = validate_fixed_route_outcome(
            journal.get("outcome"),
            route_id=self.preplan_route_id,
            terminal_status=handoff.status,
            has_last_trial_commit=bool(handoff.last_trial_commit),
        )
        if diagnosis:
            return diagnosis
        if handoff.last_trial_commit:
            violation, _ = self._validate_candidate(worktree, handoff.last_trial_commit)
            if violation:
                return "invalid fixed-route last_trial_commit: " + violation
        return ""

    def _after_episode_recorded(
        self,
        *,
        worktree: EpisodeWorktree,
        state,
        result,
        journal: dict[str, Any],
        attempt: dict[str, Any],
        accepted: bool,
    ) -> None:
        del state, attempt
        if not self.preplan_route_id:
            return
        outcome = journal.get("outcome") if isinstance(journal, dict) else None
        route = outcome.get("preplan_route") if isinstance(outcome, dict) else None
        disposition = str(route.get("disposition")) if isinstance(route, dict) else ""
        handoff = result.handoff
        checkpoint = handoff.last_trial_commit if handoff else ""
        if (
            not checkpoint
            and handoff
            and handoff.status == "candidate_ready"
            and not accepted
        ):
            checkpoint = handoff.candidate_commit
        self.fixed_route_store.record_episode(
            worktree=worktree.path,
            base_commit=worktree.base_commit,
            checkpoint=checkpoint,
            accepted=accepted,
            disposition=disposition,
            episode=worktree.episode,
        )

    def _after_recovered_outcome_recorded(
        self, *, episode: int, base_commit: str, outcome_commit: str
    ) -> None:
        if not self.preplan_route_id:
            return
        self.fixed_route_store.reanchor_after_recovery(
            episode=episode,
            base_commit=base_commit,
            outcome_commit=outcome_commit,
            editable_roots=tuple(self.repository_manifest.editable_workspace_roots),
        )

    def _validate_candidate(
        self, worktree: EpisodeWorktree, candidate_commit: str
    ) -> tuple[str, list[str]]:
        # Reuse main's exact commit/branch/ancestry/cleanliness checks, replacing
        # only its single-kernel path rule with the manifest-declared source roots.
        resolved = git_text(
            worktree.path,
            "rev-parse",
            "--verify",
            f"{candidate_commit}^{{commit}}",
            check=False,
        )
        if not resolved:
            return "candidate_commit does not resolve", []
        if resolved != git_head(worktree.path):
            return "candidate_commit must equal episode HEAD", []
        branch = git_text(
            worktree.path,
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            check=False,
        )
        if branch != worktree.branch:
            return "episode worktree left its isolated branch", []
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", worktree.base_commit, resolved],
            cwd=str(worktree.path),
            check=False,
            capture_output=True,
        )
        if ancestor.returncode:
            return "candidate_commit is not descended from incumbent", []
        dirty = working_changes(worktree.path)
        if dirty:
            return (
                "candidate_ready requires a clean worktree: " + ", ".join(dirty[:8]),
                [],
            )
        paths = changed_paths(worktree.path, worktree.base_commit, resolved)
        if not paths:
            return "candidate has no changes relative to incumbent", []
        return self.candidate_contract.validate_changed_paths(paths), paths

    def _candidate_policy_violations(self, workspace: Path) -> list[str]:
        # Main's production checker intentionally evaluates a single ``kernel.py``.
        # Repository Horizon keeps that adapter immutable and changes the locked
        # manifest roots instead, so its equivalent mechanical gate is the
        # repository candidate contract below.
        return self.candidate_contract.workspace_violations(
            self.base_campaign, workspace
        )

    def _verification_paths(self, paths: list[str]) -> list[str]:
        return self.candidate_contract.verification_paths(paths)

    def _copy_runtime_artifacts(
        self, worktree: EpisodeWorktree, episode_dir: Path
    ) -> None:
        super()._copy_runtime_artifacts(worktree, episode_dir)
        source = worktree.path / ".repository_horizon_runtime"
        if source.is_dir():
            destination = episode_dir / "repository_runtime"
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination, symlinks=True)

    def _repository_memory(self, memory: dict[str, Any]) -> dict[str, Any]:
        memory["repository_source"] = {
            "name": self.repository_manifest.source_name,
            "revision": self.repository_manifest.revision,
            "editable_roots": list(self.repository_manifest.editable_workspace_roots),
            "measurement": {
                "schedule": "A-B-B-A",
                "warmup": self.repository_manifest.measurement.warmup,
                "timed_runs": self.repository_manifest.measurement.timed_runs,
            },
        }
        if self.preplan_route_id:
            state = self.fixed_route_store.load()
            memory["fixed_preplan_route"] = {
                **asdict(state),
                "wip_patch_present": self.fixed_route_store.wip_patch_path.is_file(),
            }
        return memory

    def _memory_record(
        self,
        *,
        version: int,
        candidate_commit: str,
        journal: dict[str, Any],
        verification: VerificationResult,
    ) -> dict[str, Any]:
        return self._repository_memory(
            super()._memory_record(
                version=version,
                candidate_commit=candidate_commit,
                journal=journal,
                verification=verification,
            )
        )

    def _outcome_memory_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._repository_memory(super()._outcome_memory_record(**kwargs))
