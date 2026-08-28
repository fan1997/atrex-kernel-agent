from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
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
from long_horizon.journal import load as load_journal
from long_horizon.models import VerificationResult
from orchestrator.campaign import Campaign
from orchestrator.constants import ATREX_PRIVATE_REFERENCE_ENV

from .baseline import RepositoryBaselineManager
from .candidate import RepositoryCandidateContract
from .config import EvaluationPolicy
from .manifest import RepositoryManifest
from .prompt import render_prompt
from .runtime import link_repository_runtime
from .strategy import (
    ARCHITECTURE_MAP_WORKTREE_PATH,
    RepositoryStrategyStore,
    load_architecture_map,
    render_strategy_directive,
    validate_architecture_outcome,
)


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
    architecture_escape_after: int = 5
    architecture_review_interval: int = 8
    architecture_commitment_episodes: int = 3

    @property
    def repository_manifest(self) -> RepositoryManifest:
        if self.manifest is None:
            raise RuntimeError("repository horizon has no source manifest")
        return self.manifest

    @property
    def candidate_contract(self) -> RepositoryCandidateContract:
        return RepositoryCandidateContract(self.repository_manifest)

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
            return
        # The workspace now has a measured V0.  Delegate resume validation,
        # interrupted-change preservation, framework policy, and private-input
        # checks to the current main implementation.
        main_adapter.prepare_campaign(self.base_campaign)

    def _link_episode_runtime(self, workspace: Path) -> None:
        link_repository_runtime(self.base_campaign, workspace, self.repository_manifest)

    @property
    def strategy_store(self) -> RepositoryStrategyStore:
        return RepositoryStrategyStore(self.workspace)

    def _prepare_episode_worktree(self, worktree: EpisodeWorktree, state) -> None:
        # Outcome commits normally advance only supervisor-owned memory.  Older
        # versions could leave a carried architecture patch anchored to the
        # preceding commit after an invalid handoff, making every subsequent
        # pre-agent setup fail before the strategy hook had a chance to repair it.
        # Reconcile that safe metadata-only transition before applying the patch.
        self._reanchor_architecture_wip(
            episode=worktree.episode,
            outcome_commit=worktree.base_commit,
            event="architecture_wip_reanchored_before_episode",
        )
        strategy = self.strategy_store.enter_if_needed(
            episode=worktree.episode,
            consecutive_without_promotion=state.consecutive_without_promotion,
            escape_after=self.architecture_escape_after,
            review_interval=self.architecture_review_interval,
            commitment_episodes=self.architecture_commitment_episodes,
        )
        wip_applied = self.strategy_store.apply_wip(worktree.path, strategy)
        self.strategy_store.stage_runtime_snapshot(worktree.path)
        self._strategy_episode_context = (strategy, wip_applied)

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
        strategy, wip_applied = getattr(
            self,
            "_strategy_episode_context",
            (self.strategy_store.load(), False),
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
            strategy_directive=render_strategy_directive(
                strategy,
                escape_after=self.architecture_escape_after,
                review_interval=self.architecture_review_interval,
                commitment_episodes=self.architecture_commitment_episodes,
                wip_applied=wip_applied,
            ),
        )

    def _completion_check(
        self, worktree: EpisodeWorktree, journal_path: Path, handoff
    ) -> str:
        diagnosis = super()._completion_check(worktree, journal_path, handoff)
        if diagnosis:
            return diagnosis
        strategy = self.strategy_store.load()
        if strategy.mode != "architecture":
            return ""
        architecture_map, diagnosis = load_architecture_map(
            worktree.path / ARCHITECTURE_MAP_WORKTREE_PATH
        )
        if diagnosis:
            return diagnosis
        try:
            journal = load_journal(journal_path)
        except (OSError, UnicodeError, ValueError) as exc:
            return f"cannot validate architecture outcome: {exc}"
        diagnosis = validate_architecture_outcome(
            journal.get("outcome"),
            terminal_status=handoff.status,
            has_last_trial_commit=bool(handoff.last_trial_commit),
        )
        if diagnosis:
            return diagnosis
        selected = architecture_map.get("selected_direction_id")
        actual = journal["outcome"]["architecture"]["direction_id"]
        if selected != actual:
            return "architecture outcome direction_id must match architecture map selection"
        architecture = journal["outcome"]["architecture"]
        if architecture.get("disposition") == "architecture_refuted":
            evidence = architecture["independent_review"].get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                return "architecture refutation requires independent review evidence"
            relative = PurePosixPath(evidence)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.as_posix().startswith("plans/")
                or not (worktree.path / relative).is_file()
            ):
                return "architecture review evidence must be an existing plans/ file"
        if handoff.last_trial_commit:
            violation, _ = self._validate_candidate(worktree, handoff.last_trial_commit)
            if violation:
                return "invalid architecture last_trial_commit: " + violation
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
        del attempt
        store = self.strategy_store
        handoff = result.handoff
        checkpoint = (
            handoff.last_trial_commit
            if handoff and handoff.last_trial_commit
            else (
                handoff.candidate_commit
                if handoff and handoff.status == "candidate_ready" and not accepted
                else ""
            )
        )
        if not accepted and not checkpoint:
            # An invalid/missing handoff still creates a canonical outcome commit.
            # Preserve any previously validated WIP patch across that metadata-only
            # commit instead of leaving it anchored to the old incumbent forever.
            self._reanchor_architecture_wip(
                episode=worktree.episode,
                expected_base_commit=worktree.base_commit,
                outcome_commit=git_head(self.workspace),
                event="architecture_wip_reanchored_after_uncheckpointed_outcome",
            )
        strategy = store.load()
        if strategy.mode != "architecture":
            return
        if handoff is None:
            # The journal and its architecture outcome are agent-authored evidence,
            # but a failed terminal handoff did not pass the protocol gate.  Keep
            # the last validated direction and WIP paired instead of combining an
            # untrusted new direction with the previous episode's patch.
            strategy.history.append(
                {
                    "event": "architecture_episode_invalid_handoff_ignored",
                    "episode": worktree.episode,
                    "preserved_direction_id": strategy.active_direction_id,
                    "preserved_wip_source_commit": strategy.wip_source_commit,
                }
            )
            store.save(strategy)
            return
        architecture_map, diagnosis = load_architecture_map(
            worktree.path / ARCHITECTURE_MAP_WORKTREE_PATH
        )
        if architecture_map is not None and not diagnosis:
            from long_horizon.protocol import atomic_write_json

            atomic_write_json(store.architecture_map_path, architecture_map)
        outcome = journal.get("outcome") if isinstance(journal, dict) else None
        architecture = (
            outcome.get("architecture") if isinstance(outcome, dict) else None
        )
        if not isinstance(architecture, dict):
            return
        strategy.active_direction_id = str(architecture.get("direction_id", ""))
        strategy.active_thesis = str(architecture.get("thesis", ""))
        strategy.commitment_remaining = max(0, strategy.commitment_remaining - 1)
        disposition = str(architecture.get("disposition", ""))
        strategy.history.append(
            {
                "event": "architecture_episode_recorded",
                "episode": worktree.episode,
                "direction_id": strategy.active_direction_id,
                "disposition": disposition,
                "accepted": accepted,
            }
        )
        if checkpoint:
            completed = subprocess.run(
                ["git", "diff", "--binary", worktree.base_commit, checkpoint, "--"],
                cwd=str(worktree.path),
                capture_output=True,
                check=True,
            )
            temporary = store.wip_patch_path.with_suffix(".patch.tmp")
            temporary.write_bytes(completed.stdout)
            temporary.replace(store.wip_patch_path)
            # A rejected candidate is recorded as a new canonical outcome commit
            # before this hook runs.  Its editable source is still the incumbent,
            # but its commit identity has advanced because memory was recorded.
            # Anchor the carry-forward patch to that new canonical HEAD so the next
            # isolated worktree can restore it without weakening apply_wip's exact
            # base safety check.
            strategy.wip_base_commit = git_head(self.workspace)
            strategy.wip_source_commit = checkpoint
            strategy.wip_patch_sha256 = hashlib.sha256(completed.stdout).hexdigest()
        elif disposition in {
            "implementation_refuted",
            "suspend",
            "architecture_refuted",
        }:
            store.wip_patch_path.unlink(missing_ok=True)
            strategy.wip_base_commit = ""
            strategy.wip_source_commit = ""
            strategy.wip_patch_sha256 = ""
        if accepted or disposition in {"suspend", "architecture_refuted"}:
            strategy.mode = "normal"
            strategy.last_review_episode = worktree.episode
            strategy.entered_episode = None
            strategy.commitment_remaining = 0
            strategy.review_required = False
            if accepted or disposition == "promote":
                store.wip_patch_path.unlink(missing_ok=True)
        elif disposition == "implementation_refuted":
            strategy.active_direction_id = ""
            strategy.active_thesis = ""
            strategy.review_required = strategy.commitment_remaining == 0
        elif strategy.commitment_remaining == 0:
            strategy.review_required = True
        store.save(strategy)

    def _reanchor_architecture_wip(
        self,
        *,
        episode: int,
        outcome_commit: str,
        event: str,
        expected_base_commit: str = "",
    ) -> None:
        store = self.strategy_store
        strategy = store.load()
        if strategy.mode != "architecture" or not store.wip_patch_path.is_file():
            return
        if strategy.wip_base_commit == outcome_commit:
            return
        old_base = strategy.wip_base_commit
        if not old_base:
            raise RuntimeError(
                "cannot re-anchor architecture WIP without its recorded base"
            )
        if expected_base_commit and old_base != expected_base_commit:
            raise RuntimeError(
                "cannot re-anchor architecture WIP after outcome: its recorded "
                "base does not match the episode base"
            )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", old_base, outcome_commit],
            cwd=str(self.workspace),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if ancestor.returncode != 0:
            raise RuntimeError(
                "cannot re-anchor architecture WIP after outcome: the recorded "
                "base is not an ancestor of the incumbent"
            )
        completed = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                old_base,
                outcome_commit,
                "--",
                *self.repository_manifest.editable_workspace_roots,
            ],
            cwd=str(self.workspace),
            check=False,
        )
        if completed.returncode == 1:
            raise RuntimeError(
                "cannot re-anchor architecture WIP after outcome: editable "
                "repository roots changed"
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "cannot verify editable repository roots while re-anchoring "
                "architecture WIP after outcome"
            )
        strategy.wip_base_commit = outcome_commit
        strategy.history.append(
            {
                "event": event,
                "episode": episode,
                "old_base_commit": old_base,
                "new_base_commit": outcome_commit,
            }
        )
        store.save(strategy)

    def _after_recovered_outcome_recorded(
        self,
        *,
        episode: int,
        base_commit: str,
        outcome_commit: str,
    ) -> None:
        self._reanchor_architecture_wip(
            episode=episode,
            expected_base_commit=base_commit,
            outcome_commit=outcome_commit,
            event="architecture_wip_reanchored_after_recovery",
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

    def _repository_memory(
        self, memory: dict[str, Any], journal: dict[str, Any] | None = None
    ) -> dict[str, Any]:
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
        strategy = self.strategy_store.load()
        memory["repository_strategy"] = {
            **asdict(strategy),
            "wip_patch_present": self.strategy_store.wip_patch_path.is_file(),
        }
        outcome = journal.get("outcome") if isinstance(journal, dict) else None
        if isinstance(outcome, dict) and isinstance(outcome.get("architecture"), dict):
            memory["architecture_episode"] = outcome["architecture"]
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
            ),
            journal,
        )

    def _outcome_memory_record(self, **kwargs: Any) -> dict[str, Any]:
        return self._repository_memory(
            super()._outcome_memory_record(**kwargs), kwargs.get("journal")
        )
