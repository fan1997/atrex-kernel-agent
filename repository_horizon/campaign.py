from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from long_horizon.models import (
    EpisodeHandoff,
    SessionResult,
    SupervisorState,
    VerificationResult,
)
from long_horizon.protocol import read_handoff

from .baseline import RepositoryBaselineManager
from .candidate import RepositoryCandidateContract
from .config import EvaluationPolicy, endpoint_is_local
from .compat import (
    CampaignStore,
    EpisodeWorktree,
    RUNTIME_DIR,
    changed_paths,
    git_head,
    git_text,
    initialize_journal,
    latest_version,
    load_journal,
    promote_candidate,
    summarize_episode,
    validate_terminal,
    working_changes,
)
from .evaluation import (
    append_evaluation_experiment,
    evaluation_handoff_path,
    load_pending,
)
from .agent_result import compact_resume_prompt
from .manifest import RepositoryManifest
from .prompt import render_prompt
from .profile_eval import collect_profile
from .runtime import autonomous_environment, install_minimal_runtime
from .session import RepositorySessionRunner
from .verifier import RepositoryABBAValidator


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@dataclass
class RepositoryAutonomousCampaign:
    base_campaign: Any
    manifest: RepositoryManifest
    baseline: RepositoryBaselineManager
    verifier: RepositoryABBAValidator
    max_episodes: int = 20
    token_budget: int = 0
    session_timeout: int = 18_000
    handoff_resumes: int = 1
    max_stall: int = 0
    worktree_root: Path | None = None
    session_runner: Any | None = None
    evaluation_policy: EvaluationPolicy = EvaluationPolicy()

    @property
    def workspace(self) -> Path:
        return Path(self.base_campaign.workspace)

    @property
    def candidate_contract(self) -> RepositoryCandidateContract:
        return RepositoryCandidateContract(self.manifest)

    def _recover_interrupted(
        self, store: CampaignStore, state: SupervisorState
    ) -> dict[str, Any] | None:
        active = store.load_active()
        if not active:
            return None
        episode = int(active.get("episode", state.episodes + 1))
        base_commit = str(active.get("base_commit", ""))
        branch = str(active.get("episode_branch", ""))
        path = Path(str(active.get("worktree", "")))
        episode_dir = store.episode_dir(episode)
        if path.is_dir() and base_commit and branch:
            worktree = EpisodeWorktree(episode, base_commit, branch, path)
            checkpoint_path = (
                path / ".repository_horizon_runtime" / "session_checkpoint.json"
            )
            checkpoint = _json_object(checkpoint_path)

            pending_path = Path(str(active.get("verification_pending", "")))
            if (
                active.get("phase") == "awaiting_verification"
                and pending_path.is_file()
            ):
                try:
                    pending_path.resolve().relative_to(
                        (path / ".repository_horizon_runtime" / "evaluations").resolve()
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        "recovered final verification escaped the episode runtime"
                    ) from exc
                pending = load_pending(pending_path)
                observed = read_handoff(path / RUNTIME_DIR / "handoff.json")
                if (
                    observed is None
                    or observed.status != "candidate_ready"
                    or observed.candidate_commit != pending.candidate_commit
                ):
                    raise RuntimeError(
                        "recovered final verification does not match the terminal candidate"
                    )
                verification = self.verifier.collect(pending)
                print(
                    f"[repository-horizon] recovered final verification "
                    f"job={pending.job_id} gate={verification.gate}",
                    flush=True,
                )
                return {
                    "active": active,
                    "worktree": worktree,
                    "checkpoint": checkpoint,
                    "verification": verification,
                }

            evaluation_path = evaluation_handoff_path(path)
            if evaluation_path.is_file() and checkpoint.get("session_id"):
                resume_prompt = self._wait_for_development_evaluation(evaluation_path)
                evaluation_path.unlink(missing_ok=True)
                (path / RUNTIME_DIR / "handoff.json").unlink(missing_ok=True)
                active["phase"] = "exploring"
                active["recovered_evaluation"] = True
                store.save_active(active)
                print(
                    f"[repository-horizon] recovered development evaluation; "
                    f"resuming native session={checkpoint['session_id']}",
                    flush=True,
                )
                return {
                    "active": active,
                    "worktree": worktree,
                    "checkpoint": checkpoint,
                    "resume_prompt": resume_prompt,
                }

            if active.get("phase") == "exploring" and checkpoint.get("session_id"):
                return {
                    "active": active,
                    "worktree": worktree,
                    "checkpoint": checkpoint,
                    "resume_prompt": (
                        "The repository supervisor restarted after your previous invocation. "
                        "Resume the same native session and current worktree. Do not repeat "
                        "completed evaluations; inspect persisted evidence and continue toward "
                        "one honest terminal handoff."
                    ),
                }

        if path.is_dir() and base_commit and branch:
            worktree = EpisodeWorktree(episode, base_commit, branch, path)
            try:
                worktree.archive(episode_dir / "interrupted_archive", "HEAD")
            except Exception as exc:
                print(
                    f"[repository-horizon] WARNING: interrupted archive failed: {exc}",
                    flush=True,
                )
            try:
                worktree.remove(self.workspace)
            except Exception as exc:
                print(
                    f"[repository-horizon] WARNING: interrupted worktree removal failed: {exc}",
                    flush=True,
                )
        if episode > state.episodes:
            state.episodes = episode
            state.interrupted += 1
            state.consecutive_without_promotion += 1
            state.attempts.append(
                {
                    "episode": episode,
                    "status": "interrupted",
                    "accepted": False,
                    "violation": "supervisor process interrupted",
                    "base_commit": base_commit or None,
                    "episode_branch": branch or None,
                }
            )
        store.save_state(state)
        store.clear_active()
        return None

    def _validate_candidate(
        self, worktree: EpisodeWorktree, candidate_commit: str
    ) -> tuple[str, list[str]]:
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
        violation = self.candidate_contract.validate_changed_paths(paths)
        if violation:
            return violation, paths
        policy = self.candidate_contract.workspace_violations(
            self.base_campaign, worktree.path
        )
        if policy:
            return "repository policy rejected candidate: " + "; ".join(policy), paths
        return "", paths

    def _completion_check(
        self,
        worktree: EpisodeWorktree,
        journal_path: Path,
        handoff: EpisodeHandoff,
    ) -> str:
        candidate = handoff.candidate_commit if handoff.status == "candidate_ready" else ""
        diagnosis = validate_terminal(
            journal_path,
            expected_episode=worktree.episode,
            base_commit=worktree.base_commit,
            branch=worktree.branch,
            state=handoff.status,
            candidate_commit=candidate,
        )
        if diagnosis or handoff.status != "candidate_ready":
            return diagnosis
        violation, _ = self._validate_candidate(worktree, candidate)
        if violation:
            return violation
        try:
            journal = load_journal(journal_path)
            finalized = datetime.fromisoformat(
                str(journal["finalized_at"]).replace("Z", "+00:00")
            ).timestamp()
            committed = float(
                git_text(worktree.path, "show", "-s", "--format=%ct", candidate)
            )
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            return f"cannot validate terminal journal ordering: {exc}"
        if finalized <= committed:
            return "candidate journal must be finalized after the exact candidate commit"
        return ""

    def _copy_runtime_artifacts(
        self, worktree: EpisodeWorktree, episode_dir: Path
    ) -> None:
        for relative, name in (
            (RUNTIME_DIR, "episode_runtime"),
            (".repository_horizon_runtime", "repository_runtime"),
        ):
            source = worktree.path / relative
            if not source.is_dir():
                continue
            destination = episode_dir / name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source, destination, symlinks=True)

    def _memory_record(
        self,
        *,
        version: int,
        candidate_commit: str,
        journal: dict[str, Any],
        verification: VerificationResult,
    ) -> dict[str, Any]:
        candidate_runs = [
            run
            for run in verification.runs
            if run.revision == "candidate" and isinstance(run.result, dict)
        ]
        representative = candidate_runs[-1].result if candidate_runs else {}
        outcome = journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
        return {
            "version": f"v{version}",
            "masked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "latency_us": verification.candidate_latency_us,
                "latency_us_geomean": verification.candidate_latency_us,
                "latency_us_arith_mean": representative.get("latency_us_arith_mean"),
                "latency_us_by_shape": representative.get("latency_us_by_shape", {}),
                "authoritative_improvement_pct": verification.improvement_pct,
            },
            "optimization": {
                "action_category": "repository_autonomous_episode",
                "action_description": str(outcome.get("summary", "verified candidate")),
                "expected_impact": "independently verified repository source improvement",
                "risks_and_rollback": "squash promotion from an archived isolated worktree",
            },
            "correctness": {
                "status": "PASS",
                "max_abs_err": representative.get("max_abs_err"),
                "max_rel_err": representative.get("max_rel_err"),
            },
            "quality_gate": {"result": "PASS", "failure_reason": None},
            "open_directions": [
                {"direction": value, "rationale": "episode terminal handoff"}
                for value in outcome.get("next_directions", [])
                if isinstance(value, str)
            ],
            "git_commit_hash": candidate_commit,
            "repository_source": {
                "name": self.manifest.source_name,
                "revision": self.manifest.revision,
                "editable_roots": list(self.manifest.editable_workspace_roots),
                "measurement": {
                    "schedule": "A-B-B-A",
                    "warmup": self.manifest.measurement.warmup,
                    "timed_runs": self.manifest.measurement.timed_runs,
                },
            },
        }

    def _wait_for_development_evaluation(self, handoff_path: Path) -> str:
        value = json.loads(handoff_path.read_text(encoding="utf-8"))
        if value.get("status") != "awaiting_evaluation":
            raise ValueError("repository evaluation handoff has an invalid status")
        pending_path = Path(str(value.get("pending_path", ""))).resolve()
        expected_root = (
            handoff_path.parent / "evaluations"
        ).resolve()
        try:
            pending_path.relative_to(expected_root)
        except ValueError as exc:
            raise ValueError(
                "repository evaluation pending path escaped the episode runtime"
            ) from exc
        pending = load_pending(pending_path)
        if pending.job_id != str(value.get("job_id", "")):
            raise ValueError("repository evaluation handoff job id does not match pending state")
        print(
            f"[repository-horizon] evaluation={pending.evaluation_id} "
            f"job={pending.job_id} phase=awaiting_evaluation",
            flush=True,
        )
        verification = (
            collect_profile(pending)
            if pending.kind == "profile"
            else self.verifier.collect(pending)
        )
        print(
            f"[repository-horizon] evaluation={pending.evaluation_id} "
            f"job={pending.job_id} gate={verification.gate} phase=evaluation_complete",
            flush=True,
        )
        append_evaluation_experiment(handoff_path.parent.parent, pending, verification)
        return compact_resume_prompt(
            pending.directory / "agent_result.json",
            self.evaluation_policy.resume_prompt_max_bytes,
        )

    def run(self) -> str:
        environment = autonomous_environment()
        os.environ["ATREX_CODEX_SESSION_SETTINGS"] = environment[
            "ATREX_CODEX_SESSION_SETTINGS"
        ]
        self.baseline.prepare(self.base_campaign)
        store = CampaignStore(self.workspace)
        state = store.load_state()
        recovered = self._recover_interrupted(store, state)
        if working_changes(self.workspace):
            raise RuntimeError(
                "repository campaign requires a clean incumbent workspace: "
                + ", ".join(working_changes(self.workspace)[:12])
            )
        runner = self.session_runner or RepositorySessionRunner(
            agent_cli=self.base_campaign.agent_cli,
            evaluation_waiter=self._wait_for_development_evaluation,
            wait_mode=self.evaluation_policy.resolved_wait_mode(
                self.base_campaign.agent_cli,
                endpoint_is_local=endpoint_is_local(
                    self.base_campaign.sandbox_url,
                    self.base_campaign.sandbox_hardware,
                ),
            ),
            suspend_enforcement=self.evaluation_policy.suspend_enforcement,
            suspend_grace_seconds=self.evaluation_policy.suspend_grace_seconds,
            evaluation_backend=self.evaluation_policy.backend,
        )
        reason = "budget: max-episodes"

        while state.episodes < self.max_episodes:
            if self.token_budget and state.tokens >= self.token_budget:
                reason = "budget: token-budget"
                break
            if self.max_stall and state.consecutive_without_promotion >= self.max_stall:
                reason = f"stall: {state.consecutive_without_promotion} episodes"
                break

            recovered_verification: VerificationResult | None = None
            prior_invocation_count = 0
            if recovered is not None:
                active = dict(recovered["active"])
                worktree = recovered["worktree"]
                checkpoint = dict(recovered.get("checkpoint") or {})
                episode = int(active["episode"])
                base_commit = str(active["base_commit"])
                memory_version = int(active.get("memory_version", latest_version(self.workspace) + 1))
                runtime = worktree.path / RUNTIME_DIR
                journal_path = runtime / "journal.json"
                handoff_path = runtime / "handoff.json"
                brief_path = store.episode_dir(episode) / "BRIEF.md"
                prompt = (
                    brief_path.read_text(encoding="utf-8")
                    if brief_path.is_file()
                    else ""
                )
                recovered_verification = recovered.get("verification")
                prior_invocation_count = int(checkpoint.get("invocation_count", 0))
                if recovered_verification is not None:
                    observed = read_handoff(handoff_path)
                    diagnosis = (
                        self._completion_check(worktree, journal_path, observed)
                        if observed is not None
                        else "missing recovered handoff"
                    )
                    result = SessionResult(
                        exit_status=0 if not diagnosis else 1,
                        timed_out=False,
                        tokens=int(checkpoint.get("tokens", 0)),
                        session_id=str(checkpoint.get("session_id", "")),
                        resume_count=int(checkpoint.get("resume_count", 0)),
                        handoff=observed,
                        completion_diagnosis=diagnosis,
                    )
                else:
                    result = runner.run(
                        worktree.path,
                        str(recovered.get("resume_prompt") or prompt),
                        timeout=self.session_timeout,
                        handoff_path=handoff_path,
                        handoff_resumes=self.handoff_resumes,
                        completion_check=lambda handoff: self._completion_check(
                            worktree, journal_path, handoff
                        ),
                        session_id=str(checkpoint["session_id"]),
                        resume_existing=True,
                        initial_tokens=int(checkpoint.get("tokens", 0)),
                        initial_resume_count=int(checkpoint.get("resume_count", 0)),
                        initial_invocation_count=prior_invocation_count,
                        initial_agent_seconds_remaining=float(
                            checkpoint.get("agent_seconds_remaining", self.session_timeout)
                        ),
                        telemetry_environment={
                            "ATREX_TELEMETRY_TRACE": str(runtime / "telemetry.jsonl"),
                            "ATREX_TELEMETRY_CAMPAIGN_ID": str(
                                getattr(self.base_campaign, "campaign_name", self.workspace.name)
                            ),
                            "ATREX_TELEMETRY_ITERATION_ID": f"episode-{episode:04d}",
                            "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
                        },
                    )
                recovered = None
            else:
                episode = state.episodes + 1
                base_commit = git_head(self.workspace)
                memory_version = latest_version(self.workspace) + 1
                worktree = EpisodeWorktree.plan(
                    self.workspace,
                    episode,
                    base_commit,
                    root=self.worktree_root,
                )
                active = {
                    "episode": episode,
                    "memory_version": memory_version,
                    "base_commit": base_commit,
                    "episode_branch": worktree.branch,
                    "worktree": str(worktree.path),
                    "phase": "preparing",
                }
                store.save_active(active)
                worktree.materialize(self.workspace)
                install_minimal_runtime(self.base_campaign, worktree.path, self.manifest)
                unexpected = working_changes(worktree.path)
                if unexpected:
                    raise RuntimeError(
                        "minimal runtime dirtied the episode boundary: "
                        + ", ".join(unexpected)
                    )
                active["phase"] = "exploring"
                store.save_active(active)
                runtime = worktree.path / RUNTIME_DIR
                journal_path = runtime / "journal.json"
                handoff_path = runtime / "handoff.json"
                initialize_journal(
                    journal_path,
                    episode=episode,
                    base_commit=base_commit,
                    branch=worktree.branch,
                )
                prompt = render_prompt(
                    campaign=self.base_campaign,
                    manifest=self.manifest,
                    episode=episode,
                    worktree=worktree,
                    journal_path=journal_path,
                    handoff_path=handoff_path,
                    state=state,
                    evaluation_policy=self.evaluation_policy,
                )
                store.write_brief(episode, prompt)
                result = runner.run(
                    worktree.path,
                    prompt,
                    timeout=self.session_timeout,
                    handoff_path=handoff_path,
                    handoff_resumes=self.handoff_resumes,
                    completion_check=lambda handoff: self._completion_check(
                        worktree, journal_path, handoff
                    ),
                    telemetry_environment={
                    "ATREX_TELEMETRY_TRACE": str(runtime / "telemetry.jsonl"),
                    "ATREX_TELEMETRY_CAMPAIGN_ID": str(
                        getattr(self.base_campaign, "campaign_name", self.workspace.name)
                    ),
                    "ATREX_TELEMETRY_ITERATION_ID": f"episode-{episode:04d}",
                    "ATREX_TELEMETRY_ATTEMPT_ID": "invocation",
                    },
                )
            state.episodes = episode
            state.tokens += result.tokens
            handoff = result.handoff
            status = handoff.status if handoff else "invalid_handoff"
            candidate_commit = handoff.candidate_commit if handoff else ""
            violation = ""
            paths: list[str] = []
            verification: VerificationResult | None = recovered_verification
            accepted = False
            if result.exit_status != 0 or result.timed_out:
                violation = (
                    f"session failed: exit={result.exit_status} timeout={result.timed_out}"
                )
            elif handoff is None:
                violation = result.completion_diagnosis or "missing terminal handoff"
            elif status == "candidate_ready":
                violation, paths = self._validate_candidate(worktree, candidate_commit)
                if not violation and verification is None:
                    active["phase"] = "submitting_verification"
                    store.save_active(active)
                    try:
                        pending = self.verifier.submit(
                            worktree.path,
                            base_commit=base_commit,
                            candidate_commit=candidate_commit,
                            changed_paths=self.candidate_contract.verification_paths(
                                paths
                            ),
                        )
                        active.update(
                            {
                                "phase": "awaiting_verification",
                                "verification_id": pending.evaluation_id,
                                "verification_job_id": pending.job_id,
                                "verification_pending": str(pending.pending_path),
                            }
                        )
                        store.save_active(active)
                        verification = self.verifier.collect(pending)
                    except Exception as exc:
                        verification = VerificationResult(
                            "ERROR",
                            None,
                            None,
                            None,
                            error=(
                                "repository final verification failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
                if verification is not None:
                    accepted = verification.passed

            episode_dir = store.episode_dir(episode)
            worktree.archive(episode_dir / "archive", "HEAD")
            self._copy_runtime_artifacts(worktree, episode_dir)
            try:
                journal = load_journal(journal_path)
            except Exception:
                journal = {}
            outcome = journal.get("outcome") if isinstance(journal.get("outcome"), dict) else {}
            attempt: dict[str, Any] = {
                "episode": episode,
                "version": memory_version if accepted else None,
                "status": status,
                "accepted": accepted,
                "violation": violation or None,
                "base_commit": base_commit,
                "episode_branch": worktree.branch,
                "episode_head": git_head(worktree.path),
                "candidate_commit": candidate_commit or None,
                "changed_paths": paths,
                "session_id": result.session_id,
                "resume_count": result.resume_count,
                "agent_invocations": max(
                    len(result.invocations),
                    int(
                        _json_object(
                            worktree.path
                            / ".repository_horizon_runtime"
                            / "session_checkpoint.json"
                        ).get("invocation_count", prior_invocation_count)
                    ),
                ),
                "child_sessions": 0,
                "wait_agent_calls": 0,
                "evaluation_jobs": len(
                    list(
                        (
                            worktree.path
                            / ".repository_horizon_runtime"
                            / "evaluations"
                        ).glob("*/pending.json")
                    )
                ),
                "prompt_bytes": len(prompt.encode("utf-8")),
                "tokens": result.tokens,
                "summary": outcome.get("summary") if isinstance(outcome, dict) else None,
                "next_directions": (
                    outcome.get("next_directions") if isinstance(outcome, dict) else None
                ),
                "verification": verification.as_dict() if verification else None,
            }
            try:
                telemetry = summarize_episode(
                    episode=episode,
                    version=memory_version,
                    status=status,
                    accepted=accepted,
                    control_tokens=result.tokens,
                    resume_count=result.resume_count,
                    invocations=result.invocations,
                )
                attempt["telemetry"] = {
                    "summary": str(
                        store.archive_telemetry(episode, telemetry).relative_to(
                            store.workspace
                        )
                    ),
                    "measurement": telemetry["measurement"],
                    "reason_codes": telemetry["reason_codes"],
                }
            except Exception as exc:
                attempt["telemetry"] = {
                    "summary": None,
                    "measurement": "unavailable",
                    "reason_codes": [
                        f"telemetry_finalize_failed:{type(exc).__name__}"
                    ],
                }

            promotion_commit = ""
            if accepted and verification is not None:
                active["phase"] = "promoting"
                store.save_active(active)
                memory = self._memory_record(
                    version=memory_version,
                    candidate_commit=candidate_commit,
                    journal=journal,
                    verification=verification,
                )
                promotion_commit = promote_candidate(
                    self.workspace,
                    base_commit=base_commit,
                    candidate_commit=candidate_commit,
                    episode=episode,
                    evidence={**attempt, "journal": journal},
                    memory_version=memory_version,
                    memory_record=memory,
                )
                attempt["promotion_commit"] = promotion_commit
                state.accepted += 1
                state.consecutive_without_promotion = 0
            else:
                state.consecutive_without_promotion += 1
                if status == "pivot" and not violation:
                    state.pivoted += 1
                elif status == "blocked" and not violation:
                    state.blocked += 1
                elif status == "invalid_handoff":
                    state.protocol_failures += 1
                else:
                    state.rejected += 1

            store.archive_attempt(episode, attempt)
            state.attempts.append(attempt)
            store.save_state(state)
            worktree.remove(self.workspace)
            store.clear_active()
            print(
                f"[repository-horizon] episode={episode} status={status} "
                f"accepted={accepted} canonical_version="
                f"{f'v{memory_version}' if accepted else '-'} tokens={result.tokens} "
                f"commit={promotion_commit or '-'}",
                flush=True,
            )
            if status == "blocked" and not violation:
                reason = "blocked"
                break

        print(
            f"[repository-horizon] STOP {reason}; episodes={state.episodes} "
            f"accepted={state.accepted} rejected={state.rejected} "
            f"pivoted={state.pivoted} blocked={state.blocked} "
            f"protocol_failures={state.protocol_failures} tokens={state.tokens}",
            flush=True,
        )
        return reason
