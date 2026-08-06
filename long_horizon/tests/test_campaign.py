from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from long_horizon.campaign import LongHorizonCampaign, _conversion_parity_passes
from long_horizon.git_episode import EpisodeWorktree, git_head, record_episode_outcome
from long_horizon.journal import append_experiment, finalize
from long_horizon.models import (
    EpisodeHandoff,
    InvocationObservation,
    SessionResult,
    VerificationResult,
    VerificationRun,
)
from long_horizon.protocol import atomic_write_json
from long_horizon.store import CampaignStore
from long_horizon.tests.helpers import init_repo, run_git
from orchestrator.agent_runtime.model import (
    AgentRuntimeCapabilities,
    NormalizedAgentEvent,
    TokenUsage,
)


class CandidateRunner:
    def __init__(
        self,
        value: int = 5,
        invocations: tuple[InvocationObservation, ...] = (),
    ):
        self.value = value
        self.invocations = invocations
        self.telemetry_environment = None

    def run(
        self,
        workspace,
        prompt,
        *,
        timeout,
        handoff_path,
        handoff_resumes,
        completion_check,
        **kwargs,
    ):
        self.telemetry_environment = kwargs.get("telemetry_environment")
        (workspace / "kernel.py").write_text(f"VALUE = {self.value}\n", encoding="utf-8")
        run_git(workspace, "add", "kernel.py")
        run_git(workspace, "commit", "-m", "candidate")
        candidate = git_head(workspace)
        journal_path = workspace / ".atrex_long_horizon" / "journal.json"
        append_experiment(
            journal_path,
            {
                "name": "rewrite",
                "hypothesis": "less work",
                "evidence": "development benchmark",
                "result": "faster",
                "decision": "continue",
            },
        )
        finalize(
            journal_path,
            state="candidate_ready",
            candidate_commit=candidate,
            outcome={"summary": "candidate is faster", "next_directions": ["more tuning"]},
        )
        handoff = EpisodeHandoff("candidate_ready", candidate_commit=candidate)
        atomic_write_json(handoff_path, handoff.as_dict())
        diagnosis = completion_check(handoff)
        if diagnosis:
            raise AssertionError(diagnosis)
        return SessionResult(
            exit_status=0,
            timed_out=False,
            tokens=123,
            session_id="session-1",
            resume_count=0,
            handoff=handoff,
            invocations=self.invocations,
        )


class BlockedRunner:
    def __init__(self):
        self.calls = 0

    def run(
        self,
        workspace,
        prompt,
        *,
        timeout,
        handoff_path,
        handoff_resumes,
        completion_check,
        **kwargs,
    ):
        self.calls += 1
        journal_path = workspace / ".atrex_long_horizon" / "journal.json"
        append_experiment(
            journal_path,
            {
                "name": f"blocked probe {self.calls}",
                "hypothesis": "external infrastructure may recover on a fresh attempt",
                "evidence": "the required external route is unavailable",
                "result": "blocked",
                "decision": "pivot",
            },
        )
        finalize(
            journal_path,
            state="blocked",
            outcome={
                "summary": f"external infrastructure blocker {self.calls}",
                "next_directions": ["retry with a fresh episode"],
            },
        )
        handoff = EpisodeHandoff("blocked")
        atomic_write_json(handoff_path, handoff.as_dict())
        diagnosis = completion_check(handoff)
        if diagnosis:
            raise AssertionError(diagnosis)
        return SessionResult(
            exit_status=0,
            timed_out=False,
            tokens=10,
            session_id=f"blocked-session-{self.calls}",
            resume_count=0,
            handoff=handoff,
        )


class FixedVerifier:
    def __init__(self, passed: bool):
        self.passed = passed

    def verify(self, workspace, *, base_commit, candidate_commit, changed_paths):
        candidate_latency = 8.0 if self.passed else 11.0
        improvement = 20.0 if self.passed else -10.0
        runs = [
            VerificationRun(
                "incumbent", 0, 0,
                {"all_pass": True, "latency_us_geomean": 10.0, "latency_us_by_shape": {"0": 10.0}},
            ),
            VerificationRun(
                "candidate", 0, 0,
                {"all_pass": True, "latency_us_geomean": candidate_latency, "latency_us_by_shape": {"0": candidate_latency}},
            ),
        ]
        return VerificationResult(
            "PASS" if self.passed else "FAIL",
            candidate_latency,
            10.0,
            improvement,
            runs=runs,
            error="" if self.passed else "regression",
        )


class BootstrapVerifier:
    def __init__(self, passed: bool):
        self.passed = passed

    def verify(self, workspace, *, base_commit, candidate_commit, changed_paths):
        latency = 8.0 if self.passed else None
        runs = [
            VerificationRun(
                "candidate",
                0,
                0 if self.passed else 1,
                {
                    "all_pass": self.passed,
                    "latency_us_geomean": latency,
                    "latency_us_by_shape": {"0": latency},
                }
                if self.passed
                else None,
            )
        ]
        return VerificationResult(
            "PASS" if self.passed else "FAIL",
            latency,
            None,
            None,
            runs=runs,
            error="" if self.passed else "candidate is not correct",
        )


def fake_base(workspace: Path):
    return SimpleNamespace(
        workspace=workspace,
        platform="B200",
        framework="CuteDSL",
        notes="test",
        arch="sm_100",
        sandbox_hardware="REMOTE_GPU",
        sandbox_profile="",
        sandbox_url="",
        sandbox_timeout=600,
        atrex_bench_root="",
        optimization_mode="leaderboard",
        agent_cli="claude",
        target_util=90.0,
        _notify_improvement=mock.Mock(),
        _notify_iteration=mock.Mock(),
    )


def seed_version(repo: Path, version: int, *, accepted_kernel: bool) -> str:
    (repo / "memory").mkdir(exist_ok=True)
    (repo / "memory" / f"v{version}.json").write_text(
        json.dumps(
            {
                "version": f"v{version}",
                "quality_gate": {"result": "PASS" if accepted_kernel else "FAIL"},
            }
        ),
        encoding="utf-8",
    )
    if accepted_kernel:
        (repo / "kernel.py").write_text("VALUE = 9\n", encoding="utf-8")
    run_git(repo, "add", ".")
    run_git(repo, "commit", "-m", f"seed v{version}")
    return git_head(repo)


class CampaignIntegrationTests(unittest.TestCase):
    def test_conversion_parity_matches_main_tolerance(self) -> None:
        runs = [
            VerificationRun("incumbent", 0, 0, {"all_pass": True}),
            VerificationRun("candidate", 0, 0, {"all_pass": True}),
        ]
        self.assertTrue(
            _conversion_parity_passes(
                VerificationResult("FAIL", 10.4, 10.0, -4.0, runs=runs)
            )
        )
        self.assertFalse(
            _conversion_parity_passes(
                VerificationResult("FAIL", 10.6, 10.0, -6.0, runs=runs)
            )
        )

    def test_main_max_iters_caps_canonical_versions_not_episode_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "campaign"
            init_repo(repo)
            (repo / "memory").mkdir()
            (repo / "memory" / "v5.json").write_text(
                json.dumps({"version": "v5", "quality_gate": {"result": "FAIL"}}),
                encoding="utf-8",
            )
            run_git(repo, "add", "memory/v5.json")
            run_git(repo, "commit", "-m", "v5 record")
            runner = mock.Mock()
            with mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None):
                reason = LongHorizonCampaign(
                    base_campaign=fake_base(repo),
                    max_version=5,
                    session_runner=runner,
                ).run()
            self.assertEqual(reason, "budget: max-iters")
            runner.run.assert_not_called()

    def _patches(self):
        return (
            mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None),
            mock.patch("long_horizon.main_adapter.link_episode_runtime", return_value=None),
            mock.patch(
                "long_horizon.main_adapter.episode_directives",
                return_value={
                    "hardware": "hardware",
                    "sandbox": "sandbox",
                    "evaluator": "evaluator",
                    "mode_policy": "policy",
                },
            ),
            mock.patch(
                "long_horizon.main_adapter.iteration_playbook",
                return_value="current main playbook",
            ),
            mock.patch("long_horizon.main_adapter.latest_version", return_value=0),
        )

    def _bootstrap_patches(self):
        return (
            mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None),
            mock.patch("long_horizon.main_adapter.link_episode_runtime", return_value=None),
            mock.patch(
                "long_horizon.main_adapter.episode_directives",
                return_value={
                    "hardware": "hardware",
                    "sandbox": "sandbox",
                    "evaluator": "evaluator",
                    "mode_policy": "policy",
                },
            ),
            mock.patch(
                "long_horizon.main_adapter.iteration_playbook",
                return_value="current main playbook",
            ),
        )

    def test_failed_bootstrap_attempt_does_not_create_v0(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            patches = self._bootstrap_patches()
            with patches[0], patches[1], patches[2], patches[3]:
                base_campaign = fake_base(repo)
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    verifier=BootstrapVerifier(False),
                    session_runner=CandidateRunner(20),
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "bootstrap: max-episodes")
            self.assertFalse((repo / "memory" / "v0.json").exists())
            self.assertTrue((repo / "memory" / "bootstrap_e0001.json").is_file())
            self.assertEqual(run_git(repo, "diff", base, "HEAD", "--", "kernel.py"), "")
            memory = json.loads(
                (repo / "memory" / "bootstrap_e0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(memory["version"], "bootstrap-e1")
            self.assertTrue(memory["long_horizon"]["bootstrap"])
            self.assertFalse(memory["long_horizon"]["canonical_version_created"])
            base_campaign._notify_iteration.assert_not_called()

    def test_first_passing_bootstrap_candidate_is_promoted_as_v0(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            init_repo(repo)
            patches = self._bootstrap_patches()
            with patches[0], patches[1], patches[2], patches[3]:
                base_campaign = fake_base(repo)
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    verifier=BootstrapVerifier(True),
                    session_runner=CandidateRunner(5),
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "max-episodes")
            self.assertTrue((repo / "memory" / "v0.json").is_file())
            self.assertFalse((repo / "memory" / "bootstrap_e0001.json").exists())
            memory = json.loads(
                (repo / "memory" / "v0.json").read_text(encoding="utf-8")
            )
            self.assertEqual(memory["version"], "v0")
            self.assertEqual(memory["optimization"]["action_category"], "repository_bringup")
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 5\n")
            base_campaign._notify_iteration.assert_called_once()

    def test_verified_candidate_is_squash_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                base_campaign = fake_base(repo)
                runner = CandidateRunner(5)
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    verifier=FixedVerifier(True),
                    session_runner=runner,
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "max-episodes")
            self.assertNotEqual(git_head(repo), base)
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 5\n")
            state = json.loads((repo / ".atrex_long_horizon/state.json").read_text())
            self.assertEqual(state["accepted"], 1)
            self.assertTrue((repo / "memory/v1.json").is_file())
            base_campaign._notify_improvement.assert_called_once()
            base_campaign._notify_iteration.assert_called_once()
            self.assertTrue(base_campaign._notify_iteration.call_args.args[2])
            self.assertIn("ATREX_TELEMETRY_TRACE", runner.telemetry_environment)
            telemetry_path = (
                repo
                / ".atrex_long_horizon/episodes/e0001/telemetry.summary.json"
            )
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            self.assertEqual(telemetry["control_tokens"], 123)
            self.assertEqual(
                telemetry["phase_tokens"]["unattributed"]["total_tokens"], 123
            )
            self.assertTrue(
                (repo / ".atrex_long_horizon/episodes/e0001/telemetry.brief.md").is_file()
            )

    def test_campaign_persists_default_structured_phase_telemetry(self) -> None:
        usage = lambda total: TokenUsage(total, 0, 0, 0, total, "exact")
        capabilities = AgentRuntimeCapabilities(True, True, True, True)
        observation = InvocationObservation(
            terminal_usage=usage(123),
            events=(
                NormalizedAgentEvent(0, "usage_delta", usage=usage(10)),
                NormalizedAgentEvent(
                    1, "phase_marker", phase="research", action="start", marker_id="r1"
                ),
                NormalizedAgentEvent(2, "usage_delta", usage=usage(20)),
                NormalizedAgentEvent(
                    3, "phase_marker", phase="research", action="end", marker_id="r2"
                ),
                NormalizedAgentEvent(
                    4, "phase_marker", phase="recording", action="start", marker_id="w1"
                ),
                NormalizedAgentEvent(5, "usage_delta", usage=usage(70)),
                NormalizedAgentEvent(
                    6, "phase_marker", phase="recording", action="end", marker_id="w2"
                ),
                NormalizedAgentEvent(7, "usage_delta", usage=usage(23)),
                NormalizedAgentEvent(8, "terminal_usage", usage=usage(123)),
            ),
            capabilities=capabilities,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            init_repo(repo)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                runner = CandidateRunner(5, invocations=(observation,))
                LongHorizonCampaign(
                    base_campaign=fake_base(repo),
                    max_episodes=1,
                    verifier=FixedVerifier(True),
                    session_runner=runner,
                    worktree_root=root / "worktrees",
                ).run()

            telemetry = json.loads(
                (repo / ".atrex_long_horizon/episodes/e0001/telemetry.summary.json").read_text()
            )
            tokens = telemetry["phase_tokens"]
            self.assertEqual(telemetry["control_tokens"], 123)
            self.assertEqual(tokens["terminal_usage"]["total_tokens"], 123)
            self.assertEqual(tokens["phases"]["research"]["usage"]["total_tokens"], 30)
            self.assertEqual(tokens["phases"]["recording"]["usage"]["total_tokens"], 70)
            self.assertEqual(tokens["orchestration"]["total_tokens"], 23)
            self.assertEqual(tokens["unattributed"]["total_tokens"], 0)
            self.assertEqual(tokens["accounted_coverage"], 1.0)
            self.assertEqual(tokens["reconciliation_status"], "reconciled")
            self.assertIn("ATREX_TELEMETRY_TRACE", runner.telemetry_environment)

    def test_telemetry_failure_does_not_block_candidate_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            init_repo(repo)
            patches = self._patches()
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                mock.patch(
                    "long_horizon.store.CampaignStore.archive_telemetry",
                    side_effect=OSError("telemetry disk unavailable"),
                ),
            ):
                reason = LongHorizonCampaign(
                    base_campaign=fake_base(repo),
                    max_episodes=1,
                    verifier=FixedVerifier(True),
                    session_runner=CandidateRunner(5),
                    worktree_root=root / "worktrees",
                ).run()

            self.assertEqual(reason, "max-episodes")
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 5\n")
            state = json.loads((repo / ".atrex_long_horizon/state.json").read_text())
            self.assertEqual(state["accepted"], 1)
            self.assertEqual(
                state["attempts"][0]["telemetry"]["measurement"], "unavailable"
            )
            self.assertEqual(
                state["attempts"][0]["telemetry"]["reason_codes"],
                ["telemetry_finalize_failed:OSError"],
            )

    def test_regressing_candidate_never_moves_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                base_campaign = fake_base(repo)
                LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    verifier=FixedVerifier(False),
                    session_runner=CandidateRunner(20),
                    worktree_root=root / "worktrees",
                ).run()
            self.assertNotEqual(git_head(repo), base)
            self.assertEqual(run_git(repo, "diff", base, "HEAD", "--", "kernel.py"), "")
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 10\n")
            rejected_memory = json.loads((repo / "memory/v1.json").read_text())
            self.assertEqual(rejected_memory["quality_gate"]["result"], "FAIL")
            state = json.loads((repo / ".atrex_long_horizon/state.json").read_text())
            self.assertEqual(state["rejected"], 1)
            base_campaign._notify_improvement.assert_not_called()
            base_campaign._notify_iteration.assert_called_once()
            self.assertFalse(base_campaign._notify_iteration.call_args.args[2])

    def test_blocked_retries_once_without_padding_a_non_bucket_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            init_repo(repo)
            seed_version(repo, 4, accepted_kernel=True)
            runner = BlockedRunner()
            with (
                mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None),
                mock.patch("long_horizon.main_adapter.link_episode_runtime", return_value=None),
                mock.patch(
                    "long_horizon.main_adapter.episode_directives",
                    return_value={
                        "hardware": "hardware",
                        "sandbox": "sandbox",
                        "evaluator": "evaluator",
                        "mode_policy": "policy",
                    },
                ),
                mock.patch(
                    "long_horizon.main_adapter.iteration_playbook",
                    return_value="current main playbook",
                ),
            ):
                base_campaign = fake_base(repo)
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    session_runner=runner,
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "blocked")
            self.assertEqual(runner.calls, 2)
            self.assertTrue((repo / "memory/v5.json").is_file())
            self.assertTrue((repo / "memory/v6.json").is_file())
            self.assertFalse((repo / "memory/v7.json").exists())
            state = json.loads((repo / ".atrex_long_horizon/state.json").read_text())
            self.assertEqual(state["episodes"], 2)
            self.assertEqual(state["blocked"], 2)
            self.assertTrue(state["attempts"][0]["blocked_retry_scheduled"])
            self.assertTrue(state["attempts"][1]["blocked_terminal"])

    def test_repeated_bucket_block_pads_to_main_barrier_without_changing_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            init_repo(repo)
            accepted_head = seed_version(repo, 4, accepted_kernel=True)
            runner = BlockedRunner()
            with (
                mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None),
                mock.patch("long_horizon.main_adapter.link_episode_runtime", return_value=None),
                mock.patch(
                    "long_horizon.main_adapter.episode_directives",
                    return_value={
                        "hardware": "hardware",
                        "sandbox": "sandbox",
                        "evaluator": "evaluator",
                        "mode_policy": "policy",
                    },
                ),
                mock.patch(
                    "long_horizon.main_adapter.iteration_playbook",
                    return_value="current main playbook",
                ),
            ):
                base_campaign = fake_base(repo)
                base_campaign.on_improvement = mock.Mock()
                base_campaign.on_iteration = mock.Mock()
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    session_runner=runner,
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "blocked")
            self.assertEqual(runner.calls, 2)
            self.assertEqual(
                run_git(repo, "diff", accepted_head, "HEAD", "--", "kernel.py"), ""
            )
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 9\n")
            for version in range(7, 11):
                memory = json.loads((repo / "memory" / f"v{version}.json").read_text())
                self.assertEqual(memory["quality_gate"]["result"], "FAIL")
                self.assertTrue(memory["long_horizon"]["terminal_padding"])
                self.assertEqual(memory["long_horizon"]["source_blocked_version"], 6)
            self.assertEqual(base_campaign._notify_improvement.call_count, 0)
            self.assertEqual(base_campaign._notify_iteration.call_count, 6)
            self.assertTrue(
                all(not call.args[2] for call in base_campaign._notify_iteration.call_args_list)
            )

    def test_repeated_bucket_block_without_an_accepted_kernel_does_not_pad(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            init_repo(repo)
            seed_version(repo, 4, accepted_kernel=False)
            runner = BlockedRunner()
            with (
                mock.patch("long_horizon.main_adapter.prepare_campaign", return_value=None),
                mock.patch("long_horizon.main_adapter.link_episode_runtime", return_value=None),
                mock.patch(
                    "long_horizon.main_adapter.episode_directives",
                    return_value={
                        "hardware": "hardware",
                        "sandbox": "sandbox",
                        "evaluator": "evaluator",
                        "mode_policy": "policy",
                    },
                ),
                mock.patch(
                    "long_horizon.main_adapter.iteration_playbook",
                    return_value="current main playbook",
                ),
            ):
                base_campaign = fake_base(repo)
                base_campaign.on_improvement = mock.Mock()
                base_campaign.on_iteration = mock.Mock()
                reason = LongHorizonCampaign(
                    base_campaign=base_campaign,
                    max_episodes=1,
                    session_runner=runner,
                    worktree_root=root / "worktrees",
                ).run()
            self.assertEqual(reason, "blocked")
            self.assertEqual(runner.calls, 2)
            self.assertFalse((repo / "memory/v7.json").exists())
            self.assertEqual((repo / "kernel.py").read_text(encoding="utf-8"), "VALUE = 10\n")

    def test_interrupted_episode_is_archived_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            store = CampaignStore(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            (episode.path / "kernel.py").write_text("VALUE = 7\n", encoding="utf-8")
            store.save_active(
                {
                    "episode": 1,
                    "base_commit": base,
                    "episode_branch": episode.branch,
                    "worktree": str(episode.path),
                    "phase": "exploring",
                }
            )
            campaign = LongHorizonCampaign(base_campaign=fake_base(repo))
            state = store.load_state()
            campaign._recover_interrupted(store, state)
            self.assertFalse(episode.path.exists())
            recovered = store.load_state()
            self.assertEqual(recovered.interrupted, 1)
            self.assertFalse(store.active_path.exists())

    def test_recovery_of_second_block_remains_terminal_without_a_third_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            repo = root / "campaign"
            base = init_repo(repo)
            store = CampaignStore(repo)
            state = store.load_state()
            state.episodes = 1
            state.blocked = 1
            state.attempts.append(
                {
                    "episode": 1,
                    "version": 1,
                    "status": "blocked",
                    "accepted": False,
                    "violation": None,
                    "blocked_retry_scheduled": True,
                    "blocked_terminal": False,
                }
            )
            store.save_state(state)
            memory = {
                "version": "v2",
                "quality_gate": {"result": "FAIL", "failure_reason": "blocked again"},
            }
            outcome_commit = record_episode_outcome(
                repo,
                base_commit=base,
                version=2,
                episode=2,
                status="blocked",
                memory_record=memory,
            )
            store.save_active(
                {
                    "episode": 2,
                    "memory_version": 2,
                    "base_commit": base,
                    "episode_branch": "atrex/long-e0002-test",
                    "worktree": str(root / "missing-worktree"),
                    "phase": "recording",
                    "terminal_status": "blocked",
                }
            )
            campaign = LongHorizonCampaign(base_campaign=fake_base(repo))
            recovered = store.load_state()
            campaign._recover_interrupted(store, recovered)
            recovered = store.load_state()
            self.assertEqual(git_head(repo), outcome_commit)
            self.assertEqual(recovered.episodes, 2)
            self.assertEqual(recovered.blocked, 2)
            self.assertEqual(len(recovered.attempts), 2)
            self.assertTrue(recovered.attempts[-1]["blocked_terminal"])
            self.assertEqual(recovered.attempts[-1]["blocked_retry_of_episode"], 1)
            self.assertFalse(campaign._blocked_retry_pending(recovered))


if __name__ == "__main__":
    unittest.main()
