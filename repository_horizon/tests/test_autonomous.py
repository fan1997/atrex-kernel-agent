from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from long_horizon.git_episode import git_head
from long_horizon.journal import append_experiment, finalize
from long_horizon.models import EpisodeHandoff, SessionResult, SupervisorState
from long_horizon.protocol import atomic_write_json
from orchestrator.agent_runtime.model import AgentRuntimeCapabilities, TokenUsage

from repository_horizon.campaign import RepositoryAutonomousCampaign
from repository_horizon.compat import assert_upstream_compatible
from repository_horizon.manifest import load_manifest
from repository_horizon.policy import install_repository_policy
from repository_horizon.prompt import MAX_PROMPT_BYTES, render_prompt
from repository_horizon.runtime import autonomous_environment, install_minimal_runtime
from repository_horizon.session import RepositorySessionRunner
from repository_horizon.tests.helpers import init_repo, run_git


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "recipes" / "fa4_fp8_paged_sm100.example.json"


class _NoopBaseline:
    def prepare(self, campaign) -> None:
        del campaign


class _PivotSession:
    def run(self, workspace, prompt, **kwargs) -> SessionResult:
        del workspace, prompt
        journal_path = Path(kwargs["handoff_path"]).with_name("journal.json")
        append_experiment(
            journal_path,
            {
                "name": "inspection",
                "hypothesis": "no safe improvement in the bounded test",
                "change": "none",
                "evidence": "fixture",
                "result": "direction exhausted",
                "decision": "pivot",
            },
        )
        finalize(
            journal_path,
            state="pivot",
            outcome={"summary": "fixture pivot", "next_directions": []},
        )
        handoff = EpisodeHandoff("pivot")
        atomic_write_json(kwargs["handoff_path"], handoff.as_dict())
        return SessionResult(
            exit_status=0,
            timed_out=False,
            tokens=123,
            session_id="fixture-session",
            resume_count=0,
            handoff=handoff,
        )


class AutonomousOverlayTests(unittest.TestCase):
    def test_upstream_compatibility_surface(self) -> None:
        assert_upstream_compatible()

    def test_prompt_is_bounded_and_has_no_guided_workflow(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            worktree = SimpleNamespace(
                path=workspace,
                base_commit="a" * 40,
                branch="atrex/test",
            )
            campaign = SimpleNamespace(
                sandbox_hardware="L20D",
                sandbox_profile="prod",
                sandbox_url="",
                notes="fixed five-shape FA4 comparison",
            )
            prompt = render_prompt(
                campaign=campaign,
                manifest=manifest,
                episode=1,
                worktree=worktree,
                journal_path=workspace / ".atrex_long_horizon" / "journal.json",
                handoff_path=workspace / ".atrex_long_horizon" / "handoff.json",
                state=SupervisorState(),
            )
        self.assertLessEqual(len(prompt.encode("utf-8")), MAX_PROMPT_BYTES)
        self.assertIn("repository_horizon.dev_eval submit", prompt)
        self.assertIn("stop the current Agent invocation immediately", prompt)
        lowered = prompt.lower()
        for forbidden in (
            "humanize",
            "gpu-kernel-research",
            "gpu-kernel-profiler",
            "wait_agent",
            "one optimization category",
            "must profile",
            "must research",
        ):
            self.assertNotIn(forbidden, lowered)

    def test_minimal_runtime_exposes_no_agent_assets(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "repo"
            init_repo(workspace)
            campaign = SimpleNamespace(workspace=workspace)
            install_minimal_runtime(campaign, workspace, manifest)
            self.assertFalse((workspace / ".agents" / "skills").exists())
            self.assertFalse((workspace / ".claude" / "agents").exists())
            self.assertFalse((workspace / ".qoder" / "agents").exists())
            self.assertTrue((workspace / ".repository_horizon_runtime").is_dir())

    def test_codex_multi_agent_is_disabled_by_default(self) -> None:
        settings = json.loads(autonomous_environment()["ATREX_CODEX_SESSION_SETTINGS"])
        self.assertIs(settings["features.multi_agent"], False)

    def test_pivot_does_not_advance_incumbent_or_create_version(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "kernel_opt_fixture"
            init_repo(workspace)
            install_repository_policy(workspace, manifest)
            (workspace / "memory").mkdir()
            (workspace / "memory" / "v2.json").write_text(
                '{"version":"v2","correctness":{"status":"PASS"}}\n',
                encoding="utf-8",
            )
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "v2 fixture")
            incumbent = git_head(workspace)
            campaign = SimpleNamespace(
                workspace=workspace,
                agent_cli="codex",
                sandbox_hardware="L20D",
                sandbox_profile="prod",
                sandbox_url="",
                notes="fixture",
            )
            controller = RepositoryAutonomousCampaign(
                base_campaign=campaign,
                manifest=manifest,
                baseline=_NoopBaseline(),
                verifier=SimpleNamespace(),
                max_episodes=1,
                session_runner=_PivotSession(),
            )
            controller.run()
            self.assertEqual(git_head(workspace), incumbent)
            self.assertFalse((workspace / "memory" / "v3.json").exists())
            state = json.loads(
                (workspace / ".atrex_long_horizon" / "state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["episodes"], 1)
            self.assertEqual(state["pivoted"], 1)
            self.assertEqual(state["attempts"][0]["prompt_bytes"] > 0, True)

    def test_repository_runner_suspends_for_evaluation_then_resumes_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            handoff_path = workspace / ".atrex_long_horizon" / "handoff.json"
            evaluation_path = (
                workspace
                / ".repository_horizon_runtime"
                / "evaluation_handoff.json"
            )
            calls: list[list[str]] = []

            def executor(command, current_workspace, timeout, environment):
                del timeout, environment
                self.assertEqual(current_workspace, workspace)
                calls.append(command)
                if len(calls) == 1:
                    atomic_write_json(
                        evaluation_path,
                        {
                            "status": "awaiting_evaluation",
                            "job_id": "job-1",
                            "pending_path": str(workspace / "pending.json"),
                        },
                    )
                else:
                    atomic_write_json(handoff_path, {"status": "pivot"})
                return "stream", "", 0, False

            waited: list[Path] = []

            def waiter(path: Path) -> str:
                waited.append(path)
                return "Evaluation complete. Read result.json and continue."

            capabilities = AgentRuntimeCapabilities(False, False, False)
            with (
                mock.patch(
                    "repository_horizon.session.main_adapter.session_environment",
                    return_value={},
                ),
                mock.patch(
                    "repository_horizon.session.main_adapter.fresh_session_command",
                    side_effect=lambda prompt, *_: ["fresh", prompt],
                ),
                mock.patch(
                    "repository_horizon.session.main_adapter.resume_session_command",
                    side_effect=lambda prompt, session_id, *_: [
                        "resume",
                        session_id,
                        prompt,
                    ],
                ),
                mock.patch(
                    "repository_horizon.session.main_adapter.session_id_from_stream",
                    return_value="thread-1",
                ),
                mock.patch(
                    "repository_horizon.session.main_adapter.normalize_stream",
                    return_value=((), TokenUsage.zero(), capabilities, ()),
                ),
                mock.patch(
                    "repository_horizon.session.main_adapter.tokens_from_stream",
                    return_value=0,
                ),
            ):
                result = RepositorySessionRunner(
                    evaluation_waiter=waiter,
                    executor=executor,
                    agent_cli="claude",
                ).run(
                    workspace,
                    "initial prompt",
                    timeout=60,
                    handoff_path=handoff_path,
                    handoff_resumes=1,
                    completion_check=lambda handoff: "",
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], ["fresh", "initial prompt"])
            self.assertEqual(calls[1][0:2], ["resume", "thread-1"])
            self.assertIn("Evaluation complete", calls[1][-1])
            self.assertEqual(waited, [evaluation_path])
            self.assertFalse(evaluation_path.exists())
            self.assertEqual(result.handoff, EpisodeHandoff("pivot"))
            self.assertEqual(result.resume_count, 1)


if __name__ == "__main__":
    unittest.main()
