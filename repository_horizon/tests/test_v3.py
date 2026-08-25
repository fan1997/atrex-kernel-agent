from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from long_horizon.campaign import LongHorizonCampaign
from long_horizon.git_episode import EpisodeWorktree, git_head
from long_horizon.journal import append_experiment, finalize
from long_horizon.models import (
    EpisodeHandoff,
    SessionResult,
    SupervisorState,
    VerificationResult,
    VerificationRun,
)
from long_horizon.protocol import atomic_write_json
from long_horizon.store import CampaignStore
from orchestrator.campaign import Campaign
from orchestrator.constants import ATREX_PRIVATE_REFERENCE_ENV

from repository_horizon.campaign import (
    RepositoryCampaign,
    RepositoryHorizonCampaign,
)
from repository_horizon.cli import _validated_campaign_name
from repository_horizon.compat import assert_upstream_compatible
from repository_horizon.config import EvaluationPolicy
from repository_horizon.dev_eval import _verifier as make_dev_verifier
from repository_horizon.manifest import load_manifest
from repository_horizon.prompt import MAX_PROMPT_BYTES, render_prompt
from repository_horizon.runtime import link_repository_runtime
from repository_horizon.staging import build_abba_stage
from repository_horizon.strategy import ArchitectureStrategyState
from repository_horizon.tests.helpers import init_repo, run_git
from repository_horizon.verifier import (
    RepositoryPhaseValidator,
    _evaluation_runtime_root,
    _remove_private_stage_inputs,
    _require_complete_shape_coverage,
    has_measured_v0,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "recipes" / "fa4_fp8_paged_sm100.example.json"


class RepositoryV3Tests(unittest.TestCase):
    def test_interrupted_architecture_recovery_reanchors_wip_to_outcome_head(
        self,
    ) -> None:
        manifest = load_manifest(MANIFEST)
        relative = f"{manifest.editable_workspace_roots[0]}/kernel.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            init_repo(workspace)
            source = workspace / relative
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "repository baseline")
            base = git_head(workspace)

            trial = EpisodeWorktree.create(
                workspace, 99, base, root=root / "trial-worktrees"
            )
            trial_source = trial.path / relative
            trial_source.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(trial.path, "add", relative)
            run_git(trial.path, "commit", "-m", "architecture checkpoint")
            checkpoint = git_head(trial.path)
            patch_bytes = subprocess.run(
                ["git", "diff", "--binary", base, checkpoint, "--"],
                cwd=str(trial.path),
                check=True,
                capture_output=True,
            ).stdout

            controller = RepositoryHorizonCampaign(
                base_campaign=SimpleNamespace(workspace=workspace),
                manifest=manifest,
            )
            strategy = ArchitectureStrategyState(
                mode="architecture",
                commitment_remaining=2,
                wip_base_commit=base,
                wip_source_commit=checkpoint,
                wip_patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
            )
            controller.strategy_store.save(strategy)
            controller.strategy_store.wip_patch_path.write_bytes(patch_bytes)

            memory = workspace / "memory"
            memory.mkdir()
            (memory / "v1.json").write_text("{}\n", encoding="utf-8")
            run_git(workspace, "add", "memory/v1.json")
            run_git(
                workspace,
                "commit",
                "-m",
                "v1: long-horizon episode 1 interrupted",
            )
            outcome_head = git_head(workspace)
            store = CampaignStore(workspace)
            store.save_active(
                {
                    "episode": 1,
                    "memory_version": 1,
                    "base_commit": base,
                    "episode_branch": "atrex/long-e0001-recovery",
                    "phase": "recorded",
                    "terminal_status": "interrupted",
                }
            )

            state = SupervisorState()
            controller._recover_interrupted(store, state)

            saved = controller.strategy_store.load()
            self.assertEqual(saved.wip_base_commit, outcome_head)
            self.assertEqual(saved.wip_source_commit, checkpoint)
            self.assertEqual(state.episodes, 1)
            self.assertEqual(state.interrupted, 1)
            self.assertIsNone(store.load_active())
            next_worktree = EpisodeWorktree.create(
                workspace, 2, outcome_head, root=root / "worktrees"
            )
            self.assertTrue(
                controller.strategy_store.apply_wip(next_worktree.path, saved)
            )
            self.assertEqual(
                (next_worktree.path / relative).read_text(encoding="utf-8"),
                "VALUE = 2\n",
            )
            next_worktree.remove(workspace)
            trial.remove(workspace)

    def test_recovered_wip_reanchor_rejects_editable_source_change(self) -> None:
        manifest = load_manifest(MANIFEST)
        relative = f"{manifest.editable_workspace_roots[0]}/kernel.py"
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            init_repo(workspace)
            source = workspace / relative
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "repository baseline")
            base = git_head(workspace)

            controller = RepositoryHorizonCampaign(
                base_campaign=SimpleNamespace(workspace=workspace),
                manifest=manifest,
            )
            controller.strategy_store.save(
                ArchitectureStrategyState(
                    mode="architecture",
                    wip_base_commit=base,
                )
            )
            controller.strategy_store.wip_patch_path.write_text(
                "placeholder\n", encoding="utf-8"
            )
            source.write_text("VALUE = 3\n", encoding="utf-8")
            run_git(workspace, "add", relative)
            run_git(workspace, "commit", "-m", "source changed")

            with self.assertRaisesRegex(
                RuntimeError, "editable repository roots changed"
            ):
                controller._after_recovered_outcome_recorded(
                    episode=1,
                    base_commit=base,
                    outcome_commit=git_head(workspace),
                )
            self.assertEqual(controller.strategy_store.load().wip_base_commit, base)

    def test_pre_agent_setup_interruption_does_not_consume_an_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            base = init_repo(workspace)
            worktree = EpisodeWorktree.create(
                workspace, 1, base, root=root / "worktrees"
            )
            store = CampaignStore(workspace)
            store.save_active(
                {
                    "episode": 1,
                    "memory_version": 1,
                    "base_commit": base,
                    "episode_branch": worktree.branch,
                    "worktree": str(worktree.path),
                    "phase": "exploring",
                }
            )
            state = SupervisorState()
            controller = LongHorizonCampaign(
                base_campaign=SimpleNamespace(workspace=workspace)
            )

            controller._recover_interrupted(store, state)

            self.assertEqual(state.episodes, 0)
            self.assertEqual(state.interrupted, 0)
            self.assertEqual(state.attempts, [])
            self.assertIsNone(store.load_active())
            self.assertFalse(worktree.path.exists())
            self.assertEqual(git_head(workspace), base)

    def test_rejected_architecture_candidate_rebases_wip_to_outcome_head(
        self,
    ) -> None:
        manifest = load_manifest(MANIFEST)
        relative = f"{manifest.editable_workspace_roots[0]}/kernel.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            init_repo(workspace)
            source = workspace / relative
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "repository baseline")
            base = git_head(workspace)
            worktree = EpisodeWorktree.create(
                workspace, 1, base, root=root / "worktrees"
            )
            candidate_source = worktree.path / relative
            candidate_source.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(worktree.path, "add", relative)
            run_git(worktree.path, "commit", "-m", "architecture candidate")
            candidate = git_head(worktree.path)

            memory = workspace / "memory"
            memory.mkdir()
            (memory / "v1.json").write_text("{}\n", encoding="utf-8")
            run_git(workspace, "add", "memory/v1.json")
            run_git(workspace, "commit", "-m", "v1: rejected outcome")
            outcome_head = git_head(workspace)

            controller = RepositoryHorizonCampaign(
                base_campaign=SimpleNamespace(workspace=workspace),
                manifest=manifest,
            )
            strategy = ArchitectureStrategyState(
                mode="architecture", commitment_remaining=2
            )
            controller.strategy_store.save(strategy)
            result = SimpleNamespace(
                handoff=EpisodeHandoff("candidate_ready", candidate)
            )
            controller._after_episode_recorded(
                worktree=worktree,
                state=SupervisorState(),
                result=result,
                journal={
                    "outcome": {
                        "architecture": {
                            "direction_id": "split-kv",
                            "thesis": "parallel KV work",
                            "disposition": "promote",
                        }
                    }
                },
                attempt={},
                accepted=False,
            )

            saved = controller.strategy_store.load()
            self.assertEqual(saved.wip_base_commit, outcome_head)
            self.assertEqual(saved.wip_source_commit, candidate)
            next_worktree = EpisodeWorktree.create(
                workspace, 2, outcome_head, root=root / "worktrees"
            )
            self.assertTrue(
                controller.strategy_store.apply_wip(next_worktree.path, saved)
            )
            self.assertEqual(
                (next_worktree.path / relative).read_text(encoding="utf-8"),
                "VALUE = 2\n",
            )
            next_worktree.remove(workspace)
            worktree.remove(workspace)

    def test_public_campaign_name_can_hide_private_operator_basename(self) -> None:
        private_operator = Path("/private/evaluator/historical-winner-label")
        self.assertEqual(
            _validated_campaign_name("fa4-hd256-private", private_operator),
            "fa4-hd256-private",
        )
        self.assertEqual(
            _validated_campaign_name("", private_operator),
            "historical-winner-label",
        )

    def test_public_campaign_name_rejects_path_syntax(self) -> None:
        for value in ("../escape", "/absolute", "has space", ""):
            with self.subTest(value=value):
                if not value:
                    continue
                with self.assertRaises(ValueError):
                    _validated_campaign_name(value, Path("/private/operator"))

    def test_measured_v0_keeps_normal_abba_after_interrupted_memory(self) -> None:
        normal = object()
        bringup = object()
        validator = RepositoryPhaseValidator(normal, bringup)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            memory = workspace / "memory"
            memory.mkdir()
            (memory / "v0.json").write_text(
                json.dumps({"quality_gate": {"result": "PASS"}}) + "\n",
                encoding="utf-8",
            )
            (memory / "v1.json").write_text(
                json.dumps(
                    {
                        "quality_gate": {"result": "FAIL"},
                        "long_horizon": {"status": "interrupted"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(has_measured_v0(workspace))
            self.assertIs(validator._validator(workspace), normal)

            (memory / "v0.json").write_text(
                json.dumps({"quality_gate": {"result": "BRINGUP_REQUIRED"}}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(has_measured_v0(workspace))
            self.assertIs(validator._validator(workspace), bringup)

    def test_dev_eval_uses_manifest_abba_after_measured_v0(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            base_commit = init_repo(workspace)
            memory = workspace / "memory"
            memory.mkdir()
            (memory / "v0.json").write_text(
                json.dumps({"quality_gate": {"result": "PASS"}}) + "\n",
                encoding="utf-8",
            )
            (memory / "v1.json").write_text(
                json.dumps(
                    {
                        "quality_gate": {"result": "FAIL"},
                        "long_horizon": {"status": "interrupted"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (workspace / "source_manifest.json").write_text(
                MANIFEST.read_text(encoding="utf-8"), encoding="utf-8"
            )
            (workspace / "source.lock.json").write_text(
                json.dumps({"atrex_bench_root": str(workspace)}) + "\n",
                encoding="utf-8",
            )
            journal_dir = workspace / ".atrex_long_horizon"
            journal_dir.mkdir()
            (journal_dir / "journal.json").write_text(
                json.dumps({"base_commit": base_commit}) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(
                hardware="local",
                profile="",
                url="http://127.0.0.1:8004",
                backend="agate",
                wait_mode="inline",
                wait_timeout=14_400,
                agent_result_max_bytes=16 * 1024,
            )
            verifier, resolved_base, candidate, paths = make_dev_verifier(
                workspace, args
            )
            manifest = load_manifest(MANIFEST)
            self.assertEqual(resolved_base, base_commit)
            self.assertEqual(candidate, base_commit)
            self.assertIn("memory/v0.json", paths)
            self.assertFalse(verifier.candidate_only)
            self.assertEqual(verifier.repeats, manifest.measurement.repeats)
            self.assertEqual(verifier.min_improvement_pct, 1.0)

    def test_v3_is_a_thin_current_main_extension(self) -> None:
        assert_upstream_compatible()
        self.assertTrue(issubclass(RepositoryHorizonCampaign, LongHorizonCampaign))
        self.assertIs(
            RepositoryHorizonCampaign.run,
            LongHorizonCampaign.run,
            "v3 must not fork main's supervisor loop",
        )

    def test_prompt_keeps_main_contract_without_wiki_or_mandatory_gen_plan(
        self,
    ) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            worktree = SimpleNamespace(
                path=workspace,
                base_commit="a" * 40,
                branch="atrex/test",
            )
            campaign = RepositoryCampaign(
                name="fixture",
                kernel_demo=str(workspace / "reference.py"),
                platform="B300",
                framework="CuteDSL",
                sandbox_hardware="L20D",
                sandbox_profile="prod",
                optimization_mode="production",
                framework_baseline="never",
                agent_cli="codex",
                repository_manifest=manifest,
            )
            prompt = render_prompt(
                campaign=campaign,
                manifest=manifest,
                episode=3,
                version=4,
                worktree=worktree,
                journal_path=workspace / ".atrex_long_horizon" / "journal.json",
                handoff_path=workspace / ".atrex_long_horizon" / "handoff.json",
                live_memory_path=workspace / "memory" / "live.json",
                evaluation_policy=EvaluationPolicy(wait_mode="inline"),
            )
        self.assertLessEqual(len(prompt.encode("utf-8")), MAX_PROMPT_BYTES)
        self.assertIn("supervisor, verification, canonical memory, recovery", prompt)
        self.assertIn("gen-plan` is available but optional", prompt)
        self.assertIn("GPU Wiki and KernelWiki are intentionally not installed", prompt)
        self.assertIn("profiles/<episode>/public_driver.py", prompt)
        self.assertNotIn("repository_horizon.dev_eval submit", prompt)
        self.assertNotIn("repository_horizon.dev_eval profile", prompt)
        self.assertIn("Do not invoke\n`repository_horizon.dev_eval`", prompt)
        self.assertIn("Mandatory pre-bring-up repository reconnaissance", prompt)
        self.assertIn("repository_horizon.reconnaissance seal", prompt)
        self.assertLess(
            prompt.index("Mandatory pre-bring-up repository reconnaissance"),
            prompt.index("## Execution boundary"),
        )
        self.assertNotIn("<PLAN_GENERATOR>", prompt)
        self.assertNotIn("execute its loop", prompt)
        self.assertNotIn("Invoke the `$gen-plan`", prompt)

    def test_repository_agent_environment_excludes_private_evaluator(self) -> None:
        campaign = RepositoryCampaign(
            name="fixture",
            kernel_demo="reference.py",
            platform="B300",
            framework="CuteDSL",
        )
        with patch.object(
            Campaign,
            "agent_environment",
            return_value={ATREX_PRIVATE_REFERENCE_ENV: "/private", "KEEP": "yes"},
        ):
            environment = campaign.agent_environment()
        self.assertEqual(environment, {"KEEP": "yes"})

    def test_repository_resume_uses_repository_candidate_policy(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            (workspace / "memory").mkdir()
            (workspace / "memory" / "v0.json").write_text("{}\n")
            campaign = RepositoryCampaign(
                name="fixture",
                kernel_demo=str(workspace / "reference.py"),
                platform="B300",
                framework="CuteDSL",
                work_dir=str(workspace.parent),
                repository_manifest=manifest,
            )
            self.assertEqual(campaign._production_kernel_violations(workspace), [])

    def test_public_profile_never_materializes_a_private_case_without_capability(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop(ATREX_PRIVATE_REFERENCE_ENV, None)
            sandbox = runpy.run_path(str(ROOT.parent / "tools" / "sandbox.py"))
            self.assertIsNone(sandbox["_private_profile_case"](Path(temp), []))

    def test_local_gateway_redacts_and_scrubs_staged_payloads(self) -> None:
        gateway = runpy.run_path(str(ROOT.parent / "tools" / "local_gateway.py"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            store = gateway["JobStore"](root / "jobs.db")
            job, created = store.create(
                "dev",
                {
                    "spec": {"target_hardware": ["local"]},
                    "files": {"runtime/shapes.json": "private-shape"},
                    "command": "python repo_abba.py",
                },
                "trace",
            )
            self.assertTrue(created)
            claimed = store.claim_next()
            self.assertEqual(
                claimed[2]["files"]["runtime/shapes.json"], "private-shape"
            )
            persisted = store.request(job["job_id"])
            self.assertTrue(persisted["_payload_redacted"])
            self.assertNotIn("files", persisted)
            store.close()

            workdir = root / "workdir"
            (workdir / "runtime").mkdir(parents=True)
            (workdir / "runtime" / "shapes.json").write_text("private")
            (workdir / "runtime" / "metadata.json").write_text("private")
            (workdir / ".runs" / "00_candidate_0").mkdir(parents=True)
            (workdir / ".runs" / "00_candidate_0" / "shapes.json").write_text("private")
            (workdir / "reference").mkdir()
            (workdir / "reference" / "roofline.json").write_text("private")
            (workdir / "keep.txt").write_text("public")
            (workdir / "__atrex_workspace.tar.gz.b64.part000").write_text("bundle")
            gateway["_scrub_job_payload"](workdir)
            self.assertFalse((workdir / "runtime" / "shapes.json").exists())
            self.assertFalse((workdir / "runtime" / "metadata.json").exists())
            self.assertFalse(
                (workdir / ".runs" / "00_candidate_0" / "shapes.json").exists()
            )
            self.assertFalse((workdir / "reference" / "roofline.json").exists())
            self.assertFalse(
                (workdir / "__atrex_workspace.tar.gz.b64.part000").exists()
            )
            self.assertTrue((workdir / "keep.txt").is_file())

    def test_runtime_matches_main_assets_except_wiki(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            init_repo(workspace)
            campaign = SimpleNamespace(
                workspace=workspace,
                optimization_mode="production",
                framework="CuteDSL",
                atrex_bench_root="",
                repository_capabilities={},
            )
            link_repository_runtime(campaign, workspace, manifest)
            self.assertTrue((workspace / "tools").is_symlink())
            self.assertTrue((workspace / "skills").is_symlink())
            self.assertTrue(
                (workspace / ".agents" / "skills" / "gen-plan").is_symlink()
            )
            self.assertFalse((workspace / "gpu-wiki").exists())
            self.assertFalse((workspace / ".agents" / "skills" / "KernelWiki").exists())

    def test_repository_candidate_uses_main_git_checks_and_manifest_roots(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            init_repo(workspace)
            editable = workspace / manifest.editable_workspace_roots[0]
            editable.mkdir(parents=True)
            (editable / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "repository baseline")
            base = git_head(workspace)
            (editable / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
            run_git(workspace, "add", str(editable.relative_to(workspace)))
            run_git(workspace, "commit", "-m", "candidate")
            branch = run_git(workspace, "branch", "--show-current")
            worktree = EpisodeWorktree(1, base, branch, workspace)
            controller = RepositoryHorizonCampaign(
                base_campaign=SimpleNamespace(workspace=workspace),
                manifest=manifest,
            )
            violation, paths = controller._validate_candidate(
                worktree, git_head(workspace)
            )
            self.assertEqual(violation, "")
            self.assertEqual(
                paths, [f"{manifest.editable_workspace_roots[0]}/kernel.py"]
            )

    def test_private_shapes_are_staged_but_never_added_to_workspace(self) -> None:
        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            init_repo(workspace)
            private = root / "private"
            private.mkdir()
            (private / "shapes.json").write_text(
                json.dumps({"s0": {"init_kwargs": {}, "input_kwargs": {}}}),
                encoding="utf-8",
            )
            (private / "metadata.json").write_text("{}\n", encoding="utf-8")
            atrex = root / "atrex"
            (atrex / "src" / "atrex_bench").mkdir(parents=True)
            (atrex / "src" / "atrex_bench" / "__init__.py").write_text("")
            (atrex / "scripts").mkdir()
            (atrex / "scripts" / "run_eval.py").write_text("# fixture\n")
            stage = root / "stage"
            build_abba_stage(
                workspace,
                base_commit=git_head(workspace),
                candidate_commit=git_head(workspace),
                changed_paths=[],
                manifest=manifest,
                atrex_bench_root=atrex,
                destination=stage,
                schedule=[{"revision": "candidate", "repeat": 0}],
                per_run_timeout=1,
                private_reference_dir=private,
            )
            self.assertFalse((workspace / "shapes.json").exists())
            self.assertTrue((stage / "runtime" / "shapes.json").is_file())
            self.assertTrue((stage / "runtime" / "metadata.json").is_file())

    def test_private_verifier_stage_is_out_of_band_and_scrubbed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            runtime_root = _evaluation_runtime_root(workspace)
            with self.assertRaises(ValueError):
                runtime_root.relative_to(workspace)
            stage = runtime_root / "fixture" / "staging"
            (stage / "runtime").mkdir(parents=True)
            for name in ("shapes.json", "metadata.json", "roofline.json"):
                (stage / "runtime" / name).write_text("{}\n", encoding="utf-8")
            _remove_private_stage_inputs(SimpleNamespace(stage=str(stage)))
            for name in ("shapes.json", "metadata.json", "roofline.json"):
                self.assertFalse((stage / "runtime" / name).exists())

    def test_every_abba_run_must_cover_every_private_shape(self) -> None:
        verification = VerificationResult(
            "PASS",
            1.0,
            2.0,
            50.0,
            runs=[
                VerificationRun(
                    "candidate",
                    0,
                    0,
                    {"all_pass": True, "latency_us_by_shape": {"a": 1.0}},
                ),
                VerificationRun(
                    "incumbent",
                    0,
                    0,
                    {
                        "all_pass": True,
                        "latency_us_by_shape": {"a": 2.0, "b": 2.0},
                    },
                ),
            ],
        )
        checked = _require_complete_shape_coverage(verification, ("a", "b"))
        self.assertFalse(checked.passed)
        self.assertIn("candidate[0]=1/2", checked.error)

    def test_pivot_uses_main_canonical_memory_semantics(self) -> None:
        class NoopBaseline:
            def prepare(self, campaign) -> None:
                del campaign

        class PivotSession:
            def run(self, workspace, prompt, **kwargs) -> SessionResult:
                del workspace, prompt
                journal = Path(kwargs["handoff_path"]).with_name("journal.json")
                append_experiment(
                    journal,
                    {
                        "name": "fixture",
                        "hypothesis": "direction is exhausted",
                        "change": "none",
                        "evidence": "unit test",
                        "result": "no candidate",
                        "decision": "pivot",
                    },
                )
                finalize(
                    journal,
                    state="pivot",
                    outcome={"summary": "fixture pivot", "next_directions": []},
                )
                handoff = EpisodeHandoff("pivot")
                atomic_write_json(kwargs["handoff_path"], handoff.as_dict())
                return SessionResult(0, False, 7, "fixture-session", 0, handoff)

        manifest = load_manifest(MANIFEST)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = RepositoryCampaign(
                name="fixture",
                kernel_demo=str(root / "operator" / "reference.py"),
                platform="B300",
                framework="CuteDSL",
                work_dir=str(root),
                sandbox_hardware="L20D",
                optimization_mode="production",
                framework_baseline="never",
                agent_cli="codex",
                repository_manifest=manifest,
            )
            workspace = campaign.workspace
            init_repo(workspace)
            (workspace / "memory").mkdir()
            (workspace / "memory" / "v0.json").write_text(
                json.dumps(
                    {
                        "version": "v0",
                        "performance": {
                            "latency_us": 10.0,
                            "latency_us_geomean": 10.0,
                            "latency_us_by_shape": {"fixture": 10.0},
                        },
                        "correctness": {"status": "PASS"},
                        "quality_gate": {"result": "PASS"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            link_repository_runtime(campaign, workspace, manifest)
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "measured repository v0")
            initial = git_head(workspace)
            controller = RepositoryHorizonCampaign(
                base_campaign=campaign,
                manifest=manifest,
                baseline=NoopBaseline(),
                max_version=1,
                session_runner=PivotSession(),
                evaluation_policy=EvaluationPolicy(wait_mode="inline"),
            )
            reason = controller.run()
            self.assertEqual(reason, "budget: max-iters")
            self.assertNotEqual(git_head(workspace), initial)
            memory = json.loads(
                (workspace / "memory" / "v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(memory["long_horizon"]["status"], "pivot")
            self.assertEqual(memory["quality_gate"]["result"], "FAIL")
            state = json.loads(
                (workspace / ".atrex_long_horizon" / "state.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["pivoted"], 1)
            self.assertEqual(state["episodes"], 1)

    def test_repository_candidate_uses_main_verification_and_promotion(self) -> None:
        class NoopBaseline:
            def prepare(self, campaign) -> None:
                del campaign

        class CandidateSession:
            def __init__(self, relative: str):
                self.relative = relative

            def run(self, workspace, prompt, **kwargs) -> SessionResult:
                del prompt
                target = workspace / self.relative
                target.write_text("VALUE = 2\n", encoding="utf-8")
                run_git(workspace, "add", self.relative)
                run_git(workspace, "commit", "-m", "repository candidate")
                candidate = git_head(workspace)
                journal = Path(kwargs["handoff_path"]).with_name("journal.json")
                append_experiment(
                    journal,
                    {
                        "name": "candidate",
                        "hypothesis": "repository edit improves latency",
                        "change": self.relative,
                        "evidence": "fixture benchmark",
                        "result": "candidate is faster",
                        "decision": "continue",
                    },
                )
                finalize(
                    journal,
                    state="candidate_ready",
                    candidate_commit=candidate,
                    outcome={"summary": "verified fixture", "next_directions": []},
                )
                handoff = EpisodeHandoff("candidate_ready", candidate)
                atomic_write_json(kwargs["handoff_path"], handoff.as_dict())
                return SessionResult(0, False, 11, "fixture-session", 0, handoff)

        class PassingVerifier:
            def verify(self, workspace, **kwargs) -> VerificationResult:
                del workspace, kwargs
                return VerificationResult(
                    "PASS",
                    5.0,
                    10.0,
                    50.0,
                    runs=[
                        VerificationRun(
                            "incumbent",
                            0,
                            0,
                            {
                                "all_pass": True,
                                "latency_us_geomean": 10.0,
                                "latency_us_by_shape": {"fixture": 10.0},
                            },
                        ),
                        VerificationRun(
                            "candidate",
                            0,
                            0,
                            {
                                "all_pass": True,
                                "latency_us_geomean": 5.0,
                                "latency_us_by_shape": {"fixture": 5.0},
                            },
                        ),
                    ],
                )

        manifest = load_manifest(MANIFEST)
        relative = f"{manifest.editable_workspace_roots[0]}/kernel.py"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            campaign = RepositoryCampaign(
                name="fixture",
                kernel_demo=str(root / "operator" / "reference.py"),
                platform="B300",
                framework="CuteDSL",
                work_dir=str(root),
                sandbox_hardware="L20D",
                optimization_mode="production",
                framework_baseline="never",
                agent_cli="codex",
                repository_manifest=manifest,
            )
            workspace = campaign.workspace
            init_repo(workspace)
            source = workspace / relative
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            (workspace / "memory").mkdir()
            (workspace / "memory" / "v0.json").write_text(
                json.dumps(
                    {
                        "version": "v0",
                        "performance": {
                            "latency_us": 10.0,
                            "latency_us_geomean": 10.0,
                            "latency_us_by_shape": {"fixture": 10.0},
                        },
                        "correctness": {"status": "PASS"},
                        "quality_gate": {"result": "PASS"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            link_repository_runtime(campaign, workspace, manifest)
            run_git(workspace, "add", ".")
            run_git(workspace, "commit", "-m", "measured repository v0")
            controller = RepositoryHorizonCampaign(
                base_campaign=campaign,
                manifest=manifest,
                baseline=NoopBaseline(),
                verifier=PassingVerifier(),
                max_version=1,
                session_runner=CandidateSession(relative),
                evaluation_policy=EvaluationPolicy(wait_mode="inline"),
            )
            reason = controller.run()
            self.assertEqual(reason, "budget: max-iters")
            self.assertEqual(source.read_text(encoding="utf-8"), "VALUE = 2\n")
            memory = json.loads(
                (workspace / "memory" / "v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(memory["quality_gate"]["result"], "PASS")
            self.assertEqual(
                memory["performance"]["latency_us_by_shape"], {"fixture": 5.0}
            )
            self.assertEqual(memory["repository_source"]["revision"], manifest.revision)


if __name__ == "__main__":
    unittest.main()
