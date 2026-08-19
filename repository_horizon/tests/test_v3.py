from __future__ import annotations

import json
import os
import runpy
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
    VerificationResult,
    VerificationRun,
)
from long_horizon.protocol import atomic_write_json
from orchestrator.campaign import Campaign
from orchestrator.constants import ATREX_PRIVATE_REFERENCE_ENV

from repository_horizon.campaign import (
    RepositoryCampaign,
    RepositoryHorizonCampaign,
)
from repository_horizon.compat import assert_upstream_compatible
from repository_horizon.config import EvaluationPolicy
from repository_horizon.manifest import load_manifest
from repository_horizon.prompt import MAX_PROMPT_BYTES, render_prompt
from repository_horizon.runtime import link_repository_runtime
from repository_horizon.staging import build_abba_stage
from repository_horizon.tests.helpers import init_repo, run_git
from repository_horizon.verifier import (
    _evaluation_runtime_root,
    _remove_private_stage_inputs,
    _require_complete_shape_coverage,
)
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "recipes" / "fa4_fp8_paged_sm100.example.json"


class RepositoryV3Tests(unittest.TestCase):
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
            self.assertEqual(
                campaign._production_kernel_violations(workspace), []
            )

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
            self.assertEqual(claimed[2]["files"]["runtime/shapes.json"], "private-shape")
            persisted = store.request(job["job_id"])
            self.assertTrue(persisted["_payload_redacted"])
            self.assertNotIn("files", persisted)
            store.close()

            workdir = root / "workdir"
            (workdir / "runtime").mkdir(parents=True)
            (workdir / "runtime" / "shapes.json").write_text("private")
            (workdir / "runtime" / "metadata.json").write_text("private")
            (workdir / "keep.txt").write_text("public")
            (workdir / "__atrex_workspace.tar.gz.b64.part000").write_text("bundle")
            gateway["_scrub_job_payload"](workdir)
            self.assertFalse((workdir / "runtime" / "shapes.json").exists())
            self.assertFalse((workdir / "runtime" / "metadata.json").exists())
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
