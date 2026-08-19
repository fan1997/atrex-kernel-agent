from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from long_horizon.git_episode import (
    EpisodeWorktree,
    changed_paths,
    git_head,
    promote_candidate,
)
from long_horizon.models import VerificationResult, VerificationRun
from repository_horizon.tests.helpers import init_repo, run_git
from long_horizon.verifier import verification_schedule
from repository_horizon.candidate import RepositoryCandidateContract
from repository_horizon.corpus import CORPUS_RELATIVE, validate_source_corpus
from repository_horizon.dev_eval import _require_reconnaissance
from repository_horizon.baseline import RepositoryBaselineManager
from repository_horizon.manifest import RuntimeSupportWheel, load_manifest
from repository_horizon.policy import install_repository_policy
from repository_horizon.repository_profile import _profile_target_python
from repository_horizon.reconnaissance import (
    REPORT_RELATIVE,
    SEAL_RELATIVE,
    reconnaissance_gate_violations,
    seal_reconnaissance,
)
from repository_horizon.seed import (
    _archive_paths_with_package_boundaries,
    seed_workspace,
)
from repository_horizon.staging import build_abba_stage
from repository_horizon.support_wheel import extract_support_wheel
from repository_horizon.transport import (
    _payload,
    get_agate_job,
    submit_agate_dev,
    submit_local_dev,
)


def make_source(root: Path) -> tuple[Path, str]:
    source = root / "source"
    (source / "flash_attn" / "cute").mkdir(parents=True)
    (source / "flash_attn" / "__init__.py").write_text("", encoding="utf-8")
    (source / "flash_attn" / "cute" / "kernel.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    subprocess.run(["git", "init"], cwd=source, check=True, capture_output=True)
    run_git(source, "add", ".")
    run_git(source, "commit", "-m", "source")
    return source, git_head(source)


def make_manifest(
    root: Path,
    revision: str,
    runtime_support: list[dict] | None = None,
    *,
    repository_search: dict | None = None,
    bringup: dict | None = None,
) -> Path:
    (root / "adapter.py").write_text("VALUE = 0\n", encoding="utf-8")
    path = root / "repository.json"
    payload = {
        "schema_version": 1,
        "name": "fixture",
        "adapter": "adapter.py",
        "source": {
            "name": "flash_attention",
            "revision": revision,
            "archive_paths": ["flash_attn"],
            "package_root": ".",
        },
        "editable_roots": ["flash_attn/cute"],
        "measurement": {"warmup": 1, "timed_runs": 1, "repeats": 2},
    }
    if runtime_support is not None:
        payload["runtime_support"] = runtime_support
    if repository_search is not None:
        payload["repository_search"] = repository_search
    if bringup is not None:
        payload["bringup"] = bringup
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_support_wheel(root: Path, *, version: str = "0.5.3") -> Path:
    wheel = root / f"quack_kernels-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.1\nName: quack-kernels\nVersion: {version}\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("quack/__init__.py", "raise RuntimeError('eager import')\n")
        archive.writestr("quack/core.py", "from quack import helper\nVALUE = 1\n")
        archive.writestr("quack/helper.py", "VALUE = 2\n")
        archive.writestr(f"quack_kernels-{version}.dist-info/METADATA", metadata)
        archive.writestr(
            f"quack_kernels-{version}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    return wheel


def support_manifest() -> list[dict]:
    return [
        {
            "distribution": "quack-kernels",
            "version": "0.5.3",
            "package": "quack",
            "members": ["quack/core.py", "quack/helper.py"],
            "dist_info_members": ["METADATA", "WHEEL"],
            "generate_minimal_init": True,
        }
    ]


class RepositoryContractTests(unittest.TestCase):
    def test_repository_policy_remains_clean_after_generic_runtime_relink(self) -> None:
        from orchestrator.optimization_policy import install_workspace_policy

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, revision = make_source(root)
            manifest = load_manifest(make_manifest(root, revision))
            workspace = root / "workspace"
            install_workspace_policy(workspace, "production", "CuteDSL")
            install_repository_policy(workspace, manifest)
            first = (workspace / "CLAUDE.md").read_bytes()
            install_workspace_policy(workspace, "production", "CuteDSL")
            install_repository_policy(workspace, manifest)
            self.assertEqual((workspace / "CLAUDE.md").read_bytes(), first)
            self.assertIn(b"locked `flash_attention` source", first)

    def test_legacy_manifest_defaults_preserve_snapshot_v0_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, revision = make_source(root)
            manifest = load_manifest(make_manifest(root, revision))
            self.assertEqual(manifest.repository_search.mode, "snapshot")
            self.assertFalse(manifest.repository_search.require_report)
            self.assertEqual(manifest.bringup.mode, "disabled")

    def test_source_change_is_candidate_and_adapter_is_protected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, revision = make_source(root)
            manifest = load_manifest(make_manifest(root, revision))
            contract = RepositoryCandidateContract(manifest)
            source_path = "vendor/flash_attention/flash_attn/cute/kernel.py"
            self.assertEqual(contract.validate_changed_paths([source_path]), "")
            self.assertIn("protected", contract.validate_changed_paths(["kernel.py"]))
            self.assertIn(
                "protected",
                contract.validate_changed_paths(["vendor_support/quack/core.py"]),
            )
            self.assertIn(
                "undeclared",
                contract.validate_changed_paths(["vendor/flash_attention/setup.py"]),
            )

    def test_pre_v0_candidate_requires_a_repository_search_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, revision = make_source(root)
            manifest = load_manifest(
                make_manifest(
                    root,
                    revision,
                    repository_search={"mode": "replay_strict"},
                    bringup={"mode": "auto"},
                )
            )
            contract = RepositoryCandidateContract(manifest)
            workspace = root / "workspace"
            workspace.mkdir()
            self.assertIn(
                "requires plans/repository_search.json",
                contract.workspace_violations(None, workspace)[0],
            )
            (workspace / "plans").mkdir()
            (workspace / "plans" / "repository_search.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "source_revision": revision,
                        "queries": ["HD256 seqused_k page table"],
                        "candidates": [
                            {
                                "commit": revision,
                                "path": "flash_attn/cute/kernel.py",
                                "mechanism": "reuse the existing tiled kernel structure",
                                "workload_relevance": "the public contract uses this path",
                                "transfer_gap": "paged addressing is not implemented",
                                "risks": "layout assumptions may not transfer",
                            }
                        ],
                        "selected": {
                            "commit": revision,
                            "path": "flash_attn/cute/kernel.py",
                            "rationale": "closest bounded implementation evidence",
                            "gap": "paged dispatch",
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (
                mock.patch(
                    "repository_horizon.reconnaissance.validate_source_corpus",
                    return_value=[],
                ),
                mock.patch(
                    "repository_horizon.reconnaissance.corpus_has_commit",
                    return_value=True,
                ),
                mock.patch(
                    "repository_horizon.reconnaissance.corpus_has_path",
                    return_value=True,
                ),
                mock.patch(
                    "repository_horizon.reconnaissance.read_catalog",
                    return_value={"schema_version": 1},
                ),
            ):
                self.assertEqual(
                    contract.workspace_violations(
                        SimpleNamespace(workspace=workspace), workspace
                    ),
                    [],
                )
            (workspace / "memory").mkdir()
            (workspace / "memory" / "v0.json").write_text("{}\n", encoding="utf-8")
            (workspace / "plans" / "repository_search.json").unlink()
            self.assertEqual(contract.workspace_violations(None, workspace), [])

    def test_clean_reconnaissance_seal_gates_first_bringup_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, revision = make_source(root)
            manifest = load_manifest(
                make_manifest(
                    root,
                    revision,
                    repository_search={
                        "mode": "replay_strict",
                        "require_report": True,
                        "seal_before_first_eval": True,
                        "min_candidates": 1,
                    },
                    bringup={"mode": "auto"},
                )
            )
            campaign = SeedAndStagingTests._campaign_fixture(root)
            with mock.patch("repository_horizon.seed.install_minimal_runtime"):
                seed_workspace(campaign, manifest, source)
            workspace = campaign.workspace
            base = git_head(workspace)
            runtime = workspace / ".atrex_long_horizon"
            runtime.mkdir(exist_ok=True)
            (runtime / "journal.json").write_text(
                json.dumps({"base_commit": base}), encoding="utf-8"
            )
            report = {
                "schema_version": 1,
                "source_revision": revision,
                "queries": ["paged attention history relevant to the public contract"],
                "candidates": [
                    {
                        "commit": revision,
                        "path": "flash_attn/cute/kernel.py",
                        "mechanism": "reuse the existing tiled kernel structure",
                        "workload_relevance": "the public contract enters this kernel",
                        "transfer_gap": "paged addressing is absent",
                        "risks": "the historical layout may not transfer",
                    }
                ],
                "selected": {
                    "commit": revision,
                    "path": "flash_attn/cute/kernel.py",
                    "rationale": "closest bounded implementation evidence",
                    "gap": "add paged addressing without changing the adapter",
                },
            }
            report_path = workspace / REPORT_RELATIVE
            report_path.parent.mkdir()
            report_path.write_text(json.dumps(report), encoding="utf-8")

            self.assertTrue(reconnaissance_gate_violations(workspace, manifest))
            rejecting_parser = mock.Mock()
            rejecting_parser.error.side_effect = RuntimeError
            with self.assertRaises(RuntimeError):
                _require_reconnaissance(rejecting_parser, workspace)
            self.assertIn(
                "reconnaissance gate", rejecting_parser.error.call_args.args[0]
            )
            seal = seal_reconnaissance(workspace, manifest)
            self.assertEqual(seal, workspace / SEAL_RELATIVE)
            self.assertEqual(reconnaissance_gate_violations(workspace, manifest), [])
            self.assertEqual(
                RepositoryCandidateContract(manifest).workspace_violations(
                    campaign, workspace
                ),
                [],
            )
            accepting_parser = mock.Mock()
            _require_reconnaissance(accepting_parser, workspace)
            accepting_parser.error.assert_not_called()

            report["queries"].append("a post-seal report mutation")
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertIn(
                "report_sha256",
                reconnaissance_gate_violations(workspace, manifest)[0],
            )

            (workspace / SEAL_RELATIVE).unlink()
            editable = (
                workspace
                / "vendor"
                / "flash_attention"
                / "flash_attn"
                / "cute"
                / "kernel.py"
            )
            editable.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "before editable source changes"):
                seal_reconnaissance(workspace, manifest)

    def test_validation_and_promotion_share_repository_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, revision = make_source(root)
            manifest = load_manifest(make_manifest(root, revision))
            contract = RepositoryCandidateContract(manifest)
            repo = root / "campaign"
            init_repo(repo)
            path = (
                repo
                / "vendor"
                / "flash_attention"
                / "flash_attn"
                / "cute"
                / "kernel.py"
            )
            path.parent.mkdir(parents=True)
            path.write_text("VALUE = 1\n", encoding="utf-8")
            run_git(repo, "add", ".")
            run_git(repo, "commit", "-m", "repo baseline")
            base = git_head(repo)
            episode = EpisodeWorktree.create(repo, 1, base, root=root / "worktrees")
            candidate_path = episode.path / path.relative_to(repo)
            candidate_path.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(episode.path, "add", ".")
            run_git(episode.path, "commit", "-m", "source candidate")
            candidate = git_head(episode.path)
            paths = changed_paths(episode.path, base, candidate)
            violation = contract.validate_changed_paths(paths)
            self.assertEqual(violation, "")
            self.assertEqual(paths, [path.relative_to(repo).as_posix()])
            promoted = promote_candidate(
                repo,
                base_commit=base,
                candidate_commit=candidate,
                episode=1,
                evidence={"accepted": True},
                memory_version=1,
                memory_record={"version": "v1"},
            )
            self.assertNotEqual(promoted, base)
            episode.remove(repo)


class SeedAndStagingTests(unittest.TestCase):
    @staticmethod
    def _campaign_fixture(root: Path) -> SimpleNamespace:
        op = root / "op"
        op.mkdir()
        for name in ("reference.py", "input.py"):
            (op / name).write_text("# fixture\n", encoding="utf-8")
        (op / "shapes.json").write_text('{"0": {}}\n', encoding="utf-8")
        bench = root / "atrex-bench"
        (bench / "src" / "atrex_bench").mkdir(parents=True)
        (bench / "src" / "atrex_bench" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (bench / "scripts").mkdir()
        (bench / "scripts" / "run_eval.py").write_text(
            "# fixture\n", encoding="utf-8"
        )
        return SimpleNamespace(
            workspace=root / "campaign",
            kernel_demo=str(op / "reference.py"),
            atrex_bench_root=str(bench),
            framework="CuteDSL",
            framework_baseline="never",
            sandbox_hardware="REMOTE_GPU",
            sandbox_profile="",
            sandbox_url="",
            sandbox_timeout=600,
            optimization_mode="production",
            agent_cli="claude",
        )

    def test_fresh_seed_initializes_git_before_linking_main_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, revision = make_source(root)
            manifest = load_manifest(make_manifest(root, revision))
            campaign = self._campaign_fixture(root)

            seed_workspace(campaign, manifest, source)

            self.assertTrue((campaign.workspace / ".git").is_dir())
            self.assertEqual(run_git(campaign.workspace, "status", "--porcelain"), "")
            self.assertTrue((campaign.workspace / "tools").is_symlink())

    def test_replay_strict_corpus_contains_only_r0_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, first = make_source(root)
            kernel = source / "flash_attn" / "cute" / "kernel.py"
            kernel.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(source, "add", ".")
            run_git(source, "commit", "-m", "r0")
            r0 = git_head(source)
            kernel.write_text("VALUE = 3\n", encoding="utf-8")
            run_git(source, "add", ".")
            run_git(source, "commit", "-m", "hidden answer")
            hidden = git_head(source)
            manifest = load_manifest(
                make_manifest(
                    root,
                    r0,
                    repository_search={
                        "mode": "replay_strict",
                        "excluded_commits": [hidden],
                    },
                    bringup={"mode": "auto"},
                )
            )
            campaign = self._campaign_fixture(root)
            with mock.patch("repository_horizon.seed.install_minimal_runtime"):
                seed_workspace(campaign, manifest, source)
            corpus = campaign.workspace / CORPUS_RELATIVE
            commits = run_git(corpus, "rev-list", "--all").splitlines()
            self.assertEqual(set(commits), {first, r0})
            missing = subprocess.run(
                ["git", "cat-file", "-e", f"{hidden}^{{commit}}"],
                cwd=corpus,
                capture_output=True,
            )
            self.assertNotEqual(missing.returncode, 0)
            self.assertTrue((campaign.workspace / "memory" / "r0.json").is_file())
            self.assertFalse((campaign.workspace / "memory" / "v0.json").exists())
            catalog = json.loads(
                (campaign.workspace / "source_corpus.json").read_text(encoding="utf-8")
            )
            self.assertEqual(validate_source_corpus(campaign.workspace, catalog), [])
            run_git(
                corpus,
                "fetch",
                "--no-tags",
                str(source),
                f"{hidden}:refs/heads/leaked",
            )
            violations = validate_source_corpus(campaign.workspace, catalog)
            self.assertTrue(
                any("physical object set changed" in value for value in violations)
            )
            self.assertTrue(any("excluded commit" in value for value in violations))

    def test_archive_includes_parent_python_package_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source, revision = make_source(Path(temp))
            self.assertEqual(
                _archive_paths_with_package_boundaries(
                    source, revision, ("flash_attn/cute",)
                ),
                ("flash_attn/cute", "flash_attn/__init__.py"),
            )

    def test_allowlist_corpus_fetches_only_explicit_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, r0 = make_source(root)
            run_git(source, "checkout", "-b", "allowed")
            allowed_file = source / "flash_attn" / "cute" / "allowed.py"
            allowed_file.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(source, "add", ".")
            run_git(source, "commit", "-m", "allowed technique")
            allowed = git_head(source)
            run_git(source, "checkout", "-b", "hidden", r0)
            hidden_file = source / "flash_attn" / "cute" / "hidden.py"
            hidden_file.write_text("VALUE = 3\n", encoding="utf-8")
            run_git(source, "add", ".")
            run_git(source, "commit", "-m", "hidden answer")
            hidden = git_head(source)
            manifest = load_manifest(
                make_manifest(
                    root,
                    r0,
                    repository_search={
                        "mode": "allowlist",
                        "refs": ["allowed"],
                        "excluded_commits": [hidden],
                    },
                    bringup={"mode": "auto"},
                )
            )
            campaign = self._campaign_fixture(root)
            with mock.patch("repository_horizon.seed.install_minimal_runtime"):
                seed_workspace(campaign, manifest, source)
            corpus = campaign.workspace / CORPUS_RELATIVE
            self.assertEqual(run_git(corpus, "rev-parse", "refs/heads/ref_0000"), allowed)
            missing = subprocess.run(
                ["git", "cat-file", "-e", f"{hidden}^{{commit}}"],
                cwd=corpus,
                capture_output=True,
            )
            self.assertNotEqual(missing.returncode, 0)

    def test_r0_probe_failure_stays_pre_v0_and_pass_uses_zero_round_fast_path(self) -> None:
        for passed in (False, True):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, revision = make_source(root)
                manifest = load_manifest(
                    make_manifest(
                        root,
                        revision,
                        repository_search={"mode": "replay_strict"},
                        bringup={"mode": "auto"},
                    )
                )
                campaign = self._campaign_fixture(root)
                integration = RepositoryBaselineManager(manifest, source)
                run = VerificationRun(
                    "candidate",
                    0,
                    0 if passed else 1,
                    {"all_pass": passed, "latency_us_geomean": 7.0} if passed else None,
                )
                result = VerificationResult(
                    "PASS" if passed else "FAIL",
                    7.0 if passed else None,
                    None,
                    None,
                    runs=[run],
                    error="" if passed else "unsupported official workload",
                )
                with (
                    mock.patch("repository_horizon.seed.install_minimal_runtime"),
                    mock.patch("repository_horizon.baseline.RepositoryABBAValidator") as validator,
                ):
                    validator.return_value.verify.return_value = result
                    integration.prepare(campaign)
                    integration.prepare(campaign)
                    self.assertEqual(validator.return_value.verify.call_count, 1)
                self.assertTrue((campaign.workspace / "memory" / "r0.json").is_file())
                self.assertEqual((campaign.workspace / "memory" / "v0.json").is_file(), passed)
                r0_memory = json.loads(
                    (campaign.workspace / "memory" / "r0.json").read_text(encoding="utf-8")
                )
                if passed:
                    v0 = json.loads(
                        (campaign.workspace / "memory" / "v0.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(v0["quality_gate"]["result"], "PASS")
                    self.assertEqual(v0["version"], "v0")
                else:
                    self.assertEqual(r0_memory["quality_gate"]["result"], "BRINGUP_REQUIRED")
                self.assertEqual(run_git(campaign.workspace, "status", "--porcelain"), "")

    def test_r0_infrastructure_error_is_not_misclassified_as_bringup(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, revision = make_source(root)
            manifest = load_manifest(
                make_manifest(root, revision, bringup={"mode": "auto"})
            )
            campaign = self._campaign_fixture(root)
            integration = RepositoryBaselineManager(manifest, source)
            result = VerificationResult(
                "ERROR", None, None, None, error="GPU queue timed out"
            )
            with (
                mock.patch("repository_horizon.seed.install_minimal_runtime"),
                mock.patch("repository_horizon.baseline.RepositoryABBAValidator") as validator,
            ):
                validator.return_value.verify.return_value = result
                with self.assertRaisesRegex(RuntimeError, "no authoritative workload outcome"):
                    integration.prepare(campaign)
            memory = json.loads(
                (campaign.workspace / "memory" / "r0.json").read_text(encoding="utf-8")
            )
            self.assertNotEqual(
                (memory.get("quality_gate") or {}).get("result"), "BRINGUP_REQUIRED"
            )

    def test_seed_uses_exact_git_archive_and_stage_is_base_plus_delta(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, revision = make_source(root)
            manifest = load_manifest(make_manifest(root, revision))
            op = root / "op"
            op.mkdir()
            for name in ("reference.py", "input.py"):
                (op / name).write_text("# fixture\n", encoding="utf-8")
            (op / "shapes.json").write_text('{"0": {}}\n', encoding="utf-8")
            bench = root / "atrex-bench"
            (bench / "src" / "atrex_bench").mkdir(parents=True)
            (bench / "src" / "atrex_bench" / "__init__.py").write_text(
                "", encoding="utf-8"
            )
            (bench / "scripts").mkdir()
            (bench / "scripts" / "run_eval.py").write_text(
                "# fixture\n", encoding="utf-8"
            )
            campaign = SimpleNamespace(
                workspace=root / "campaign",
                kernel_demo=str(op / "reference.py"),
                atrex_bench_root=str(bench),
            )
            with mock.patch(
                "repository_horizon.seed.install_minimal_runtime"
            ):
                seed_workspace(campaign, manifest, source)
            vendor_file = (
                campaign.workspace
                / "vendor"
                / "flash_attention"
                / "flash_attn"
                / "cute"
                / "kernel.py"
            )
            self.assertEqual(vendor_file.read_text(encoding="utf-8"), "VALUE = 1\n")
            self.assertTrue((campaign.workspace / "source.lock.json").is_file())
            self.assertIn(
                "Repository-assisted production mode",
                (campaign.workspace / "CLAUDE.md").read_text(encoding="utf-8"),
            )
            with mock.patch(
                "repository_horizon.seed.install_minimal_runtime"
            ):
                seed_workspace(campaign, manifest, source)
            self.assertEqual(run_git(campaign.workspace, "status", "--porcelain"), "")
            base = git_head(campaign.workspace)
            vendor_file.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(campaign.workspace, "add", ".")
            run_git(campaign.workspace, "commit", "-m", "candidate")
            candidate = git_head(campaign.workspace)
            relative = vendor_file.relative_to(campaign.workspace).as_posix()
            stage = root / "stage"
            metadata = build_abba_stage(
                campaign.workspace,
                base_commit=base,
                candidate_commit=candidate,
                changed_paths=[relative],
                manifest=manifest,
                atrex_bench_root=bench,
                destination=stage,
                schedule=verification_schedule(2),
                per_run_timeout=10,
            )
            self.assertGreater(metadata["packed_bytes"], 0)
            self.assertEqual(
                (stage / "runtime" / relative).read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            request = json.loads((stage / "request.json").read_text(encoding="utf-8"))
            snapshot = request["manifests"]["candidate"][relative]
            self.assertEqual(
                (stage / snapshot).read_text(encoding="utf-8"), "VALUE = 2\n"
            )

    def test_agate_payload_can_be_nested_in_json_output(self) -> None:
        payload = {"schema_version": 1, "runs": [], "error": None}
        sentinel = "__ATREX_LONG_HORIZON_ABBA_RESULT__=" + json.dumps(payload)
        output = json.dumps({"result": {"stdout": "prefix\n" + sentinel + "\n"}})
        self.assertEqual(_payload(output), payload)

    def test_agate_development_submission_is_non_waiting(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"job_id":"job-123456789012","status":"queued"}\n',
            stderr="",
        )
        with mock.patch(
            "repository_horizon.transport.subprocess.run", return_value=completed
        ) as run:
            pending = submit_agate_dev(
                Path("/tmp/stage"),
                hardware="L20D",
                profile="prod",
                url="",
                job_timeout=600,
            )
        command = run.call_args.args[0]
        self.assertEqual(pending.job_id, "job-123456789012")
        self.assertIn("--no-wait", command)
        self.assertNotIn("--wait-timeout", command)
        self.assertNotIn("--wait", command)

    def test_local_development_uses_supervisor_python(self) -> None:
        process = mock.Mock(pid=12345)
        with mock.patch(
            "repository_horizon.transport.subprocess.Popen", return_value=process
        ) as popen:
            submit_local_dev(Path("/tmp/stage"), job_timeout=600)
        command = popen.call_args.args[0]
        encoded = command[command.index("--command-json") + 1]
        self.assertEqual(json.loads(encoded), [sys.executable, "repo_abba.py"])

    def test_local_development_honors_pinned_python(self) -> None:
        process = mock.Mock(pid=12345)
        with mock.patch.dict(
            "repository_horizon.transport.os.environ",
            {"ATREX_LOCAL_PYTHON": "/runtime/venv/bin/python"},
        ), mock.patch(
            "repository_horizon.transport.subprocess.Popen", return_value=process
        ) as popen:
            submit_local_dev(Path("/tmp/stage"), job_timeout=600)
        command = popen.call_args.args[0]
        encoded = command[command.index("--command-json") + 1]
        self.assertEqual(
            json.loads(encoded), ["/runtime/venv/bin/python", "repo_abba.py"]
        )

    def test_ncu_profile_uses_explicit_target_python(self) -> None:
        script = (
            Path(__file__).resolve().parents[2] / "tools" / "profile_nvidia.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'PROFILE_TARGET_PYTHON="${ATREX_PROFILE_TARGET_PYTHON:-${ATREX_LOCAL_PYTHON:-python3}}"',
            script,
        )
        self.assertGreaterEqual(
            script.count('"$PROFILE_TARGET_PYTHON" "$KERNEL_FILE"'), 2
        )
        self.assertNotIn('python "$KERNEL_FILE"', script)

    def test_profile_target_python_preserves_bare_installation(self) -> None:
        self.assertEqual(_profile_target_python({}), sys.executable)

    def test_profile_target_python_honors_pinned_venv(self) -> None:
        self.assertEqual(
            _profile_target_python(
                {"ATREX_LOCAL_PYTHON": "/runtime/venv/bin/python"}
            ),
            "/runtime/venv/bin/python",
        )

    def test_agate_get_preserves_failed_terminal_job(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=json.dumps(
                {
                    "job_id": "job-123456789012",
                    "status": "failed",
                    "error": "worker unavailable",
                    "result": None,
                }
            ),
            stderr="",
        )
        with mock.patch(
            "repository_horizon.transport.subprocess.run", return_value=completed
        ):
            snapshot = get_agate_job(
                "job-123456789012",
                profile="prod",
                url="",
            )
        self.assertTrue(snapshot.terminal)
        self.assertEqual(snapshot.status, "failed")

    def test_support_wheel_is_minimized_locked_and_importable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, revision = make_source(root)
            manifest = load_manifest(
                make_manifest(root, revision, runtime_support=support_manifest())
            )
            wheel = make_support_wheel(root)
            op = root / "op"
            op.mkdir()
            (op / "reference.py").write_text("# fixture\n", encoding="utf-8")
            bench = root / "atrex-bench"
            (bench / "src" / "atrex_bench").mkdir(parents=True)
            (bench / "scripts").mkdir()
            (bench / "scripts" / "run_eval.py").write_text(
                "# fixture\n", encoding="utf-8"
            )
            campaign = SimpleNamespace(
                workspace=root / "campaign",
                kernel_demo=str(op / "reference.py"),
                atrex_bench_root=str(bench),
            )
            with mock.patch(
                "repository_horizon.seed.install_minimal_runtime"
            ):
                seed_workspace(
                    campaign,
                    manifest,
                    source,
                    support_wheels={"quack-kernels": wheel},
                )
            support = campaign.workspace / "vendor_support"
            self.assertFalse((support / "quack" / "activation.py").exists())
            self.assertIn(
                "Generated minimal package shim",
                (support / "quack" / "__init__.py").read_text(encoding="utf-8"),
            )
            lock = json.loads(
                (campaign.workspace / "source.lock.json").read_text(encoding="utf-8")
            )
            self.assertEqual(lock["runtime_support"][0]["version"], "0.5.3")
            self.assertTrue(lock["runtime_support_tree_sha256"])
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import importlib.metadata, quack.core; "
                    "print(importlib.metadata.version('quack-kernels'), quack.core.VALUE)",
                ],
                env={"PYTHONPATH": str(support)},
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "0.5.3 1")

            (support / "quack" / "core.py").write_text("VALUE = 9\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "vendor_support"):
                with mock.patch(
                    "repository_horizon.seed.install_minimal_runtime"
                ):
                    seed_workspace(campaign, manifest, source)

    def test_support_wheel_rejects_missing_import_closure_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel = make_support_wheel(root)
            config = RuntimeSupportWheel(
                distribution="quack-kernels",
                version="0.5.3",
                package="quack",
                members=("quack/core.py",),
                dist_info_members=("METADATA", "WHEEL"),
            )
            with self.assertRaisesRegex(ValueError, "import closure"):
                extract_support_wheel(config, wheel, root / "support")
            wrong = RuntimeSupportWheel(
                distribution="quack-kernels",
                version="9.9.9",
                package="quack",
                members=("quack/core.py", "quack/helper.py"),
                dist_info_members=("METADATA", "WHEEL"),
            )
            with self.assertRaisesRegex(ValueError, "version mismatch"):
                extract_support_wheel(wrong, wheel, root / "wrong")

    def test_manifest_rejects_support_wheel_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, revision = make_source(root)
            support = support_manifest()
            support[0]["members"] = ["../quack/core.py"]
            with self.assertRaisesRegex(ValueError, "unsafe relative path"):
                load_manifest(make_manifest(root, revision, runtime_support=support))


if __name__ == "__main__":
    unittest.main()
