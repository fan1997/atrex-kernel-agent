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

from long_horizon.git_episode import EpisodeWorktree, git_head, promote_candidate
from long_horizon.tests.helpers import init_repo, run_git
from long_horizon.verifier import verification_schedule
from repository_horizon.candidate import RepositoryCandidateContract
from repository_horizon.manifest import RuntimeSupportWheel, load_manifest
from repository_horizon.seed import seed_workspace
from repository_horizon.staging import build_abba_stage
from repository_horizon.support_wheel import extract_support_wheel
from repository_horizon.transport import _payload


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
    root: Path, revision: str, runtime_support: list[dict] | None = None
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
            violation, paths = episode.validate_candidate(candidate, contract)
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
                contract=contract,
            )
            self.assertNotEqual(promoted, base)
            episode.remove(repo)


class SeedAndStagingTests(unittest.TestCase):
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
                "repository_horizon.seed.main_adapter.link_episode_runtime"
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
                "repository_horizon.seed.main_adapter.link_episode_runtime"
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
                    "repository_horizon.seed.main_adapter.link_episode_runtime"
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
