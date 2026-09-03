from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from long_horizon.git_episode import git_head
from long_horizon.protocol import atomic_write_json
from orchestrator.operator_layout import agent_visible_operator_files

from .corpus import (
    CATALOG_NAME,
    CORPUS_RELATIVE,
    build_source_corpus,
    validate_source_corpus,
)
from .lockfile import tree_digest, write_lock
from .manifest import RepositoryManifest
from .policy import install_repository_policy
from .runtime import install_minimal_runtime
from .support_wheel import (
    canonical_distribution,
    extract_support_wheel,
    validate_support_imports,
    wheel_sha256,
)

CAMPAIGN_GIT_USER_NAME = "atrex-long-horizon"
CAMPAIGN_GIT_USER_EMAIL = "atrex-long-horizon@local"


def _git(source: Path, *args: str, binary: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(source),
        check=True,
        capture_output=True,
        text=not binary,
    )


def _ensure_campaign_git_identity(workspace: Path) -> None:
    """Install a repository-local identity when no inherited identity exists."""
    for key, value in (
        ("user.name", CAMPAIGN_GIT_USER_NAME),
        ("user.email", CAMPAIGN_GIT_USER_EMAIL),
    ):
        configured = subprocess.run(
            ["git", "config", "--local", "--get", key],
            cwd=str(workspace),
            capture_output=True,
            text=True,
        )
        if configured.returncode == 0 and configured.stdout.strip():
            continue
        subprocess.run(
            ["git", "config", "--local", key, value],
            cwd=str(workspace),
            check=True,
            capture_output=True,
            text=True,
        )


def _extract_archive(data: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe git archive member: {member.name}")
        archive.extractall(destination, filter="data")


def _archive_paths_with_package_boundaries(
    source: Path,
    revision: str,
    archive_paths: tuple[str, ...],
    *,
    mode: str = "upstream",
) -> tuple[str, ...]:
    """Include immutable parent package markers needed to import editable trees."""
    if mode == "minimal":
        return archive_paths
    if mode != "upstream":
        raise ValueError(f"unsupported package boundary mode: {mode}")
    paths = list(archive_paths)
    seen = set(paths)
    for raw in archive_paths:
        current = PurePosixPath(raw).parent
        while current.parts:
            marker = (current / "__init__.py").as_posix()
            if marker not in seen:
                exists = (
                    subprocess.run(
                        ["git", "cat-file", "-e", f"{revision}:{marker}"],
                        cwd=str(source),
                        capture_output=True,
                    ).returncode
                    == 0
                )
                if exists:
                    paths.append(marker)
                    seen.add(marker)
            current = current.parent
    return tuple(paths)


def _install_minimal_package_boundaries(
    destination: Path, archive_paths: tuple[str, ...]
) -> tuple[str, ...]:
    """Create inert parent package markers around a minimized source subtree."""
    generated: list[str] = []
    for raw in archive_paths:
        path = PurePosixPath(raw)
        current = path.parent
        parents: list[PurePosixPath] = []
        while current.parts:
            parents.append(current)
            current = current.parent
        for package in reversed(parents):
            if any(not part.isidentifier() for part in package.parts):
                raise ValueError(
                    "minimal package boundaries require Python package paths: "
                    + package.as_posix()
                )
            marker = destination / package / "__init__.py"
            if marker.exists():
                continue
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                '"""Generated inert package boundary for Repository Horizon."""\n',
                encoding="utf-8",
            )
            generated.append(marker.relative_to(destination).as_posix())
    return tuple(generated)


def seed_workspace(
    campaign,
    manifest: RepositoryManifest,
    source_checkout: Path,
    support_wheels: dict[str, Path] | None = None,
) -> None:
    support_wheels = support_wheels or {}
    workspace = campaign.workspace
    if workspace.exists():
        if not git_head(workspace):
            raise RuntimeError(
                f"existing repository campaign has no Git HEAD: {workspace}"
            )
        _ensure_campaign_git_identity(workspace)
        lock_path = workspace / "source.lock.json"
        if not lock_path.is_file():
            raise RuntimeError("existing repository campaign has no source.lock.json")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("source_revision") != manifest.revision:
            raise RuntimeError(
                "source manifest revision differs from resumed campaign lock"
            )
        workspace_manifest = workspace / "source_manifest.json"
        expected_manifest_hash = str(lock.get("manifest_sha256", ""))
        actual_workspace_hash = hashlib.sha256(
            workspace_manifest.read_bytes()
        ).hexdigest()
        requested_manifest_hash = hashlib.sha256(manifest.path.read_bytes()).hexdigest()
        if (
            not expected_manifest_hash
            or actual_workspace_hash != expected_manifest_hash
        ):
            raise RuntimeError(
                "protected source_manifest.json no longer matches source.lock.json"
            )
        if requested_manifest_hash != expected_manifest_hash:
            raise RuntimeError(
                "requested source manifest differs from resumed campaign lock"
            )
        support_root = workspace / "vendor_support"
        locked_support = lock.get("runtime_support") or []
        if len(locked_support) != len(manifest.runtime_support):
            raise RuntimeError(
                "runtime support manifest differs from resumed campaign lock"
            )
        configured_support = {
            canonical_distribution(config.distribution): config
            for config in manifest.runtime_support
        }
        for item in locked_support:
            distribution = canonical_distribution(str(item.get("distribution", "")))
            config = configured_support.get(distribution)
            if config is None:
                raise RuntimeError(
                    "runtime support distribution differs from resumed campaign lock"
                )
            if item.get("version") != config.version:
                raise RuntimeError(
                    "runtime support version differs from resumed campaign lock"
                )
            supplied = support_wheels.get(distribution)
            if supplied and wheel_sha256(supplied) != item.get("wheel_sha256"):
                raise RuntimeError(
                    f"supplied support wheel differs from resumed lock: {distribution}"
                )
        expected_support = lock.get("runtime_support_tree_sha256")
        actual_support, _ = tree_digest(support_root)
        if locked_support and (
            not expected_support or actual_support != expected_support
        ):
            raise RuntimeError(
                "protected vendor_support no longer matches source.lock.json"
            )
        locked_corpus = lock.get("source_corpus")
        catalog_path = workspace / CATALOG_NAME
        if locked_corpus:
            if not catalog_path.is_file():
                raise RuntimeError("resumed campaign is missing source_corpus.json")
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            if catalog != locked_corpus:
                raise RuntimeError(
                    "protected source_corpus.json no longer matches source.lock.json"
                )
            if not (workspace / CORPUS_RELATIVE).is_dir():
                raise RuntimeError(
                    "resumed campaign is missing its bounded source corpus"
                )
            violations = validate_source_corpus(workspace, catalog)
            if violations:
                raise RuntimeError(
                    "resumed campaign source corpus failed integrity: "
                    + "; ".join(violations)
                )
        elif manifest.repository_search.mode != "snapshot":
            raise RuntimeError("resumed campaign lock has no declared source corpus")
        install_minimal_runtime(campaign, workspace, manifest)
        install_repository_policy(workspace, manifest)
        if hasattr(campaign, "_assert_generalized_inputs_are_private"):
            campaign._assert_generalized_inputs_are_private()
        return

    resolved = _git(
        source_checkout, "rev-parse", "--verify", f"{manifest.revision}^{{commit}}"
    )
    exact_revision = resolved.stdout.strip()
    if exact_revision != manifest.revision:
        raise RuntimeError(
            "source.revision must be a full immutable commit id "
            f"(manifest={manifest.revision}, resolved={exact_revision})"
        )
    workspace.mkdir(parents=True)
    shutil.copy2(manifest.adapter, workspace / "kernel.py")
    op_dir = Path(campaign.kernel_demo).resolve().parent
    generalized = getattr(campaign, "private_reference_dir", None) is not None
    for name in agent_visible_operator_files(op_dir, generalized=generalized):
        source = op_dir / name
        if source.is_file():
            shutil.copy2(source, workspace / name)
    if hasattr(campaign, "_ensure_agent_problem"):
        campaign._ensure_agent_problem()
    if hasattr(campaign, "_assert_generalized_inputs_are_private"):
        campaign._assert_generalized_inputs_are_private()
    harness = (
        Path(__file__).resolve().parent.parent
        / "reference"
        / "atrex_bench_test_kernel.py"
    )
    if campaign.atrex_bench_root:
        shutil.copy2(harness, workspace / "test_kernel.py")
    else:
        raise RuntimeError(
            "repository horizon v1 requires an Atrex-Bench native operator"
        )

    archive_paths = _archive_paths_with_package_boundaries(
        source_checkout,
        exact_revision,
        manifest.archive_paths,
        mode=manifest.package_boundaries,
    )
    archive = _git(
        source_checkout,
        "archive",
        "--format=tar",
        exact_revision,
        "--",
        *archive_paths,
        binary=True,
    ).stdout
    vendor_root = workspace / manifest.vendor_root
    _extract_archive(archive, vendor_root)
    if manifest.package_boundaries == "minimal":
        _install_minimal_package_boundaries(vendor_root, manifest.archive_paths)
    support_records: list[dict[str, object]] = []
    if manifest.runtime_support:
        support_root = workspace / "vendor_support"
        missing = []
        for config in manifest.runtime_support:
            key = canonical_distribution(config.distribution)
            wheel = support_wheels.get(key)
            if wheel is None:
                missing.append(config.distribution)
                continue
            package_root = support_root / config.package.replace(".", "/")
            if package_root.exists():
                raise RuntimeError(
                    f"runtime support packages overlap: {config.package}"
                )
            support_records.append(extract_support_wheel(config, wheel, support_root))
        if missing:
            raise RuntimeError(
                "new campaign requires --support-wheel for: " + ", ".join(missing)
            )
        validate_support_imports(vendor_root, manifest.runtime_support, support_root)
    manifest_bytes = manifest.path.read_bytes()
    (workspace / "source_manifest.json").write_bytes(manifest_bytes)
    source_corpus = build_source_corpus(
        source_checkout,
        exact_revision,
        manifest.repository_search,
        workspace / CORPUS_RELATIVE,
    )
    if source_corpus is not None:
        atomic_write_json(workspace / CATALOG_NAME, source_corpus)
    write_lock(
        workspace / "source.lock.json",
        source_checkout=source_checkout,
        source_revision=exact_revision,
        source_root=vendor_root,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        atrex_bench_root=campaign.atrex_bench_root,
        runtime_support=support_records,
        runtime_support_root=(
            (workspace / "vendor_support") if support_records else None
        ),
        source_corpus=source_corpus,
    )
    (workspace / ".gitignore").write_text(
        "__pycache__/\n*.pyc\n/.repository_horizon_runtime/\n", encoding="utf-8"
    )
    # Main's runtime setup installs campaign-local Git excludes.  A genuinely fresh
    # repository workspace therefore has to become a Git repository before linking
    # that runtime; tests and resumed workspaces may already have hidden this ordering
    # requirement by arriving pre-initialized.
    subprocess.run(["git", "init"], cwd=str(workspace), check=True, capture_output=True)
    _ensure_campaign_git_identity(workspace)
    install_minimal_runtime(campaign, workspace, manifest)
    install_repository_policy(workspace, manifest)
    bringup_enabled = manifest.bringup.mode == "auto"
    baseline_name = "r0.json" if bringup_enabled else "v0.json"
    atomic_write_json(
        workspace / "memory" / baseline_name,
        {
            "version": "r0" if bringup_enabled else "v0",
            "masked": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": {
                "name": manifest.source_name,
                "revision": exact_revision,
                "manifest": "source_manifest.json",
                "lock": "source.lock.json",
            },
            "performance": {
                "latency_us": None,
                "measurement": "deferred_to_first_ABBA",
            },
            "correctness": {"status": "UNMEASURED"},
            "quality_gate": {
                "result": "SOURCE_SEEDED" if bringup_enabled else "BASELINE_SEEDED"
            },
            "optimization": {
                "action_category": (
                    "repository_r0_seed" if bringup_enabled else "repository_v0_seed"
                ),
                "action_description": (
                    "mechanical immutable source snapshot awaiting capability probe"
                    if bringup_enabled
                    else "mechanical immutable source snapshot and fixed adapter"
                ),
            },
        },
    )
    subprocess.run(["git", "add", "."], cwd=str(workspace), check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=atrex-repository-horizon",
            "-c",
            "user.email=atrex-repository-horizon@local",
            "commit",
            "-m",
            (
                f"R0: seed {manifest.source_name} at {exact_revision[:12]}"
                if bringup_enabled
                else f"V0: seed {manifest.source_name} at {exact_revision[:12]}"
            ),
        ],
        cwd=str(workspace),
        check=True,
        capture_output=True,
    )
