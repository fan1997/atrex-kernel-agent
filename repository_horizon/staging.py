from __future__ import annotations

import io
import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from long_horizon.protocol import atomic_write_json

from .manifest import RepositoryManifest

PRUNE_ROOTS = {"memory", "plans", "profiles", ".git", ".atrex_long_horizon"}


def _safe_extract(data: bytes, destination: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def _archive_revision(workspace: Path, revision: str, destination: Path) -> None:
    process = subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=str(workspace),
        check=True,
        capture_output=True,
    )
    _safe_extract(process.stdout, destination)
    for name in PRUNE_ROOTS:
        target = destination / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _copy_python_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in source.rglob("*"):
        if (
            not path.is_file()
            or path.suffix in {".pyc", ".so"}
            or "__pycache__" in path.parts
        ):
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def add_atrex_runtime(stage_runtime: Path, atrex_bench_root: Path) -> None:
    root = stage_runtime / "atrex-bench"
    _copy_python_tree(atrex_bench_root / "src", root / "src")
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        atrex_bench_root / "scripts" / "run_eval.py", root / "scripts" / "run_eval.py"
    )


def _blob(workspace: Path, revision: str, relative: str) -> bytes | None:
    process = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=str(workspace),
        capture_output=True,
    )
    return process.stdout if process.returncode == 0 else None


def packed_size(root: Path) -> int:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        archive.add(root, arcname=".")
    return len(stream.getvalue())


def build_abba_stage(
    workspace: Path,
    *,
    base_commit: str,
    candidate_commit: str,
    changed_paths: list[str],
    manifest: RepositoryManifest,
    atrex_bench_root: Path,
    destination: Path,
    schedule: list[dict[str, int | str]],
    per_run_timeout: int,
    working_snapshot: bool = False,
) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    runtime = destination / "runtime"
    runtime.mkdir()
    _archive_revision(workspace, base_commit, runtime)
    # Agate's working-dir packer honors nested .gitignore files.  Campaign
    # runtime entries such as /atrex-bench are intentionally ignored locally,
    # but staging must materialize them, so do not ship that ignore file.
    (runtime / ".gitignore").unlink(missing_ok=True)
    add_atrex_runtime(runtime, atrex_bench_root)
    candidate_manifest: dict[str, str | None] = {}
    for index, relative in enumerate(changed_paths):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe changed path: {relative}")
        source_path = workspace / path
        content = (
            source_path.read_bytes() if working_snapshot and source_path.is_file()
            else (None if working_snapshot else _blob(workspace, candidate_commit, relative))
        )
        if content is None:
            candidate_manifest[relative] = None
        else:
            # working-dir accepts UTF-8 source and excludes binary-looking
            # extensions by default. Repository Horizon v1 admits Python/JIT
            # source, so a text snapshot is the correct transport contract.
            snapshot = f"snapshots/candidate/{index:04d}.source"
            target = destination / snapshot
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            candidate_manifest[relative] = snapshot
    request = {
        "schema_version": 1,
        "schedule": schedule,
        "manifests": {"incumbent": {}, "candidate": candidate_manifest},
        "command": [
            "python3",
            "test_kernel.py",
            "--version",
            "vlong",
            "--no-memory",
            "--warmup",
            str(manifest.measurement.warmup),
            "--timed-runs",
            str(manifest.measurement.timed_runs),
        ],
        "run_timeout_seconds": per_run_timeout,
        "python_roots": [
            f"{manifest.vendor_root}/{manifest.package_root}".rstrip("/"),
            "atrex-bench/src",
        ]
        + (["vendor_support"] if (runtime / "vendor_support").is_dir() else []),
        "runtime_requirements": [
            {
                "distribution": item.distribution,
                "import": item.import_name,
                "version": item.version,
            }
            for item in manifest.runtime_requirements
        ],
    }
    atomic_write_json(destination / "request.json", request)
    shutil.copy2(
        Path(__file__).with_name("remote_abba.py"), destination / "repo_abba.py"
    )
    size = packed_size(destination)
    atomic_write_json(
        destination / "staging.manifest.json",
        {
            "schema_version": 1,
            "base_commit": base_commit,
            "candidate_commit": candidate_commit,
            "changed_paths": changed_paths,
            "packed_tar_gz_bytes_before_manifest": size,
        },
    )
    digest = hashlib.sha256()
    for path in sorted(item for item in destination.rglob("*") if item.is_file()):
        digest.update(path.relative_to(destination).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return {
        "packed_bytes": packed_size(destination),
        "request": request,
        "request_digest": digest.hexdigest(),
    }
