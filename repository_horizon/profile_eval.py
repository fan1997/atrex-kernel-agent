from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from long_horizon.models import VerificationResult
from long_horizon.protocol import atomic_write_json

from .agent_result import write_agent_result
from .evaluation import (
    PendingVerification,
    load_completed_verification,
    save_pending,
    utc_now,
    wait_for_terminal_job,
)
from .manifest import load_manifest
from .staging import build_abba_stage, packed_size
from .transport import (
    profile_payload,
    submit_agate_dev,
    submit_agate_profile,
    submit_local_dev,
)


def _git_lines(workspace: Path, *args: str) -> list[str]:
    process = subprocess.run(
        ["git", *args], cwd=str(workspace), check=True, capture_output=True, text=True
    )
    return [line for line in process.stdout.splitlines() if line]


def _working_paths(workspace: Path, base_commit: str) -> list[str]:
    tracked = _git_lines(workspace, "diff", "--name-only", base_commit, "--")
    untracked = _git_lines(workspace, "ls-files", "--others", "--exclude-standard")
    return sorted(set(tracked + untracked))


def _stage_digest(stage: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        digest.update(path.relative_to(stage).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _build_repository_stage(
    workspace: Path,
    directory: Path,
    *,
    detail: dict,
    job_timeout: int,
) -> tuple[Path, str, int]:
    journal = json.loads(
        (workspace / ".atrex_long_horizon" / "journal.json").read_text(encoding="utf-8")
    )
    base_commit = str(journal["base_commit"])
    candidate_commit = _git_lines(workspace, "rev-parse", "HEAD")[0]
    lock = json.loads((workspace / "source.lock.json").read_text(encoding="utf-8"))
    manifest = load_manifest(workspace / "source_manifest.json")
    stage = directory / "staging"
    build_abba_stage(
        workspace,
        base_commit=base_commit,
        candidate_commit=candidate_commit,
        changed_paths=_working_paths(workspace, base_commit),
        manifest=manifest,
        atrex_bench_root=Path(lock["atrex_bench_root"]),
        destination=stage,
        schedule=[{"revision": "candidate", "repeat": 0}],
        per_run_timeout=job_timeout,
        working_snapshot=True,
    )
    (stage / "repo_abba.py").unlink(missing_ok=True)
    root = Path(__file__).resolve().parent.parent
    shutil.copy2(
        Path(__file__).with_name("repository_profile.py"),
        stage / "repository_profile.py",
    )
    tools = stage / "tools"
    tools.mkdir()
    shutil.copy2(root / "tools" / "profile_nvidia.sh", tools / "profile_nvidia.sh")
    shutil.copy2(root / "tools" / "classify_ncu.py", tools / "classify_ncu.py")
    shutil.copytree(root / "tools" / "ncu_helpers", tools / "ncu_helpers")
    atomic_write_json(
        stage / "profile_request.json",
        {
            **detail,
            "job_timeout": max(1, job_timeout - 30),
            "kernel_filter": (
                detail.get("kernel_name")
                or (
                    f"regex:{detail['kernel_regex']}"
                    if detail.get("kernel_regex")
                    else ""
                )
            ),
        },
    )
    return stage, _stage_digest(stage), packed_size(stage)


def submit_profile(
    workspace: Path,
    *,
    candidate: Path,
    reference_dir: Path,
    hardware: str,
    profile: str,
    url: str,
    level: str,
    wait_mode: str,
    wait_timeout: int,
    job_timeout: int,
    kernel_name: str = "",
    kernel_regex: str = "",
    source: bool = False,
    launch_skip: int = 0,
    launch_count: int = 0,
    agent_result_max_bytes: int = 16 * 1024,
    backend: str = "agate",
    route: str = "auto",
) -> PendingVerification:
    if level == "deep" and not (kernel_name or kernel_regex):
        raise ValueError("deep profile requires --kernel-name or --kernel-regex")
    if kernel_regex and not (
        kernel_regex.startswith("^") and kernel_regex.endswith("$")
    ):
        raise ValueError("--kernel-regex must be anchored with ^ and $")
    if kernel_name and kernel_regex:
        raise ValueError("--kernel-name and --kernel-regex are mutually exclusive")
    evaluation_id = uuid.uuid4().hex
    directory = (
        workspace / ".repository_horizon_runtime" / "evaluations" / evaluation_id
    )
    directory.mkdir(parents=True, exist_ok=False)
    detail = {
        "level": level,
        "candidate": str(candidate),
        "reference_dir": str(reference_dir),
        "kernel_name": kernel_name or None,
        "kernel_regex": kernel_regex or None,
        "source": source,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "route": route,
    }
    if route not in {"auto", "typed", "repository"}:
        raise ValueError(f"unsupported profile route: {route}")
    if route == "auto":
        journal = json.loads(
            (workspace / ".atrex_long_horizon" / "journal.json").read_text(
                encoding="utf-8"
            )
        )
        changed = _working_paths(workspace, str(journal["base_commit"]))
        candidate_relative = candidate.relative_to(workspace).as_posix()
        route = (
            "typed"
            if backend == "agate"
            and all(path == candidate_relative for path in changed)
            and all(
                (reference_dir / name).is_file()
                for name in ("reference.py", "input.py", "shapes.json")
            )
            else "repository"
        )
        detail["route"] = route
    if backend == "local" and route == "typed":
        raise ValueError("direct-local profiling requires --route repository")

    if route == "typed":
        digest = hashlib.sha256(
            json.dumps(detail, sort_keys=True).encode("utf-8") + candidate.read_bytes()
        ).hexdigest()
        stage = directory / "profile"
        packed_bytes = 0
    else:
        stage, digest, packed_bytes = _build_repository_stage(
            workspace, directory, detail=detail, job_timeout=job_timeout
        )
        if backend == "agate" and packed_bytes > 900_000:
            shutil.rmtree(directory)
            raise ValueError(
                f"repository profile staging payload is {packed_bytes} bytes; "
                "Agate working-dir limit is 900000"
            )

    runtime_root = directory.parent
    for existing_path in runtime_root.glob("*/pending.json"):
        existing = PendingVerification.from_dict(
            json.loads(existing_path.read_text(encoding="utf-8"))
        )
        if existing.request_digest == digest and not existing.result_path.is_file():
            shutil.rmtree(directory)
            return existing

    if route == "typed":
        submitted = submit_agate_profile(
            candidate=candidate,
            reference_dir=reference_dir,
            hardware=hardware,
            profile=profile,
            url=url,
            level=level,
            kernel_name=kernel_name,
            kernel_regex=kernel_regex,
            source=source,
            launch_skip=launch_skip,
            launch_count=launch_count,
            job_timeout=job_timeout,
        )
    else:
        local_python = os.environ.get("ATREX_LOCAL_PYTHON", sys.executable)
        submitted = (
            submit_local_dev(
                stage,
                job_timeout=job_timeout,
                remote_command=(local_python, "repository_profile.py"),
            )
            if backend == "local"
            else submit_agate_dev(
                stage,
                hardware=hardware,
                profile=profile,
                url=url,
                job_timeout=job_timeout,
                remote_command="python3 repository_profile.py",
                intent="profile_adhoc",
                note="repository horizon custom NCU via upstream profile_nvidia.sh",
            )
        )
    pending = PendingVerification(
        schema_version=1,
        evaluation_id=evaluation_id,
        job_id=submitted.job_id,
        hardware=hardware,
        profile=profile,
        url=url,
        job_timeout=job_timeout,
        wait_timeout=wait_timeout,
        stage=str(stage),
        base_commit="",
        candidate_commit="",
        changed_paths=(),
        schedule=(),
        repeats=1,
        min_improvement_pct=0.0,
        candidate_only=True,
        packed_bytes=packed_bytes,
        submit_command=submitted.command,
        submitted_at=utc_now(),
        backend=backend,
        wait_mode=wait_mode,
        request_digest=digest,
        snapshot_kind="worktree",
        kind="profile",
        detail=detail,
        agent_result_max_bytes=agent_result_max_bytes,
    )
    save_pending(pending)
    atomic_write_json(
        pending.directory / "submission.json",
        {
            "schema_version": 1,
            "job_id": submitted.job_id,
            "request_digest": digest,
            "detail": detail,
        },
    )
    return pending


def collect_profile(pending: PendingVerification) -> VerificationResult:
    completed = load_completed_verification(pending)
    if completed is not None:
        return completed
    snapshot = wait_for_terminal_job(pending)
    response = snapshot.response
    detail = pending.detail or {}
    repository_route = detail.get("route") == "repository"
    if repository_route:
        try:
            payload = profile_payload(snapshot)
        except ValueError as exc:
            payload = {
                "command_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "artifacts": response.get("artifacts"),
            }
    else:
        payload = response
    if repository_route:
        artifacts_dir = pending.directory / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for key, relative in (
            ("summary", "summary.txt"),
            ("metrics", "metrics_key_run.txt"),
            ("hotspots", "stall_hotspots_run.txt"),
            ("source_lines", "source_metrics_line_run.txt"),
            ("sass", "disasm_run.txt"),
        ):
            if isinstance(payload.get(key), str):
                (artifacts_dir / relative).write_text(payload[key], encoding="utf-8")
    ok = (
        snapshot.status == "succeeded"
        and not response.get("error")
        and response.get("command_ok", True) is not False
        and payload.get("command_ok", True) is not False
    )
    result = VerificationResult(
        "PASS" if ok else "ERROR",
        None,
        None,
        None,
        error=(
            ""
            if ok
            else str(
                payload.get("error")
                or response.get("error")
                or f"profile status={snapshot.status}"
            )
        ),
        artifact=str(pending.directory),
    )
    atomic_write_json(
        pending.directory / "transport_result.json",
        {"gateway": response, "profile": payload},
    )
    atomic_write_json(pending.result_path, result.as_dict())
    summary = payload.get("summary") or payload.get("profile_summary")
    compact_profile = {
        "level": (pending.detail or {}).get("level"),
        "kernel_name": (pending.detail or {}).get("kernel_name"),
        "kernel_regex": (pending.detail or {}).get("kernel_regex"),
        "source": (pending.detail or {}).get("source"),
        "route": detail.get("route"),
        "summary": summary[:4000] if isinstance(summary, str) else summary,
        "artifacts": (
            payload.get("artifacts")[:20]
            if isinstance(payload.get("artifacts"), list)
            else payload.get("artifacts")
        ),
    }
    write_agent_result(
        pending.directory,
        result,
        evaluation_id=pending.evaluation_id,
        backend=pending.backend,
        request_digest=pending.request_digest,
        profile=compact_profile,
        max_bytes=pending.agent_result_max_bytes,
    )
    return result
