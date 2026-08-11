#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


PREFIX = "__REPOSITORY_HORIZON_PROFILE_RESULT__="


def _profile_target_python(environment: dict[str, str]) -> str:
    """Use an explicitly pinned runtime, otherwise preserve the launching Python."""
    return environment.get("ATREX_LOCAL_PYTHON") or sys.executable


def _safe(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _apply(root: Path, stage: Path, manifest: dict[str, str | None]) -> None:
    for raw, source in manifest.items():
        target = root / _safe(raw)
        if source is None:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            continue
        snapshot = stage / _safe(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, target)


def _text(path: Path, limit: int = 64 * 1024) -> str | None:
    if not path.is_file() or path.stat().st_size > limit:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _manifest(root: Path) -> list[dict[str, object]]:
    values = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        values.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return values


def main() -> int:
    stage = Path.cwd()
    request = json.loads((stage / "request.json").read_text(encoding="utf-8"))
    profile = json.loads((stage / "profile_request.json").read_text(encoding="utf-8"))
    run_root = stage / ".profile_run"
    artifacts = stage / "artifacts"
    try:
        shutil.copytree(stage / "runtime", run_root)
        _apply(run_root, stage, request["manifests"]["candidate"])
        env = os.environ.copy()
        env["ATREX_PROFILE_TARGET_PYTHON"] = _profile_target_python(env)
        roots = [str(run_root / item) for item in request.get("python_roots", [])]
        env["PYTHONPATH"] = os.pathsep.join(
            roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        cache = stage / ".jit_cache"
        env["CUTE_DSL_CACHE_DIR"] = str(cache / "cute")
        env["TRITON_CACHE_DIR"] = str(cache / "triton")
        env["TMPDIR"] = str(cache / "tmp")
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        command = [
            "bash",
            str(stage / "tools" / "profile_nvidia.sh"),
            str(run_root / "test_kernel.py"),
            "--output-dir",
            str(artifacts),
            "--launch-skip",
            str(profile.get("launch_skip", 0)),
            "--launch-count",
            str(profile.get("launch_count", 1) or 1),
            "--ncu-helpers",
            str(stage / "tools" / "ncu_helpers"),
        ]
        kernel_filter = str(profile.get("kernel_filter") or "")
        if kernel_filter:
            command += ["--kernel-name", kernel_filter]
        if profile.get("source"):
            command.append("--source")
        if profile.get("level") == "survey":
            command.append("--no-classify")
        process = subprocess.run(
            command,
            cwd=str(run_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=int(profile.get("job_timeout", 600)),
        )
        (stage / "profile.stdout.log").write_text(process.stdout or "", encoding="utf-8")
        (stage / "profile.stderr.log").write_text(process.stderr or "", encoding="utf-8")
        payload = {
            "command_ok": process.returncode == 0,
            "exit_code": process.returncode,
            "error": None if process.returncode == 0 else "repository profile wrapper failed",
            "summary": _text(artifacts / "summary.txt"),
            "metrics": _text(artifacts / "analysis" / "metrics_key_run.txt"),
            "hotspots": _text(artifacts / "analysis" / "stall_hotspots_run.txt"),
            "source_lines": _text(
                artifacts / "analysis" / "source_metrics_line_run.txt"
            ),
            "sass": _text(artifacts / "analysis" / "disasm_run.txt"),
            "artifacts": _manifest(artifacts) if artifacts.is_dir() else [],
        }
    except Exception as exc:
        payload = {
            "command_ok": False,
            "exit_code": -1,
            "error": f"{type(exc).__name__}: {exc}",
            "artifacts": _manifest(artifacts) if artifacts.is_dir() else [],
        }
    print(PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0 if payload["command_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
