from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from long_horizon.protocol import atomic_write_json

from .config import EvaluationPolicy


def _run(command: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        return process.returncode, (process.stdout + "\n" + process.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        output = str(exc.stdout or "") + "\n" + str(exc.stderr or "")
        return 124, (output + f"\nprobe timed out after {timeout}s").strip()


def probe_capabilities(
    policy: EvaluationPolicy,
    *,
    hardware: str,
    profile: str = "",
    url: str = "",
) -> dict[str, Any]:
    if policy.backend == "local":
        return {
            "schema_version": 1,
            "backend": "local",
            "hardware": hardware,
            "correctness": True,
            "benchmark": True,
            "compile_check": True,
            "ncu": shutil.which("ncu") is not None,
            "disassemble": shutil.which("cuobjdump") is not None,
            "probe_errors": [],
        }
    endpoint = ["--url", url] if url else (["--profile", profile] if profile else [])
    errors: list[str] = []
    version_code, version = _run(["agate", "--version"])
    profile_code, profile_help = _run(["agate", "profile", "--help"])
    limits_code, limits = _run(
        ["agate", "env", hardware, *endpoint, "--limits"], timeout=15
    )
    ncu_code, ncu = _run(
        ["agate", "env", hardware, *endpoint, "--counters", "ncu"], timeout=15
    )
    for name, code, output in (
        ("version", version_code, version),
        ("profile_help", profile_code, profile_help),
        ("limits", limits_code, limits),
        ("ncu", ncu_code, ncu),
    ):
        if code:
            errors.append(f"{name} probe exited {code}: {output[-500:]}")
    return {
        "schema_version": 1,
        "backend": "agate",
        "hardware": hardware,
        "agate_version": version.splitlines()[0] if version else "unknown",
        "correctness": True,
        "benchmark": True,
        "compile_check": profile_code == 0,
        "ncu": ncu_code == 0 and "ncu" in ncu.lower(),
        "disassemble": True,
        "profile_api": profile_code == 0 and "--level" in profile_help,
        "limits_summary": limits[-4000:],
        "probe_errors": errors,
    }


def install_capabilities(workspace: Path, capabilities: dict[str, Any]) -> Path:
    path = workspace / ".repository_horizon_runtime" / "capabilities.json"
    atomic_write_json(path, capabilities)
    return path
