#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

RESULT_PREFIX = "[test_kernel] RESULT_JSON="
ABBA_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="


def _safe(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _apply(root: Path, stage: Path, manifest: dict) -> None:
    for raw, source in manifest.items():
        relative = _safe(raw)
        target = root / relative
        if source is None:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)
            continue
        snapshot = stage / _safe(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot, target)


def _result(stdout: str):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            value = json.loads(line[len(RESULT_PREFIX) :])
            return value if isinstance(value, dict) else None
    return None


def _version_matches(actual: str, constraint: str) -> bool:
    if constraint.startswith(">="):

        def parts(value: str):
            result = []
            for item in value.split("+")[0].split("."):
                digits = "".join(character for character in item if character.isdigit())
                result.append(int(digits or 0))
            return tuple(result)

        return parts(actual) >= parts(constraint[2:])
    if constraint.startswith("=="):
        constraint = constraint[2:]
    return actual == constraint


def _check_runtime(requirements: list) -> None:
    for item in requirements:
        distribution = item["distribution"]
        import_name = item["import"]
        constraint = item["version"]
        if importlib.util.find_spec(import_name) is None:
            raise RuntimeError(f"required import is unavailable: {import_name}")
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required distribution is unavailable: {distribution}"
            ) from exc
        if not _version_matches(actual, constraint):
            raise RuntimeError(
                f"runtime version mismatch: {distribution}={actual}, required {constraint}"
            )


def main() -> int:
    stage = Path.cwd()
    request = json.loads((stage / "request.json").read_text(encoding="utf-8"))
    runs = []
    try:
        schedule = request["schedule"]
        manifests = request["manifests"]
        command = list(request["command"])
        local_python = os.environ.get("ATREX_LOCAL_PYTHON", "")
        if local_python and command and command[0] in {"python", "python3"}:
            # A local Codex invocation may resolve the prompt's bare Python to
            # a system interpreter. Keep the nested benchmark worker on the
            # launcher-pinned GPU runtime as well. Remote Agate jobs do not set
            # this variable and retain their portable `python3` command.
            command[0] = local_python
        timeout = int(request["run_timeout_seconds"])
        probe_roots = [
            str(stage / "runtime" / item) for item in request.get("python_roots", [])
        ]
        sys.path[:0] = probe_roots
        _check_runtime(request.get("runtime_requirements", []))
        run_root = stage / ".runs"
        cache_root = stage / ".jit_cache"
        for index, step in enumerate(schedule):
            revision = step["revision"]
            repeat = int(step["repeat"])
            root = run_root / f"{index:02d}_{revision}_{repeat}"
            shutil.copytree(stage / "runtime", root)
            _apply(root, stage, manifests[revision])
            env = os.environ.copy()
            roots = [str(root / item) for item in request.get("python_roots", [])]
            env["PYTHONPATH"] = os.pathsep.join(
                roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
            )
            cache = cache_root / revision
            cache.mkdir(parents=True, exist_ok=True)
            env["CUTE_DSL_CACHE_DIR"] = str(cache / "cute")
            # The evaluator starts one fresh Python/CUDA process per shape.  Keep
            # the persistent CuTe artifact cache enabled for every ABBA arm so
            # incumbent and candidate pay the same compilation policy while
            # reusing artifacts across shapes and repeats.
            env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
            env["TRITON_CACHE_DIR"] = str(cache / "triton")
            env["TMPDIR"] = str(cache / "tmp")
            Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
            try:
                process = subprocess.run(
                    command,
                    cwd=str(root),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                runs.append(
                    {
                        "revision": revision,
                        "repeat": repeat,
                        "exit_code": process.returncode,
                        "result": _result(process.stdout),
                        "stdout_tail": process.stdout[-4000:],
                        "stderr_tail": process.stderr[-4000:],
                    }
                )
            except subprocess.TimeoutExpired as exc:
                runs.append(
                    {
                        "revision": revision,
                        "repeat": repeat,
                        "exit_code": -1,
                        "result": None,
                        "stdout_tail": str(exc.stdout or "")[-4000:],
                        "stderr_tail": "evaluation timed out",
                    }
                )
        payload = {"schema_version": 1, "runs": runs, "error": None}
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "runs": runs,
            "error": f"{type(exc).__name__}: {exc}",
        }
    (stage / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(ABBA_PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
