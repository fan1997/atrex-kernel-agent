from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from long_horizon.protocol import atomic_write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Detached repository evaluation worker"
    )
    parser.add_argument("--stage", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--command-json", default='["python3","repo_abba.py"]')
    args = parser.parse_args(argv)
    stage = Path(args.stage).resolve()
    stdout_path = stage.parent / "worker.stdout.log"
    stderr_path = stage.parent / "worker.stderr.log"
    status_path = stage.parent / "local_job_result.json"
    started = _now()
    try:
        command = json.loads(args.command_json)
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError("--command-json must be a non-empty JSON argv array")
        process = subprocess.run(
            command,
            cwd=str(stage),
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        stdout_path.write_text(process.stdout or "", encoding="utf-8")
        stderr_path.write_text(process.stderr or "", encoding="utf-8")
        value = {
            "status": "succeeded" if process.returncode == 0 else "failed",
            "command_ok": process.returncode == 0,
            "error": None if process.returncode == 0 else "local worker command failed",
            "result": {"exit_code": process.returncode},
            "started_at": started,
            "finished_at": _now(),
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(str(exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text(str(exc.stderr or ""), encoding="utf-8")
        value = {
            "status": "timed_out",
            "command_ok": False,
            "error": "local evaluation timed out",
            "result": {"exit_code": -1},
            "started_at": started,
            "finished_at": _now(),
        }
    except Exception as exc:
        value = {
            "status": "failed",
            "command_ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "result": {"exit_code": -1},
            "started_at": started,
            "finished_at": _now(),
        }
    atomic_write_json(status_path, value)
    return 0 if value["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
