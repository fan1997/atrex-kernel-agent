from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ABBA_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="


@dataclass(frozen=True)
class AgateDevResult:
    payload: dict[str, Any]
    stdout: str
    stderr: str
    job_id: str
    command: tuple[str, ...]


def _all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _payload(output: str) -> dict[str, Any]:
    candidates = [output]
    decoder = json.JSONDecoder()
    responses: list[dict[str, Any]] = []
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            responses.append(value)
            candidates.extend(_all_strings(value))
    for line in output.splitlines():
        try:
            candidates.extend(_all_strings(json.loads(line)))
        except json.JSONDecodeError:
            pass
    for candidate in candidates:
        for line in candidate.splitlines():
            index = line.find(ABBA_PREFIX)
            if index < 0:
                continue
            raw = line[index + len(ABBA_PREFIX) :].strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("Agate dev output has no repository ABBA result sentinel")


def _response(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    values = []
    for index, character in enumerate(output):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value and "result" in value:
            values.append(value)
    if not values:
        raise ValueError("Agate dev output has no terminal job object")
    return values[-1]


def _job_id(output: str) -> str:
    patterns = (
        r'"job_id"\s*:\s*"([^"]+)"',
        r'"id"\s*:\s*"([0-9a-fA-F-]{16,})"',
        r"\bjob[_ -]?id[=: ]+([0-9a-zA-Z-]{16,})",
    )
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def run_agate_dev(
    stage: Path,
    *,
    hardware: str,
    profile: str,
    url: str,
    job_timeout: int,
    wait_timeout: int,
) -> AgateDevResult:
    command = ["agate", "dev"]
    if url:
        command += ["--url", url]
    elif profile:
        command += ["--profile", profile]
    command += [
        "--gpu",
        hardware,
        "--working-dir",
        str(stage),
        "--job-timeout",
        str(job_timeout),
        "--wait-timeout",
        str(wait_timeout),
        "--intent",
        "custom_harness",
        "--note",
        "repository horizon same-allocation ABBA verification",
        "python3 repo_abba.py",
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=wait_timeout + 120
    )
    combined = process.stdout + "\n" + process.stderr
    if process.returncode != 0:
        raise RuntimeError(f"agate dev exited {process.returncode}: {combined[-5000:]}")
    response = _response(combined)
    result = response.get("result") if isinstance(response.get("result"), dict) else {}
    if (
        response.get("status") != "succeeded"
        or response.get("error")
        or response.get("command_ok") is not True
        or result.get("exit_code") != 0
    ):
        raise RuntimeError(
            "Agate dev terminal status rejected: "
            + json.dumps(
                {
                    "job_id": response.get("job_id"),
                    "status": response.get("status"),
                    "error": response.get("error"),
                    "command_ok": response.get("command_ok"),
                    "exit_code": result.get("exit_code"),
                    "stderr": result.get("stderr"),
                },
                ensure_ascii=False,
            )
        )
    payload = _payload(combined)
    return AgateDevResult(
        payload=payload,
        stdout=process.stdout,
        stderr=process.stderr,
        job_id=str(response.get("job_id") or _job_id(combined)),
        command=tuple(command),
    )
