from __future__ import annotations

import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .bridge import CampaignBridge, SupervisorRuntime
from .cutlass_compat import workspace_compatibility
from .models import AttemptResult, ControlRequest


def _prompt_kind(prompt: str) -> str:
    lowered = prompt.lower()
    if "setup session" in lowered or "produce the v0 baseline" in lowered:
        return "baseline"
    if "triton->gluon" in lowered or "triton→gluon" in lowered:
        return "convert"
    if "recombine" in lowered:
        return "recombine"
    if "decompose" in lowered:
        return "decompose"
    return "iteration"


def _append_guidance(prompt: str, guidance: str) -> str:
    if not guidance:
        return prompt
    return (
        prompt.rstrip()
        + "\n\n## Campaign Supervisor guidance\n\n"
        + guidance.strip()
        + "\n\nTreat this as independent macro guidance. Verify it against current profiler, Git, "
          "correctness, and workload evidence before acting.\n"
    )


def _terminate_process_group(proc: subprocess.Popen[str], grace_seconds: float = 3.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stream_attempt(
    cmd: list[str],
    cwd: Path,
    timeout: int,
    env: dict[str, str],
    bridge: CampaignBridge,
    run_id: str,
) -> AttemptResult:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=env,
    )
    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def reader(stream_name: str, handle: Any) -> None:
        try:
            for line in iter(handle.readline, ""):
                output_queue.put((stream_name, line))
        finally:
            output_queue.put((stream_name, None))

    assert proc.stdout is not None and proc.stderr is not None
    threads = [
        threading.Thread(target=reader, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=reader, args=("stderr", proc.stderr), daemon=True),
    ]
    for thread in threads:
        thread.start()

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    closed_streams: set[str] = set()
    deadline = time.monotonic() + timeout
    timed_out = False
    control_request: ControlRequest | None = None

    while len(closed_streams) < 2 or proc.poll() is None:
        if time.monotonic() >= deadline and proc.poll() is None:
            timed_out = True
            _terminate_process_group(proc, grace_seconds=0.5)
        if proc.poll() is None and control_request is None:
            control_request = bridge.poll_control(run_id)
            if control_request is not None:
                bridge.publish(
                    run_id,
                    "control_accepted",
                    raw=control_request.reason,
                    metadata={
                        "request_id": control_request.request_id,
                        "action": control_request.action,
                        "evidence_event_ids": list(control_request.evidence_event_ids),
                    },
                )
                _terminate_process_group(proc)
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except queue.Empty:
            if proc.poll() is not None and len(closed_streams) >= 2:
                break
            continue
        if line is None:
            closed_streams.add(stream_name)
            continue
        if stream_name == "stdout":
            stdout_parts.append(line)
        else:
            stderr_parts.append(line)
        bridge.publish(
            run_id,
            "agent_output",
            raw=line.rstrip("\n"),
            stream=stream_name,
        )

    for thread in threads:
        thread.join(timeout=1)
    proc.stdout.close()
    proc.stderr.close()
    return AttemptResult(
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        exit_status=proc.returncode if proc.returncode is not None else 124,
        timed_out=timed_out,
        interrupted=control_request is not None,
        control_request=control_request,
    )


def _workspace_clean(workspace: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and not result.stdout.strip()


def _git_head(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _restore_tracked_attempt(workspace: Path, pre_head: str, started_clean: bool) -> bool:
    """Restore tracked edits only when the attempt began clean and did not create a commit."""
    if not started_clean or not pre_head or _git_head(workspace) != pre_head:
        return False
    try:
        result = subprocess.run(
            ["git", "reset", "--hard", pre_head],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def supervised_run_session(
    base: Any,
    runtime: SupervisorRuntime,
    workspace: Path,
    prompt: str,
    timeout: int,
    agent_cli: str = "claude",
    sandbox_hardware: str = "",
    sandbox_profile: str = "",
    sandbox_url: str = "",
    sandbox_timeout: int = 600,
) -> Any:
    bridge = runtime.bridge_for(workspace)
    # A resumed SOL campaign has no setup model session.  Confirm its controller-owned v0
    # record before the first logical iteration so the baseline checkpoint is not skipped.
    confirm_existing_baseline = getattr(bridge, "confirm_existing_baseline", None)
    if confirm_existing_baseline is not None:
        confirm_existing_baseline()
    env = base._session_env(agent_cli)
    env["IS_SANDBOX"] = "1"
    if sandbox_hardware:
        env["ATREX_SANDBOX_GPU"] = sandbox_hardware
    if sandbox_url:
        env["ATREX_SANDBOX_URL"] = sandbox_url
        env.pop("ATREX_SANDBOX_PROFILE", None)
    elif sandbox_profile:
        env["ATREX_SANDBOX_PROFILE"] = sandbox_profile
        env.pop("ATREX_SANDBOX_URL", None)
    env["ATREX_SANDBOX_TIMEOUT"] = str(sandbox_timeout)

    all_stdout: list[str] = []
    all_stderr: list[str] = []
    total_tokens = 0
    restart_count = 0
    final_result: AttemptResult | None = None
    deadline = time.monotonic() + timeout
    prompt_kind = _prompt_kind(prompt)
    next_guidance = bridge.take_guidance()
    with workspace_compatibility(workspace):
        session_pre_head = _git_head(workspace)
        session_started_clean = _workspace_clean(workspace)

        while True:
            remaining = max(1, int(deadline - time.monotonic()))
            effective_prompt = _append_guidance(prompt, next_guidance)
            run_id = f"{bridge.campaign_id}:{uuid.uuid4().hex[:12]}"
            bridge.begin_run(run_id, agent_cli, restart_count + 1, prompt_kind)
            cmd = base._session_command(agent_cli, effective_prompt, str(uuid.uuid4()))
            result = _stream_attempt(cmd, workspace, remaining, env, bridge, run_id)
            bridge.end_run(run_id, result.exit_status, result.timed_out, result.interrupted)
            all_stdout.append(result.stdout)
            all_stderr.append(result.stderr)
            total_tokens += base._tokens_from_stream(result.stdout)
            final_result = result

            request = result.control_request
            if (
                request is None
                or request.action != "interrupt_and_restart"
                or restart_count >= runtime.config.max_restarts_per_session
                or time.monotonic() >= deadline
            ):
                break

            restart_count += 1
            restored = _restore_tracked_attempt(workspace, session_pre_head, session_started_clean)
            next_guidance = request.guidance or request.reason
            bridge.publish(
                run_id,
                "restart_prepared",
                raw=next_guidance,
                metadata={
                    "strategy": request.strategy,
                    "tracked_state_restored": restored,
                    "restart_count": restart_count,
                },
            )

    assert final_result is not None
    bridge.checkpoint(
        prompt_kind=prompt_kind,
        exit_status=final_result.exit_status,
        timed_out=final_result.timed_out,
    )
    total_tokens += bridge.take_supervisor_tokens()
    return base.SessionResult(
        exit_status=final_result.exit_status,
        timed_out=final_result.timed_out,
        tokens=total_tokens,
        stdout_tail="".join(all_stdout)[-2000:],
        stderr_tail="".join(all_stderr)[-2000:],
    )
