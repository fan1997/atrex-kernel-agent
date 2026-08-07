from __future__ import annotations

import signal
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from long_horizon import main_adapter
from long_horizon.models import EpisodeHandoff, InvocationObservation, SessionResult
from long_horizon.protocol import handoff_diagnosis, read_handoff
from long_horizon.session import (
    CommandExecutor,
    CompletionCheck,
    _claude_transient_api_error,
    _codex_invocation_usage,
    _replace_terminal_usage,
)
from orchestrator.agent_runtime.codex_ledger import CodexSessionLedgerObserver, codex_home
from orchestrator.agent_runtime.model import TokenUsage, token_usage_exceeds

from .evaluation import evaluation_handoff_path


EvaluationWaiter = Callable[[Path], str]


class RepositorySessionRunner:
    """Run one episode while the supervisor, never the Agent, waits for Agate."""

    def __init__(
        self,
        *,
        evaluation_waiter: EvaluationWaiter,
        executor: CommandExecutor | None = None,
        agent_cli: str = "claude",
        max_evaluations: int = 32,
    ):
        self.evaluation_waiter = evaluation_waiter
        self.executor = executor or main_adapter.run_bounded
        self.agent_cli = agent_cli
        self.max_evaluations = max(1, max_evaluations)

    def run(
        self,
        workspace: Path,
        prompt: str,
        *,
        timeout: int,
        handoff_path: Path,
        handoff_resumes: int,
        completion_check: CompletionCheck,
        reasoning_effort: str = "max",
        session_id: str = "",
        telemetry_environment: Mapping[str, str] | None = None,
    ) -> SessionResult:
        requested_session_id = session_id or str(uuid.uuid4())
        is_codex = self.agent_cli == "codex"
        active_session_id = "" if is_codex else requested_session_id
        handoff_path.parent.mkdir(parents=True, exist_ok=True)
        handoff_path.unlink(missing_ok=True)
        evaluation_path = evaluation_handoff_path(workspace)
        evaluation_path.unlink(missing_ok=True)

        environment = main_adapter.session_environment(self.agent_cli)
        environment["IS_SANDBOX"] = "1"
        if telemetry_environment:
            environment.update(
                {str(key): str(value) for key, value in telemetry_environment.items()}
            )
        telemetry_attempt_prefix = environment.get("ATREX_TELEMETRY_ATTEMPT_ID")

        codex_observer = None
        codex_setup_errors: tuple[str, ...] = ()
        if is_codex:
            try:
                codex_observer = CodexSessionLedgerObserver(codex_home(environment))
            except Exception as exc:
                codex_setup_errors = (
                    f"codex_ledger_setup_failed:{type(exc).__name__}",
                )
        codex_stdout_session_usage: TokenUsage | None = None
        codex_ledger_usable = codex_observer is not None

        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        invocations: list[InvocationObservation] = []
        total_tokens = 0
        agent_seconds_remaining = float(timeout)
        completion_diagnosis = ""
        handoff: EpisodeHandoff | None = None
        exit_status = 0
        timed_out = False
        resume_count = 0
        repair_count = 0
        evaluation_count = 0
        turn_prompt = prompt
        fresh = True

        while True:
            if agent_seconds_remaining <= 0:
                timed_out = True
                exit_status = -1
                completion_diagnosis = "engineering budget expired before a valid terminal handoff"
                break
            if fresh:
                command = main_adapter.fresh_session_command(
                    turn_prompt,
                    requested_session_id,
                    reasoning_effort,
                    self.agent_cli,
                )
            else:
                if not active_session_id or not main_adapter.supports_same_session_resume(
                    self.agent_cli
                ):
                    completion_diagnosis = (
                        completion_diagnosis
                        or f"{self.agent_cli} ended without exposing a resumable session id"
                    )
                    break
                command = main_adapter.resume_session_command(
                    turn_prompt,
                    active_session_id,
                    reasoning_effort,
                    self.agent_cli,
                )
                resume_count += 1

            if telemetry_attempt_prefix:
                environment["ATREX_TELEMETRY_ATTEMPT_ID"] = (
                    f"{telemetry_attempt_prefix}-{len(invocations) + 1}"
                )
            turn_timeout = max(1, int(agent_seconds_remaining))
            if handoff_resumes > repair_count and turn_timeout > 610:
                turn_timeout -= 600
            started = time.monotonic()
            stdout, stderr, exit_status, turn_timed_out = self.executor(
                command,
                workspace,
                turn_timeout,
                environment,
            )
            agent_seconds_remaining -= time.monotonic() - started
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            observed_session_id = main_adapter.session_id_from_stream(
                self.agent_cli, stdout, requested_session_id
            )
            if observed_session_id:
                active_session_id = observed_session_id

            stream_session_usage = (
                main_adapter.terminal_usage_from_stream(stdout) if is_codex else None
            )
            events, terminal_usage, capabilities, observation_errors = (
                main_adapter.normalize_stream(
                    self.agent_cli,
                    stdout,
                    session_id=active_session_id,
                    codex_observer=(codex_observer if codex_ledger_usable else None),
                )
            )
            if not invocations and codex_setup_errors:
                observation_errors = codex_setup_errors + observation_errors
            ledger_failed = any(
                value.startswith("codex_ledger_unavailable:")
                for value in observation_errors
            )
            if ledger_failed:
                codex_ledger_usable = False
            resume_usage_qualified = bool(
                is_codex
                and capabilities.usage_delta_observed
                and not ledger_failed
            )
            if is_codex and not resume_usage_qualified:
                fallback_usage = _codex_invocation_usage(
                    stream_session_usage or TokenUsage.unavailable(),
                    codex_stdout_session_usage,
                )
                if fallback_usage.total_tokens is not None:
                    terminal_usage = fallback_usage
                    events = _replace_terminal_usage(events, fallback_usage)
                else:
                    terminal_usage = TokenUsage.unavailable()
                    events = _replace_terminal_usage(events, terminal_usage)
                    observation_errors += (
                        "codex_cumulative_fallback_unavailable",
                    )
            if (
                stream_session_usage is not None
                and stream_session_usage.total_tokens is not None
                and (
                    codex_stdout_session_usage is None
                    or not token_usage_exceeds(
                        codex_stdout_session_usage, stream_session_usage
                    )
                )
            ):
                codex_stdout_session_usage = stream_session_usage
            total_tokens += (
                terminal_usage.total_tokens
                if terminal_usage.total_tokens is not None
                else (0 if is_codex else main_adapter.tokens_from_stream(stdout))
            )
            invocations.append(
                InvocationObservation(
                    terminal_usage=terminal_usage,
                    events=events,
                    capabilities=capabilities,
                    observation_errors=observation_errors,
                    resume_usage_qualified=resume_usage_qualified,
                )
            )
            timed_out = turn_timed_out

            if evaluation_path.is_file():
                evaluation_count += 1
                if evaluation_count > self.max_evaluations:
                    completion_diagnosis = (
                        f"episode exceeded {self.max_evaluations} external evaluations"
                    )
                    break
                if not active_session_id:
                    completion_diagnosis = (
                        "Agent submitted an evaluation without exposing a resumable session id"
                    )
                    break
                try:
                    turn_prompt = self.evaluation_waiter(evaluation_path)
                except Exception as exc:
                    turn_prompt = (
                        "The repository supervisor could not collect the submitted evaluation: "
                        f"{type(exc).__name__}: {exc}. Inspect the persisted evaluation evidence, "
                        "record the infrastructure outcome, and continue or publish an honest blocked handoff."
                    )
                evaluation_path.unlink(missing_ok=True)
                handoff_path.unlink(missing_ok=True)
                completion_diagnosis = ""
                timed_out = False
                exit_status = 0
                fresh = False
                continue

            observed = read_handoff(handoff_path)
            if observed is not None:
                completion_diagnosis = completion_check(observed)
                if not completion_diagnosis:
                    handoff = observed
                    exit_status = 0
                    timed_out = False
                    break
            else:
                completion_diagnosis = handoff_diagnosis(handoff_path)

            transient_api_error = (
                _claude_transient_api_error(stdout)
                if self.agent_cli == "claude"
                else ""
            )
            externally_terminated = exit_status in {
                -signal.SIGTERM,
                128 + signal.SIGTERM,
            }
            dependency_terminated = "dependency policy violation" in stderr.lower()
            recoverable_failure = (
                (externally_terminated or bool(transient_api_error))
                and not dependency_terminated
            )
            can_repair = (
                repair_count < max(0, handoff_resumes)
                and bool(active_session_id)
                and main_adapter.supports_same_session_resume(self.agent_cli)
                and agent_seconds_remaining > 0
                and (exit_status == 0 or timed_out or recoverable_failure)
            )
            if not can_repair:
                break
            repair_count += 1
            if timed_out:
                turn_prompt = (
                    "The active engineering budget is nearly exhausted. Stop further exploration. "
                    "Commit a coherent candidate if one exists, finalize the journal, and publish "
                    "candidate_ready, pivot, or blocked now."
                )
            elif transient_api_error:
                turn_prompt = (
                    f"The previous invocation hit transient {transient_api_error}. Resume the same "
                    "episode and publish a valid terminal handoff when the work is coherent."
                )
            else:
                turn_prompt = (
                    "Continue the same repository optimization episode. The previous invocation did "
                    f"not satisfy the terminal contract: {completion_diagnosis}. Resume concrete work "
                    "from the current Git worktree and publish a valid terminal handoff before stopping."
                )
            fresh = False

        return SessionResult(
            exit_status=exit_status,
            timed_out=timed_out,
            tokens=total_tokens,
            session_id=active_session_id or requested_session_id,
            resume_count=resume_count,
            handoff=handoff,
            stdout_tail="\n".join(stdout_parts)[-4000:],
            stderr_tail="\n".join(stderr_parts)[-4000:],
            completion_diagnosis=completion_diagnosis,
            invocations=tuple(invocations),
        )
