from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .model_session import ModelSessionConfig, SupervisorModelSession
from .models import AgentEvent, ControlRequest
from .service import SupervisorService
from .store import CampaignStore


@dataclass(frozen=True)
class SupervisorConfig:
    data_root: Path
    repository_root: Path
    cli: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "max"
    settings: str = ""
    activation_timeout: int = 900
    every_iterations: int = 5
    max_activations: int = 100
    max_restarts_per_session: int = 2
    required: bool = False
    optimize_args: tuple[str, ...] = ()


def _redact_optimize_args(args: tuple[str, ...]) -> list[str]:
    """Keep the execution contract visible without persisting command-line secrets."""
    sensitive_markers = ("token", "secret", "password", "credential", "api-key", "api_key")
    result: list[str] = []
    redact_next = False
    for value in args:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        lowered = value.lower()
        if value.startswith("--") and any(marker in lowered for marker in sensitive_markers):
            if "=" in value:
                result.append(value.split("=", 1)[0] + "=<redacted>")
            else:
                result.append(value)
                redact_next = True
            continue
        if "://" in value:
            try:
                parsed = urlsplit(value)
                host = parsed.hostname or ""
                if parsed.port:
                    host += f":{parsed.port}"
                value = urlunsplit((parsed.scheme, host, parsed.path, "", ""))
            except ValueError:
                value = "<redacted-url>"
        result.append(value)
    return result


class CampaignBridge:
    def __init__(self, config: SupervisorConfig, workspace: Path):
        self.config = config
        self.store = CampaignStore(config.data_root, workspace)
        self._active_guidance: dict = {}
        existing_context = self.store.execution_context()
        optimize_args = (
            _redact_optimize_args(config.optimize_args)
            if config.optimize_args
            else list(existing_context.get("optimize_args") or [])
        )
        self.store.set_execution_context(
            {
                **existing_context,
                "repository_root": str(config.repository_root.resolve()),
                "optimize_args": optimize_args,
                "supervision": {
                    "activation_policy": "baseline + fixed completed-iteration cadence + before-stop + manual",
                    "every_completed_iterations": config.every_iterations,
                    "max_restarts_per_session": config.max_restarts_per_session,
                    "event_activation": False,
                },
                "guidance_contract": {
                    "kind": "standing_campaign_strategy",
                    "delivery": "appended after the original rendered AKA prompt",
                    "authority": "advisory strategy; original AKA execution contract remains authoritative",
                    "executor_interpretation": "apply the strategy while completing the normal one-cycle workflow; do not execute every direction in one session",
                },
            }
        )
        model = SupervisorModelSession(
            ModelSessionConfig(
                cli=config.cli,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                timeout=config.activation_timeout,
                settings=config.settings,
            ),
            repository_root=config.repository_root,
        )
        self.service = SupervisorService(
            self.store,
            model,
            every_iterations=config.every_iterations,
            max_activations=config.max_activations,
            required=config.required,
        )

    @property
    def campaign_id(self) -> str:
        return self.store.campaign_id

    def publish(
        self,
        run_id: str,
        kind: str,
        raw: str = "",
        stream: str = "control",
        metadata: dict | None = None,
    ) -> AgentEvent:
        recorded = self.store.append_event(
            AgentEvent(
                campaign_id=self.campaign_id,
                run_id=run_id,
                kind=kind,
                raw=raw,
                stream=stream,
                metadata=metadata or {},
            )
        )
        return recorded

    def begin_run(self, run_id: str, agent_cli: str, attempt: int, prompt_kind: str) -> None:
        prompt_metadata = next(
            (
                item
                for item in self.store.list_session_prompts(limit=5)
                if item.get("run_id") == run_id
            ),
            {},
        )
        data = {
            "campaign_id": self.campaign_id,
            "run_id": run_id,
            "status": "running",
            "agent_cli": agent_cli,
            "attempt": attempt,
            "prompt_kind": prompt_kind,
            "controller_pid": os.getpid(),
            "prompt_snapshot": prompt_metadata,
        }
        self.store.set_current_run(data)
        self.store.append_fact("run_started", data)
        self.publish(run_id, "session_started", metadata=data)

    def end_run(self, run_id: str, exit_status: int, timed_out: bool, interrupted: bool) -> None:
        data = {
            **self.store.current_run(),
            "run_id": run_id,
            "status": "interrupted" if interrupted else "finished",
            "exit_status": exit_status,
            "timed_out": timed_out,
        }
        self.store.set_current_run(data)
        self.store.append_fact("run_finished", data)
        self.publish(run_id, "session_finished", metadata=data)

    def take_guidance(self) -> str:
        guidance = self.store.consume_guidance(self.store.next_logical_iteration())
        self._active_guidance = guidance or {}
        return str((guidance or {}).get("message") or "").strip()

    def record_prompt(
        self,
        run_id: str,
        base_prompt: str,
        effective_prompt: str,
        prompt_kind: str,
        attempt: int,
    ) -> dict:
        return self.store.record_session_prompt(
            run_id=run_id,
            base_prompt=base_prompt,
            effective_prompt=effective_prompt,
            prompt_kind=prompt_kind,
            attempt=attempt,
            guidance_request_id=str(self._active_guidance.get("request_id") or ""),
        )

    def poll_control(self, run_id: str) -> ControlRequest | None:
        return self.store.consume_control(run_id)

    def checkpoint(
        self,
        prompt_kind: str,
        exit_status: int,
        timed_out: bool,
    ) -> None:
        self.service.session_checkpoint(prompt_kind, exit_status, timed_out)

    def confirm_existing_baseline(self) -> bool:
        """Bootstrap the fixed baseline checkpoint when a valid workspace is resumed.

        SOL baselines are produced by a controller-owned sandbox path and therefore do not
        necessarily pass through run_session.  Treat only an explicit PASS/PASS v0 record as a
        completed baseline; the service makes the resulting activation idempotent.
        """
        memory_path = self.store.workspace / "memory" / "v0.json"
        try:
            import json

            memory = json.loads(memory_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        correctness = str((memory.get("correctness") or {}).get("status") or "").upper()
        quality_gate = str((memory.get("quality_gate") or {}).get("result") or "").upper()
        if correctness != "PASS" or quality_gate != "PASS":
            return False
        self.service.session_checkpoint("baseline", exit_status=0, timed_out=False)
        return True

    def before_stop(self, reason: str = "orchestrator returned") -> bool:
        return self.service.before_stop(reason)

    def manual_review(self, reason: str) -> bool:
        return self.service.manual_review(reason)

    def take_supervisor_tokens(self) -> int:
        return self.service.take_unaccounted_tokens()

    def close(self) -> None:
        self.service.close()


class SupervisorRuntime:
    """Process-local bridge registry. No model behavior leaks into base optimize.py."""

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self._lock = threading.Lock()
        self._bridges: dict[Path, CampaignBridge] = {}

    def bridge_for(self, workspace: Path) -> CampaignBridge:
        key = workspace.resolve()
        with self._lock:
            bridge = self._bridges.get(key)
            if bridge is None:
                bridge = CampaignBridge(self.config, key)
                self._bridges[key] = bridge
            return bridge

    def close(self) -> None:
        with self._lock:
            bridges = list(self._bridges.values())
        for bridge in bridges:
            bridge.close()

    def before_stop(self, reason: str = "orchestrator returned") -> bool:
        with self._lock:
            bridges = list(self._bridges.values())
        success = True
        for bridge in bridges:
            success = bridge.before_stop(reason) and success
        return success
