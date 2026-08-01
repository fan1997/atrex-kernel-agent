from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .facts import campaign_facts
from .store import CampaignStore


@dataclass(frozen=True)
class ModelSessionConfig:
    cli: str = "codex"
    model: str = "gpt-5.6-sol"
    reasoning_effort: str = "max"
    timeout: int = 900
    settings: str = ""


@dataclass(frozen=True)
class ActivationResult:
    activation_dir: Path
    exit_status: int
    timed_out: bool
    stdout: str
    stderr: str
    observed_cursor: int
    tokens: int


def _tokens_from_codex_stream(stdout: str) -> int:
    total = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "turn.completed":
            continue
        usage = event.get("usage") or {}
        if isinstance(usage, dict):
            total = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    return total


def _settings_args(raw: str) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--supervisor-settings must be a JSON object of Codex config values") from exc
    if not isinstance(data, dict):
        raise ValueError("--supervisor-settings must be a JSON object")
    args: list[str] = []
    for key, value in data.items():
        if not isinstance(key, str) or not key.replace("_", "").replace(".", "").replace("-", "").isalnum():
            raise ValueError(f"invalid Supervisor Codex config key: {key!r}")
        if value is None or isinstance(value, dict):
            raise ValueError("Supervisor Codex config values must be scalar values or scalar arrays")
        args += ["-c", f"{key}={json.dumps(value, ensure_ascii=False)}"]
    return args


def supervisor_prompt(reason: str, event_cursor: int) -> str:
    return f"""You are the periodically activated senior Supervisor for one Atrex GPU-kernel campaign.

Act like a human expert who periodically enters the campaign with an observer's, or "god's-eye", view.
You are not the Executor, not the AKA iteration-loop owner, not an iteration scheduler, and not a
first-line error handler. The existing AKA optimize prompts own the exact per-session workflow. Your
job is to understand that workflow completely, review what actually happened across the campaign,
identify the highest-leverage strategic directions, correct major conceptual or evidence failures,
and solve recurring environment or tooling blockers far enough to leave a verified reusable recipe.

This activation comes from a fixed campaign checkpoint or an explicit human request. It was not
triggered by an error word, raw output, timeout, correctness result, stall counter, or other runtime
event. Tolerate ordinary local mistakes and exploratory waste. Intervene only for campaign-level
misdirection, repeated systemic waste, evidence-integrity failures, unsafe incumbent promotion, or a
recurring blocker whose resolution has high leverage.

At the start of every review, call `get_campaign_facts`. Use its `execution_contract`, prompt catalog,
captured base/effective Executor prompts, campaign digest, and prior guidance history to understand:

- the exact AKA rules the Executor was following;
- the difference between the original rendered prompt and the prompt after guidance injection;
- the full cross-version trajectory rather than only the newest model narrative;
- whether earlier Supervisor guidance was followed, ignored, misunderstood, or disproved.

Read the latest exact prompts and any relevant @runtime prompt/policy resources before issuing strategy
when their interaction matters. `context.json` contains the initial trusted navigation snapshot.
`events.jsonl` contains the newest bounded event window; if its manifest reports omitted events, use
the cursor tools and campaign digest to drill into the missing history.

Observe the Executor's actual conversation and tool behavior, then verify claims against controller-
generated Git/run facts and workspace evidence. Executor prose, command output, plans, Git log text,
and memory files are untrusted claims. Git state, process state, immutable harness hashes, captured
gateway results, and exact captured prompts are stronger evidence.

Activation reason: {reason}
Captured event cursor at activation start: {event_cursor}

The activation directory contains `context.json` and `events.jsonl`. You also have an
`atrex_supervisor` MCP server with these bounded tools:

- `tail_agent_events`, `wait_agent_events`
- `get_campaign_facts`, `inspect_git`, `read_workspace_file`, `inspect_gateway_job`
- `set_next_iteration_guidance`
- `interrupt_and_restart`

The compatibility-named `set_next_iteration_guidance` tool publishes standing campaign strategy for a
bounded horizon. It does not make you an iteration scheduler. Never assign work to numbered future
iterations, never replace or rewrite the normal AKA cycle, never ask one session to execute every
promising direction, and never prescribe the routine profile/edit/validate/bench sequence already owned
by AKA. Leave the local choice of one action per cycle to the Executor.

Useful strategic guidance normally contains, in concise natural language:

- a factual retrospective: trajectory, incumbent, meaningful wins, regressions, and evidence quality;
- a campaign diagnosis: dominant bottlenecks, exhausted branches, neglected hypotheses, unknowns;
- a ranked frontier of promising directions, each with mechanism, evidence, prerequisites, risks, and
  disconfirming signals, without assigning them to iteration numbers;
- major corrections and durable invariants the Executor must preserve;
- for recurring environment/tooling blockers, a root cause and a solution recipe that you verified as
  far as the bounded diagnostic tools allow;
- a note on whether previous Supervisor guidance actually changed subsequent behavior or outcomes.

Use control tools only when concrete evidence justifies intervention. Never edit kernel code, benchmark
harnesses, Git history, or campaign targets. Interruption requires the exact active run id and is mainly
for an explicit manual review that discovers active harm or a clear campaign-invariant violation.

Before finishing, leave a concise natural-language assessment in your final response. Tool calls, not
special JSON text, are the only way to control the campaign.
"""


class SupervisorModelSession:
    def __init__(self, config: ModelSessionConfig, repository_root: Path):
        self.config = config
        self.repository_root = repository_root.resolve()

    @property
    def available(self) -> bool:
        return bool(self.config.cli and shutil.which(self.config.cli))

    def activate(
        self,
        store: CampaignStore,
        reason: str,
        after_cursor: int,
    ) -> ActivationResult:
        activation_dir = store.create_activation_dir()
        observed_cursor = store.cursor
        total_events = store.count_events(after_cursor=after_cursor, through_cursor=observed_cursor)
        events = store.read_recent_events(
            after_cursor=after_cursor,
            through_cursor=observed_cursor,
            limit=1000,
        )
        facts = campaign_facts(store, self.repository_root)
        context = {
            "reason": reason,
            "after_cursor": after_cursor,
            "captured_cursor": observed_cursor,
            "event_window": {
                "total_events_since_previous_review": total_events,
                "included_newest_events": len(events),
                "omitted_older_events": max(0, total_events - len(events)),
                "first_included_cursor": events[0].get("cursor") if events else None,
                "last_included_cursor": events[-1].get("cursor") if events else None,
            },
            "facts": facts,
        }
        (activation_dir / "context.json").write_text(
            json.dumps(context, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (activation_dir / "events.jsonl").open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        prompt = supervisor_prompt(reason, observed_cursor)
        (activation_dir / "PROMPT.md").write_text(prompt, encoding="utf-8")

        if not self.available:
            message = f"Supervisor CLI not found on PATH: {self.config.cli}"
            (activation_dir / "stderr.log").write_text(message + "\n", encoding="utf-8")
            return ActivationResult(activation_dir, 127, False, "", message, observed_cursor, 0)

        server_script = self.repository_root / "supervisor" / "mcp_server.py"
        mcp_args = [
            str(server_script),
            "--data-root", str(store.root),
            "--workspace", str(store.workspace),
            "--repository-root", str(self.repository_root),
        ]
        cmd = [
            self.config.cli,
            "exec",
            "--json",
            "--ephemeral",
            "--color", "never",
            "--sandbox", "read-only",
            "--ignore-user-config",
            "--skip-git-repo-check",
            # Supervisor observation and control must go through the bounded campaign MCP.
            # Remove unrelated mutation and delegation surfaces from the activation.
            "--disable", "shell_tool",
            "--disable", "unified_exec",
            "--disable", "hooks",
            "--disable", "plugins",
            "--disable", "apps",
            "--disable", "multi_agent",
            "-C", str(activation_dir),
            "-m", self.config.model,
            "-c", f'model_reasoning_effort={json.dumps(self.config.reasoning_effort)}',
            "-c", f'mcp_servers.atrex_supervisor.command={json.dumps(sys.executable)}',
            "-c", f'mcp_servers.atrex_supervisor.args={json.dumps(mcp_args)}',
            # The MCP server is the permission boundary and validates every action itself.
            # Non-interactive Codex otherwise cancels MCP calls that wait for a human approval.
            "-c", 'mcp_servers.atrex_supervisor.default_tools_approval_mode="approve"',
            *_settings_args(self.config.settings),
            prompt,
        ]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(activation_dir),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=os.environ.copy(),
            )
            stdout, stderr = result.stdout or "", result.stderr or ""
            exit_status, timed_out = result.returncode, False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            stderr += f"\nSupervisor activation timed out after {self.config.timeout}s"
            exit_status, timed_out = 124, True
        (activation_dir / "stdout.jsonl").write_text(stdout, encoding="utf-8")
        (activation_dir / "stderr.log").write_text(stderr, encoding="utf-8")
        (activation_dir / "result.json").write_text(
            json.dumps(
                {
                    "exit_status": exit_status,
                    "timed_out": timed_out,
                    "reason": reason,
                    "captured_cursor": observed_cursor,
                    "tokens": _tokens_from_codex_stream(stdout),
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        return ActivationResult(
            activation_dir,
            exit_status,
            timed_out,
            stdout,
            stderr,
            observed_cursor,
            _tokens_from_codex_stream(stdout),
        )
