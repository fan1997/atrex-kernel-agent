# Atrex Campaign Supervisor (experimental V1)

The Supervisor route is an independent entry point layered over the existing orchestrator. It does
not modify `orchestrator/optimize.py` and does not require executor agents to return a Supervisor-
specific schema.

```bash
python -m supervisor.optimize \
  --supervisor-model gpt-5.6-sol \
  --supervisor-data-root /path/outside/executor/workspaces/atrex-supervisor-data \
  --op-dir /path/to/op \
  --platform TARGET_GPU --sandbox-hardware REMOTE_GPU \
  --framework Triton --agent-cli codex
```

All non-`--supervisor-*` arguments are passed unchanged to `orchestrator/optimize.py`. The original
route remains available as usual:

```bash
python orchestrator/optimize.py ...
```

## Architecture

The supervised entry point temporarily replaces the imported orchestrator's `run_session` function
inside the supervised process only. The adapter preserves the normal setup, quality gates, stall
logic, framework dispatch, rollback, packaging, and stopping behavior.

Each executor session is streamed through `supervisor/session_runner.py`. Raw stdout/stderr lines,
controller facts, and interventions are stored outside the executor workspace under:

```text
<data-root>/campaigns/<workspace-derived-id>/
├── raw-events.jsonl
├── trusted-facts.jsonl
├── activations/
├── sessions/                  # exact base/effective Executor prompts per run
├── control/
├── guidance/
└── state/
```

## Supervisor role

The strong model is a periodically activated senior campaign expert, not a replacement loop and not
an iteration scheduler. The base AKA prompt remains authoritative for the exact clean-session workflow.
The Supervisor reconstructs the cross-version trajectory, verifies claims, identifies the most valuable
optimization frontier, retires exhausted branches, corrects major evidence or incumbent mistakes, and
investigates recurring environment/tooling blockers. It leaves the single local action for each cycle to
the Executor.

Standing guidance must not allocate work to numbered future iterations or restate the ordinary
profile/edit/validate/bench workflow. A validity horizon means that the same campaign strategy remains
visible for several normal AKA cycles; it is not a multi-step schedule.

## God's-eye execution context

Before every Executor attempt the wrapper durably captures:

- `base-prompt.md`: the exact rendered AKA prompt before Supervisor advice;
- `effective-prompt.md`: the exact prompt delivered to the Executor;
- hashes, prompt kind, attempt number, and the injected guidance id.

At activation, `get_campaign_facts` exposes:

- controller/Git/process facts;
- sanitized base optimize arguments and guidance-injection semantics;
- the prompt catalog for setup/iteration/conversion/recombination and production policy;
- recent exact prompt snapshots;
- a compact cross-version memory trajectory and run outcomes;
- prior Supervisor guidance, its deliveries, and lifecycle state.

The bounded `read_workspace_file` tool also understands virtual resources:

```text
@executor/latest/base
@executor/latest/effective
@executor/<session-id>/base
@executor/<session-id>/effective
@runtime/orchestrator/prompts/iteration.md
@runtime/orchestrator/optimization_policy.py
@runtime/reference/CLAUDE.md
```

Activation artifacts contain the newest 1,000 events since the prior review plus an explicit omission
count. The campaign digest and cursor tools provide navigation into older evidence, avoiding the old
failure mode where the first 1,000 events silently hid the end of a long review window.

The long-lived service maintains campaign state, while every strong-model observation is a fresh
Codex `exec --ephemeral` activation. The Supervisor runs with a read-only shell sandbox and receives
campaign actions through a separate stdio MCP server. The executor does not receive these tools.

## V1 tools

Observation:

- `tail_agent_events`
- `wait_agent_events`
- `get_campaign_facts`
- `inspect_git`
- `read_workspace_file`
- `inspect_gateway_job` (raw captured evidence in V1)

Control:

- `set_next_iteration_guidance`
- `interrupt_and_restart`

`set_next_iteration_guidance` retains its V1 name for compatibility, but now publishes
`standing_campaign_strategy`. Its MCP description and Executor framing explicitly prohibit treating the
message as an alternate iteration workflow or a checklist to finish in one session.

`inspect_gateway_job` also supports fixed non-mutating diagnostics without expanding the V1 tool
surface: local toolchain versions, workspace layout/static imports, and Python/shell syntax checks.
Syntax checks copy files to a temporary directory before execution. Gateway jobs remain observation-only;
the Supervisor cannot mutate or cancel shared gateway state.

The Supervisor cannot directly edit the workspace, run arbitrary Git mutations, change hard budgets,
or submit GPU work. A guarded interrupt requires the exact active run id. The parent runner executes
the process-group termination and may restore tracked edits only when the supervised session began
clean and no commit was created.

## Fixed V1 activation policy

The strong Supervisor is deliberately not event-triggered. Raw executor text, command output, Git log
text, memory contents, non-zero exits, timeouts, correctness failures, restarts, and stall counters are
recorded as evidence but never wake the strong model.

Automatic activation happens only:

- once after the baseline session completes successfully;
- after every 5 completed non-baseline logical iterations by default;
- once immediately before the base campaign executes its original STOP/finalize path; the
  supervised process exit also has an idempotent fallback for non-standard campaign paths.

An internal restart/attempt remains part of the same logical iteration and does not advance the
schedule. Non-baseline iterations count when their supervised session returns, whether they improved,
regressed, or failed. This makes cost and timing predictable while allowing the executor to absorb
ordinary mistakes and exploratory waste.

Supervisor failure is fail-open by default so the original mechanical orchestrator continues. Pass
`--supervisor-required` to make a failed checkpoint activation stop the supervised entry point instead.

Codex `turn.completed` usage from finished Supervisor activations is added to the next executor
`SessionResult`, so the base orchestrator's existing `--token-budget` accounting includes Supervisor
cost rather than treating observations as free.

Useful options:

```text
--supervisor-cli codex
--supervisor-model gpt-5.6-sol
--supervisor-reasoning-effort max
--supervisor-settings '{"model_reasoning_effort":"xhigh"}'
--supervisor-activation-timeout 900
--supervisor-every-iterations 5
--supervisor-max-activations 100
--supervisor-max-restarts 2
--supervisor-required
```

Run an explicit manual review without starting the optimizer:

```bash
python -m supervisor.optimize \
  --supervisor-model gpt-5.6-sol \
  --supervisor-activate-now /path/to/existing/workspace \
  --supervisor-manual-reason "review the current optimization path before resuming"
```

Manual review uses the same bounded MCP tools and provider settings. If the workspace belongs to a
currently active supervised process, guarded interruption still requires its exact active run id.

Pending strategy uses a newest-wins policy and is scoped to a logical-iteration validity horizon. A newer
campaign strategy supersedes older pending advice; legacy unscoped V1 guidance is not replayed after a
resume. The horizon controls delivery lifetime only and never assigns a direction to a particular cycle.

The Supervisor intentionally starts Codex with `--ignore-user-config` so unrelated user MCP servers
and approval settings are not inherited. Shell execution, hooks, plugins, apps, and multi-agent
delegation are disabled for Supervisor activations; campaign observation/control goes through the
bounded MCP server. If the selected model uses a custom provider, pass the
provider's non-secret config explicitly. Secrets remain in the referenced environment variable:

```bash
export ATREX_SUPERVISOR_SESSION_SETTINGS='{
  "model_provider":"company",
  "model_providers.company.name":"Company Gateway",
  "model_providers.company.base_url":"https://gateway.example/v1",
  "model_providers.company.wire_api":"responses",
  "model_providers.company.env_key":"COMPANY_OPENAI_KEY"
}'
```

The same JSON may be passed directly with `--supervisor-settings`. The campaign tool server is
explicitly auto-approved because non-interactive Codex otherwise cancels MCP calls awaiting a human;
the MCP implementation itself remains the action permission boundary.

Use `python -m supervisor.optimize --supervisor-help` for the Supervisor-only options.

## Security boundary

Executor prose, plans, and memory are treated as untrusted claims. The Supervisor is instructed to
cross-check them against controller-generated process/Git facts and captured gateway output. The
actual boundary is the MCP tool implementation and parent runner, not model obedience.

The existing executor backends still use their original permissions. For stronger isolation, place
`--supervisor-data-root` outside all campaign workspaces and combine this route with a future executor
workspace-write sandbox rather than relying on path secrecy alone.
