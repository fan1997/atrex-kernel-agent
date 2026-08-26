# Repository Horizon v3 episode {{EPISODE}}

Own one complete engineering direction in this isolated Git worktree. Continue through as many
inspection, edit, compile, correctness, benchmark, profile, and repair cycles as the direction
needs. Planning is optional; `gen-plan` is never a required gate.

The supervisor, verification, canonical memory, recovery, and promotion semantics are the current
main Long Horizon implementation. You own only this episode branch and its structured evidence.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Locked source: `{{SOURCE_NAME}}` at `{{SOURCE_REVISION}}`
- Editable repository roots: {{EDITABLE_ROOTS}}
- Bounded source corpus: `{{SOURCE_CORPUS}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, and `reference-projects/` follow main and are linked into the
  worktree. `gpu-wiki/` and KernelWiki are intentionally absent.
{{AGENT_RUNTIME}}

Never switch branches, push, merge, rebase, or alter refs. Private checkpoint commits on this
episode branch are allowed. A terminal candidate may change only manifest-declared editable roots;
plans and profiles are ignored evidence and must not be committed. Never edit the adapter,
evaluator, private/public workload contract, source lock, source manifest, corpus catalog, vendored
runtime support, canonical memory, README, or agent policy files.

For a generalized production workload, `agent_problem.json` is the complete public contract.
Do not search outside the workspace for hidden evaluator cases. Exact `shapes.json` and release
metadata are staged only at the private verification boundary.

{{MODE_POLICY}}

{{HARDWARE}}

{{SANDBOX}}

{{REPOSITORY_SEARCH_REQUIREMENT}}

{{ROUTE_DIRECTIVE}}

## Execution boundary

All GPU imports, compilation, correctness, benchmarking, and profiling must use the local gateway.
The coding session has only the public contract: construct representative synthetic cases from
`agent_problem.json` and keep temporary drivers under ignored `profiles/`. The supervisor alone
owns exact hidden-shape correctness, measurement, and ABBA promotion. Do not invoke
`repository_horizon.dev_eval`, request `PROFILE_SHAPE_ID`, read a private profile case, or search
outside the workspace for evaluator data.

```bash
{{PUBLIC_DEV_COMMAND}}
```

Adapt the explicit allowlist and public driver path as needed. Development measurements are evidence
only. Final acceptance uses the supervisor's same-allocation ABBA verification and complete
hidden-shape coverage checks from current main.

## Engineering loop

Reconstruct the incumbent from canonical `memory/v*.json` and repository history. Choose one
coherent direction, gather only the evidence needed, implement it, repair concrete failures, and
record each decisive experiment. You may use repository-local skills or write a plan when helpful,
but no skill, Wiki lookup, profile, or generated plan is a mandatory stage.

Record experiments with:

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"...","hypothesis":"...","change":"...","evidence":"...","result":"...","decision":"continue|revert|pivot"}'
```

Publish a promotable candidate once it passes complete development correctness and has credible
performance evidence. Secondary tweaks belong to a later episode.

## Terminal contract

Reach exactly one evidence-backed state: `candidate_ready`, `pivot`, or `blocked`.

For `candidate_ready`, commit the exact repository source candidate, leave the worktree clean, then
finalize the journal:

```bash
git add -- {{EDITABLE_ROOTS_SHELL}}
git commit -m "v{{VERSION}}: repository candidate"
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."]}'
```

For `pivot` or `blocked`, finalize with that state and omit `--candidate-commit`. Every terminal
journal needs at least one experiment and a non-empty summary.

Only after finalizing, atomically publish complete JSON to `{{HANDOFF_PATH}}` via a temporary file:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. Do not claim an improvement merely to terminate; an evidence-backed
pivot is valid.
