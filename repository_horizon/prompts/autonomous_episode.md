# Repository optimization episode {{EPISODE}}

Optimize the locked repository-backed kernel in this isolated Git worktree. You own the engineering
decisions inside this episode: inspect the source, edit, compile, measure, repair, or change strategy
as useful. Profiling, research, and planning are optional tools, not required stages.

## Objective and boundary

- Workspace: `{{WORKSPACE}}`
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Source: `{{SOURCE_NAME}}` at `{{SOURCE_REVISION}}`
- Editable roots: {{EDITABLE_ROOTS}}
- Bounded source corpus: `{{SOURCE_CORPUS}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}

{{REPOSITORY_SEARCH_REQUIREMENT}}

Only manifest-declared editable roots may change. Do not modify the adapter, evaluator, workload,
source lock, source manifest, source corpus catalog, memory, runtime support, README, CLAUDE.md, or
Git refs. Do not switch branches, push, merge, rebase, or rewrite the incumbent. Private checkpoint
commits on the current episode branch are allowed.

Own the engineering process. Profiling, research, planning, delegation, and checkpoint commits are
optional rather than required stages. Project workflow skills and project Agent definitions are not
installed; use ordinary source inspection and shell tools.

All GPU imports, compilation, correctness checks, benchmarks, and profiling must use:

```bash
{{DEV_EVAL_COMMAND}}
```

Optional typed profiling is available when the candidate/reference contract supports it:

```bash
{{PROFILE_COMMAND}} --level sol
```

Check `.repository_horizon_runtime/capabilities.json` first. Profiling is optional. A deep profile
requires an exact `--kernel-name` or anchored `--kernel-regex`; use `--source` only for source-line
evidence.

Configured wait mode: `{{EVALUATION_WAIT_MODE}}`.

{{EVALUATION_BEHAVIOR}}

Do not run `agate get`, development-evaluation status/collect subcommands, shell sleep loops, terminal
polling, or a second identical submission. Complete NCU logs and binary artifacts are persisted out of
band; the command returns only a compact result card. Read deeper evidence selectively when useful.

The external supervisor independently validates a candidate with repository-staged correctness and
same-allocation ABBA timing. Development measurements are evidence, not promotion authority.

## Recent attempts

```json
{{HISTORY}}
```

{{STALL_SIGNAL}}

## Terminal contract

Reach exactly one honest terminal state: `candidate_ready`, `pivot`, or `blocked`.

Record each decisive experiment:

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"...","hypothesis":"...","change":"...","evidence":"...","result":"...","decision":"continue|revert|pivot"}'
```

For `candidate_ready`, commit the exact candidate, make the worktree clean, then finalize:

```bash
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."]}'
```

For `pivot` or `blocked`, finalize with that state and omit `--candidate-commit`. A terminal journal
requires at least one experiment and a non-empty summary.

Finally write complete JSON to `{{HANDOFF_PATH}}.tmp` and atomically rename it to
`{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional"
}
```

Chat text is not a handoff. Do not claim an improvement merely to terminate; `pivot` is valid.
