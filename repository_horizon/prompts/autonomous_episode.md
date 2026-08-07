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

Only manifest-declared editable roots may change. Do not modify the adapter, evaluator, workload,
source lock, source manifest, source corpus catalog, memory, runtime support, README, CLAUDE.md, or
Git refs. Do not switch branches, push, merge, rebase, or rewrite the incumbent. Private checkpoint
commits on the current episode branch are allowed.

Do the work in this one coding session. Delegation and project/global workflow skills are
unavailable; use ordinary source inspection and shell tools.

All GPU imports, compilation, correctness checks, benchmarks, and profiling must use:

```bash
{{DEV_EVAL_COMMAND}}
```

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
