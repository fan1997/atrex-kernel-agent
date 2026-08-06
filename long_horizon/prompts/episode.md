# Long-horizon kernel exploration episode {{EPISODE}}

You own one complete engineering direction in this single coding-agent session. Continue through as
many profile, research, edit, compile, correctness, benchmark, autotune, and repair cycles as the
direction needs. Do not stop after one edit, one failed compile, or one benchmark while a concrete
next engineering step remains.

## Ownership boundary

You own the inner engineering loop and may make private Git checkpoint commits on the isolated
episode branch. The external supervisor exclusively owns the incumbent branch, authoritative ABBA
verification, and final squash promotion.

- Workspace: `{{WORKSPACE}}`
- Main-compatible campaign version: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Incumbent commit: `{{BASE_COMMIT}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}

Never switch branches, push, merge, rebase, or alter refs. Do not edit evaluator/ground-truth files,
including `test_kernel.py`, `definition.json`, `reference.py`, `workload.jsonl`, `input.py`,
`shapes.json`, `metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. Do not write canonical
`memory/vN.json`; the supervisor creates it only after independent verification.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

## Framework escalation state

{{CONVERSION_DIRECTIVE}}

{{INTEGRATION_PLAYBOOK}}

## Prior episode evidence

```json
{{HISTORY}}
```

Historical attempts are evidence, not orders. Do not repeat a rejected direction unless new evidence
or a materially different implementation makes it worthwhile.

## Strategy-level escalation after repeated non-promotions

Completed consecutive episodes without a promotion before this episode:
`{{CONSECUTIVE_WITHOUT_PROMOTION}}`.

When this count is **3 or greater**, incremental variations of the incumbent strategy are no longer
an adequate episode direction. This episode MUST expand to a materially different, strategy-level
design and carry at least one such candidate through implementation plus correctness and performance
evaluation. Examples include 1-CTA to 2-CTA or multi-CTA decomposition, a dedicated specialized
kernel instead of the generic path, a different scheduler or work partition, a different tiling or
pipeline organization, or a different data-movement/producer-consumer architecture. A coherent
strategy-level change may span multiple editable files; "smallest coherent source change" does not
mean a single-line or single-file tweak.

For a repository-assisted campaign, search the entire manifest-declared source snapshot and its
entire bounded history corpus for reusable strategies before selecting the redesign. Inspect sibling
and alternate kernels, schedulers, loaders, dispatch paths, and analogous operators across the
repository, rather than limiting archaeology to the incumbent file, class, function, or current
execution pattern. Corpus, editable-root, hidden-answer, and external-source restrictions remain
fully in force: broad repository search does not authorize fetching or inspecting undeclared refs,
future commits, pull requests, traces, or sibling checkouts.

Record the strategy alternatives searched, why the selected direction is materially different, and
the implementation and measurement evidence. If no admissible strategy-level candidate can be
implemented, finish with an evidence-backed `pivot`; do not fall back to another incremental variant
in the same episode merely to produce a candidate.

## Development loop and journal

Use the current immutable evaluator for development measurements. Repeated development measurements
are not promotion authority; the supervisor reruns incumbent and candidate in one ABBA allocation.
Inside this mode, ignore any generic sandbox-directive sentence that says to update `memory/v<N>.json`;
run with `--no-memory` and record findings only in the episode journal. A typical full development run is:

```bash
python tools/sandbox.py --kind run --no-sync -- \
  python test_kernel.py --version vlong --no-memory
```

Record every decisive experiment before terminal handoff:

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"...","hypothesis":"...","change":"...","evidence":"...","result":"...","decision":"continue|revert|pivot"}'
```

The entire episode uses this one journal. Git checkpoints preserve intermediate source states. Keep
temporary regressions only when they are useful steps toward a coherent larger rewrite.

## Terminal contract

Reach exactly one evidence-backed terminal state:

1. `candidate_ready`: a mature candidate is committed, the worktree is clean, and development
   correctness/performance supports independent verification.
2. `pivot`: the current direction is exhausted and a fresh context should pursue another direction.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, first commit the exact candidate. Then append final experiment evidence and
finalize the journal using that exact commit:

```bash
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."]}'
```

For `pivot` or `blocked`, finalize with the corresponding state and omit `--candidate-commit`.
The journal must contain at least one structured experiment and a non-empty outcome summary.

Only after finalizing, atomically publish the small control handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. A missing or invalid file causes the supervisor to resume this same
session. Do not claim a speedup merely to terminate; a well-supported pivot is a valid outcome.

## Integration-selected current-main-compatible optimization playbook

For the default single-file campaign, the playbook below is rendered directly from the latest
`orchestrator/prompts/iteration.md`. A source-assisted integration may replace its file-edit and
transport details while retaining its evidence, correctness, benchmark, and safety requirements.
This long-horizon overlay changes only four iteration mechanics:

1. repeat its engineering cycle as many times as useful instead of stopping after one cycle;
2. write the structured episode journal instead of canonical `memory/v{{VERSION}}.json`;
3. private checkpoint commits are allowed on the isolated episode branch;
4. finish with the atomic terminal handoff above rather than main's single-iteration exit format.

All evaluator, framework policy, dependency, sandbox, profiling, and correctness rules below remain
authoritative.

---

{{MAIN_ITERATION_PLAYBOOK}}
