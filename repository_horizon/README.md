# Autonomous Repository Horizon

`repository_horizon` is a self-contained repository-backed optimization overlay for the upstream
Long Horizon runtime. It adds no imports or hooks to existing `orchestrator/` or `long_horizon/`
files: dependency direction is strictly `repository_horizon -> upstream`.

The overlay preserves immutable source manifests, bounded source-history corpora, support-wheel
locking, isolated Git worktrees, repository-staged Agate evaluation, same-allocation ABBA, squash
promotion, restart state, and native Claude/Codex token telemetry. The coding Agent receives a
bounded autonomous brief instead of the upstream profile/research/plan workflow.

Autonomous guarantees:

- generated episode prompt is at most 16 KiB;
- history injection is at most 8 KiB and five attempts;
- no project Agent definitions, Humanize, KernelWiki, NCU skill, generic skills, or reference
  projects are linked into an episode;
- Codex multi-agent support is disabled through session settings by default;
- one coding session owns one episode, with bounded terminal-handoff recovery;
- development evaluation uses Agate `--no-wait`; the coding Agent suspends after submission while
  the repository supervisor waits and then resumes the same native Claude/Codex session;
- pending jobs and terminal results are persisted below
  `.repository_horizon_runtime/evaluations/`, so GPU queue time does not consume Agent tokens;
- failed, pivoted, blocked, and interrupted attempts are archived under
  `.atrex_long_horizon/episodes/` without advancing the incumbent branch;
- canonical `memory/vN.json` versions are created only by verified promotions.

Run from the repository root:

```bash
python -m repository_horizon \
  --source-manifest repository_horizon/recipes/fa4_fp8_paged_sm100.example.json \
  --source-checkout /path/to/flash-attention \
  --support-wheel quack-kernels=/path/to/quack_kernels-0.5.3-py3-none-any.whl \
  --op-dir /path/to/atrex-bench/operator \
  --platform B300 \
  --sandbox-hardware L20D \
  --sandbox-profile prod \
  --agent-cli codex \
  --optimization-mode production \
  --framework CuteDSL \
  --framework-baseline never \
  --no-workload-bucketing \
  --max-episodes 20 \
  --workspace /path/to/campaign-root
```

The generated incumbent workspace remains compatible with the existing repository manifest and
evaluator artifacts. A pre-existing clean workspace with a matching `source.lock.json` resumes from
its current Git HEAD; `.atrex_long_horizon/state.json` controls autonomous attempt recovery.

Run tests with:

```bash
python -m unittest discover -s repository_horizon/tests -v
```
