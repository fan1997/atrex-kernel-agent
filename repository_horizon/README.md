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
- development evaluation accepts a dirty worktree snapshot; only final promotion requires an exact
  clean candidate commit;
- `--evaluation-wait-mode auto|inline|suspend` is user-controlled. Remote Claude/Codex defaults to
  suspend, localhost/direct-local and non-resumable Agents default to inline;
- suspend can be `graceful` or `enforced`; enforced mode terminates an invocation only after a valid
  atomic evaluation handoff and a configurable grace period;
- Agate and direct-local evaluation share one persistent pending/result protocol;
- pending jobs and terminal results are persisted below
  `.repository_horizon_runtime/evaluations/`, so GPU queue time does not consume Agent tokens;
- native session ID, cumulative token usage, invocation count, and remaining engineering time are
  checkpointed after every invocation. A restarted supervisor reattaches to the same Agate job ID
  or detached local PID and resumes the same Codex/Claude session;
- failed, pivoted, blocked, and interrupted attempts are archived under
  `.atrex_long_horizon/episodes/` without advancing the incumbent branch;
- canonical `memory/vN.json` versions are created only by verified promotions.
- NCU is optional. `--route auto` uses standard `agate profile` when the typed candidate contract is
  sufficient, otherwise it stages the repository snapshot through `agate dev` and reuses the
  upstream `tools/profile_nvidia.sh` plus `tools/ncu_helpers`. Direct-local uses the same wrapper;
- complete profiler output is persisted out of band. The Agent receives a bounded
  `agent_result.json` and selectively reads hotspots/source/SASS evidence.

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
  --evaluation-backend agate \
  --evaluation-wait-mode auto \
  --suspend-enforcement enforced \
  --agent-cli codex \
  --optimization-mode production \
  --framework CuteDSL \
  --framework-baseline never \
  --no-workload-bucketing \
  --max-episodes 20 \
  --workspace /path/to/campaign-root
```

Wait modes:

- `inline`: one blocking tool call returns the compact result to the current invocation;
- `suspend`: submit once, stop the invocation, wait in the Python supervisor, then resume the same
  native Codex/Claude session;
- `auto`: choose suspend only for remote, resumable Agents.

Qoder and Pi can use inline evaluation. They fail fast if suspend is explicitly requested because
the current upstream runtime has no same-session resume adapter for them.

Optional profiling is exposed through the bounded evaluation CLI, not through a workflow skill:

```bash
python -m repository_horizon.dev_eval profile \
  --workspace /path/to/episode \
  --hardware L20D --profile prod \
  --candidate kernel.py --reference-dir . \
  --level sol --route auto --wait-mode inline

python -m repository_horizon.dev_eval show \
  --evaluation /path/to/evaluation --section hotspots
```

Deep profiles require an exact kernel name or anchored regex. Raw `.ncu-rep`, full metrics, SASS,
and stdout/stderr are not injected into the Agent context. `show --section summary|hotspots|source-lines|sass|artifacts`
reads one bounded evidence slice; complete transport data stays in the evaluation directory.

The generated incumbent workspace remains compatible with the existing repository manifest and
evaluator artifacts. A pre-existing clean workspace with a matching `source.lock.json` resumes from
its current Git HEAD; `.atrex_long_horizon/state.json` controls autonomous attempt recovery.

Run tests with:

```bash
python -m unittest discover -s repository_horizon/tests -v
```
