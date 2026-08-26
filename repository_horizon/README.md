# Repository Horizon v3

Repository Horizon v3 optimizes a locked source-repository snapshot while delegating the campaign
state machine to the current `main` Long Horizon implementation.

## Preplan-only architecture frontier

The `feat/repository-horizon-v3-preplan` variant adds an explicitly pre-episode macro-planning
boundary. `--preplan-only` prepares the locked incumbent, starts one isolated max-effort session,
validates `plans/end_to_end_architecture_frontier.json`, copies the validated evidence to the
canonical workspace, and exits before episode 1. The session may run bounded public GPU probes and
small ignored prototypes, but a changed HEAD, any tracked workspace change, an invalid frontier, or
private-evaluator evidence rejects the run.

## Design contract

V3 intentionally differs from `main` in only two workflow choices:

1. GPU Wiki and KernelWiki are not linked or installed in the campaign runtime.
2. `gen-plan` remains available as a normal repository-local skill, but neither the episode prompt
   nor the supervisor requires it. Planning, profiling, and research are optional engineering tools.

Everything below is owned by current `main`:

- isolated episode worktrees and exact terminal handoffs;
- native Claude/Codex session recovery and token telemetry;
- pivot, blocked, rejected, and interrupted canonical-memory semantics;
- current mode/runtime identity and immutable workspace-policy lifecycle;
- same-allocation ABBA acceptance and squash promotion;
- aggregation of all ABBA repeats into canonical per-shape memory;
- public `agent_problem.json` plus evaluator-private exact production shapes;
- complete hidden-shape coverage checks.

`RepositoryHorizonCampaign` subclasses `long_horizon.campaign.LongHorizonCampaign` and does not
override `run()`. The small hooks in main are behavior-preserving defaults and allow v3 to replace
only runtime linking, candidate path validation, verification staging, and prompt rendering.

Repository-specific behavior remains responsible for:

- immutable source manifests and source locks;
- bounded source-history corpora;
- an optional clean reconnaissance seal that gates the first bring-up evaluation before any
  editable-source change while leaving `gen-plan` optional;
- manifest-declared editable roots;
- minimized and locked support wheels;
- repository-snapshot development evaluation and final ABBA staging.

For generalized production operators, `shapes.json`, `metadata.json`, and `roofline.json` are never
copied into the Agent worktree. They are injected only into an out-of-band verifier stage and are
scrubbed after collection. A nominal ABBA pass is rejected unless every scheduled incumbent and
candidate run reports every private shape id.

## Run

From the AKA repository root:

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
  --evaluation-wait-mode inline \
  --agent-cli codex \
  --optimization-mode production \
  --framework CuteDSL \
  --framework-baseline never \
  --no-workload-bucketing \
  --max-iters 20 \
  --workspace /path/to/campaign-root
```

V3 deliberately has no `--iter-timeout`: current main's `LongSessionRunner.run()` owns invocation
lifetime. It also has no suspended-evaluation supervisor; repository development evaluations run
inline so session and restart behavior remain identical to main.

Use `--preflight-only` to lock, stage, and author/validate the private-safe public problem, then run
the authoritative initial repository measurement without starting an optimization episode.

## Tests

```bash
python -m unittest discover -s repository_horizon/tests -v
```

The suite covers the main inheritance boundary, Wiki-free runtime, optional planning prompt,
manifest candidate contract, private shape staging, per-run hidden-shape completeness, source
corpus integrity, support-wheel locking, and transport behavior.

## V2 migration

V2 workspaces should not be resumed in place. V3 changes the supervisor and canonical-memory
semantics to current main, removes the old `iter_timeout` and custom suspended-session runner, and
keeps exact production cases private. Start a new v3 workspace from the same immutable source
manifest and revision; retain the v2 workspace only as replay evidence.
