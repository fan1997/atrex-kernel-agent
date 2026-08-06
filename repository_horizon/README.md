# Repository Horizon

`repository_horizon` is an opt-in Long Horizon overlay for Python/JIT operator
libraries such as FA4. It snapshots an exact upstream Git commit into the
campaign, keeps the adapter and evaluator immutable, admits candidates only
under manifest-declared source roots, and verifies promotion with Agate
`dev --working-dir`.

With `bringup.mode=auto`, the locked source snapshot is R0 rather than an
assumed runnable V0. Repository Horizon first runs a candidate-only capability
probe. If R0 passes, it becomes V0 without an agent episode. If it fails, Long
Horizon enters a correctness-only pre-V0 phase; failed episodes are recorded as
`memory/bootstrap_eXXXX.json` and the first correct source commit becomes V0.
V1 and later return to same-allocation A-B-B-A performance promotion. Existing
manifests default to `bringup.mode=disabled` and retain the original behavior.

`repository_search.mode=replay_strict` exposes a separate bare Git corpus that
contains only R0 and its ancestors. It is linked into episode worktrees for
source archaeology but excluded from Git and Agate staging. Declared excluded
answer commits must be physically absent. `allowlist` is available for
production campaigns that explicitly lock additional refs; arbitrary pull refs
and network discovery are never implicit.

Example:

```bash
python -m repository_horizon \
  --source-manifest repository_horizon/recipes/fa4_fp8_paged_sm100.example.json \
  --source-checkout /path/to/flash-attention \
  --support-wheel quack-kernels=/path/to/quack_kernels-0.5.3-py3-none-any.whl \
  --op-dir /path/to/flash_attention_fp8 \
  --platform B300 --sandbox-hardware L20D --sandbox-profile prod \
  --framework CuteDSL --framework-baseline never \
  --optimization-mode production --no-workload-bucketing \
  --agent-cli codex
```

Version 1 intentionally supports one source tree, explicit CuteDSL, a native
Atrex-Bench operator, and no layer decomposition or workload bucketing.
Pure-Python packages that are intentionally pinned with the baseline can be
declared as `runtime_support` and supplied with repeatable
`--support-wheel DISTRIBUTION=PATH` arguments. Repository Horizon verifies the
wheel's `METADATA`, extracts only the manifest allowlist plus its import closure,
generates an explicit minimal package shim when requested, and records the wheel
and output tree hashes in `source.lock.json`. The resulting `vendor_support/`
tree is added to `PYTHONPATH` and candidates cannot modify it. This is a
temporary bridge for missing image dependencies; normal production recipes
should prefer image-installed dependencies and remove the corresponding
`runtime_support` declaration.

The FA4 recipe locks official FA4 commit `b54df166` and uses a fixed native
`flash_attn_varlen_func` adapter with `seqused_k` and `page_table`. It does not
unpack paged KV into a dense fallback. At R0 the SM103 HD256 dispatch rejects
this capability, so the campaign must repair the real repository implementation
before it can create V0. Mudi PR #2726 commit `3e18806f` is declared as an
excluded hidden answer and is permitted only as an external validation fixture.
The example recipe requires a 1% A-B-B-A improvement after V0 so sub-percent
timing noise cannot promote a no-op repository change; manifests may choose a
different workload-appropriate threshold.
The exact Mudi evidence available on the development machine is:

```text
/Users/ruibo/work/trace/aka_fa4_history/kernel_opt_fa4_b300
/Users/ruibo/work/trace/aka_fa4_history/kernel_opt_fa4_b300/accepted/v60_overlay
/Users/ruibo/work/trace/aka_fa4_history/kernel_opt_fa4_b300/memory/v60.json
```

That trace workspace also contains the 46-shape P64 replay inputs. Use it as
`--op-dir` only for replay/audit; a clean production campaign should copy the
immutable workload bundle into its own Atrex-Bench data directory.

On completion, the overlay writes a full source archive and upstream-relative
source patch under `.repository_horizon_runtime/export/`.
