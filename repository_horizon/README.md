# Repository Horizon

`repository_horizon` is an opt-in Long Horizon overlay for Python/JIT operator
libraries such as FA4. It snapshots an exact upstream Git commit into the
campaign, keeps the adapter and evaluator immutable, admits candidates only
under manifest-declared source roots, and verifies promotion with a single
Agate `dev --working-dir` allocation using A-B-B-A.

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

The FA4 recipe starts from the same auditable strategy as Mudi V0: unpack each
request's P64 or P128 paged KV and call official `flash_attn_func`. This is
correct but slow, giving the agent a runnable path from open-source FA4 toward
direct paged dispatch and the specialized HD256 2CTA implementation. The exact
Mudi evidence available on the development machine is:

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
