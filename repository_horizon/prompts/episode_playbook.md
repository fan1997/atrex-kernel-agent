## Repository source-assisted optimization contract

This campaign optimizes an immutable snapshot of **{{SOURCE_NAME}}** at
`{{SOURCE_REVISION}}`. `kernel.py` is a fixed adapter and is not the candidate.

- Editable source roots: {{EDITABLE_ROOTS}}
- Source lock: `source.lock.json`
- Manifest: `source_manifest.json`
- Official evaluator inputs and the adapter are immutable.
- Make the smallest coherent source change under an editable root. New source
  modules are allowed there when they are part of the implementation.
- Never install packages. The Agate image is the dependency environment.
- Never import or JIT-compile FA4/CuTeDSL on the local host.
- For a development correctness/performance run, use exactly:

  ```bash
  {{DEV_EVAL_COMMAND}}
  ```

  This builds the same minimal staging payload used by promotion and runs it
  with `agate dev --working-dir`; do not use the ordinary single-file sandbox.
- Commit only source changes and optional `plans/` or `profiles/` evidence.
  Do not commit staging, caches, generated binaries, or evaluator results.
- Promotion authority is same-allocation A-B-B-A with isolated incumbent and
  candidate JIT caches. Timing is the official Atrex-Bench CUDA-event path.

Use source inspection, dispatch tracing, and focused experiments to select a
direction. Preserve the public adapter ABI and all declared shapes.
