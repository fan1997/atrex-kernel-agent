## Repository source-assisted optimization contract

This campaign optimizes an immutable snapshot of **{{SOURCE_NAME}}** at
`{{SOURCE_REVISION}}`. `kernel.py` is a fixed adapter and is not the candidate.

- Campaign phase: **{{CAMPAIGN_PHASE}}**
- Bounded source-history corpus: `{{SOURCE_CORPUS}}`

- Editable source roots: {{EDITABLE_ROOTS}}
- Source lock: `source.lock.json`
- Manifest: `source_manifest.json`
- Official evaluator inputs and the adapter are immutable.
- Make the smallest coherent source change under an editable root. New source
  modules are allowed there when they are part of the implementation.
- If the phase is `bring-up`, performance improvement is not required. The
  candidate must make the fixed adapter pass the official workload without
  weakening the evaluator. The first passing source commit becomes V0.
- During bring-up, inspect the bounded corpus before inventing a new
  implementation. Use commands such as `git --git-dir {{SOURCE_CORPUS}} log`,
  `git --git-dir {{SOURCE_CORPUS}} grep`, and `git --git-dir {{SOURCE_CORPUS}}
  show`. Do not inspect repositories, traces, branches, pull requests, or
  commits outside this corpus.
- When the manifest requires source archaeology, commit
  `plans/repository_search.json` with schema version 1, the exact R0
  `source_revision`, non-empty `queries` and `candidates`, and a non-empty
  `selected` object explaining the closest reusable source path and capability
  gap. Every candidate needs a full corpus-visible `commit` and `path`, and
  `selected` must name one of those commits. Corpus refs and the complete
  physical object set are checked before verification. Benchmark claims in
  history are hypotheses, not acceptance evidence.
- Never install packages. The Agate image is the dependency environment.
- Never import or JIT-compile FA4/CuTeDSL on the local host.
- For a development correctness/performance run, use exactly:

  ```bash
  {{DEV_EVAL_COMMAND}}
  ```

  This builds the same minimal staging payload used by promotion and runs it
  with `agate dev --working-dir`; during bring-up it is candidate-only, and
  after V0 it uses incumbent/candidate comparison. Do not use the ordinary
  single-file sandbox.
- Commit only source changes and optional `plans/` or `profiles/` evidence.
  Do not commit staging, caches, generated binaries, or evaluator results.
- Promotion authority is same-allocation A-B-B-A with isolated incumbent and
  candidate JIT caches. Timing is the official Atrex-Bench CUDA-event path.

Use source inspection, dispatch tracing, and focused experiments to select a
direction. Preserve the public adapter ABI and all declared shapes.
