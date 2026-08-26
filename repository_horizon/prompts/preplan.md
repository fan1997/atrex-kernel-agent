# Repository Horizon V3 Preplan: end-to-end architecture frontier

This is a **plan-only macro-strategy session before episode 1**. Build a scientific search space for
the end-to-end workload. You may inspect and measure, but you must not implement a candidate, modify
campaign source, commit, publish a handoff, invoke private evaluation, or begin an optimization episode.

## Public context

- Workspace: `{{WORKSPACE}}`
- Platform/framework: `{{PLATFORM}}` / `{{FRAMEWORK}}`
- Locked source: `{{SOURCE_NAME}}` at `{{SOURCE_REVISION}}`
- Manifest-declared editable roots (read-only in this session): {{EDITABLE_ROOTS}}
- Bounded historical source corpus: `{{SOURCE_CORPUS}}`
- Required artifact: `{{ARTIFACT}}`

Read `agent_problem.json`, the incumbent adapter and implementation, the source manifest/lock, and the
bounded corpus. Treat the public semantic contract as authoritative. Do not infer hidden cases or search
outside the workspace.

## Macro objective

Do not assume that the inherited operator boundary, paged representation, kernel decomposition, or
fusion choice is fixed. Model the problem as jointly choosing

`min T_end_to_end(x; representation, preprocessing, decomposition, algorithm, schedule, dispatch)`

over the public workload domain. Separate immutable semantic/interface/policy/hardware constraints from
inherited implementation choices and from unverified assumptions. Include transformation and launch
costs in the objective: a route is useful only when its unlocked fast path repays those costs.

Construct an end-to-end architecture frontier with at least one genuinely distinct route in each family:

1. `direct_native`: consume the public input representation directly.
2. `representation_transform`: pay a conversion/gather/reorder cost to unlock a simpler fast path.
3. `multi_stage`: intentionally decompose the operation into separately optimized stages.
4. `hybrid_dispatch`: use evidence-based dispatch between mechanisms or regimes without hidden-shape
   specialization.

For every route state the data representation, pipeline, transformation cost, unlocked fast path,
winning and losing regimes, required mechanisms, risks, and concrete falsification tests. Rank every
route, retain at least one hedge route, and specify the cheapest experiments that can eliminate bad
directions before full implementation. This is a frontier and portfolio decision, not a commitment to
the most obvious fused kernel.

## Scientific probing boundary

Static analysis is expected. When it materially reduces uncertainty, you may also create small ignored
drivers or prototypes under `profiles/preplan/` and execute public synthetic probes through the gateway:

```bash
{{PUBLIC_DEV_COMMAND}}
```

Keep probing bounded: at most 24 files and 2 MiB under `profiles/preplan/`. Useful measurements include
representation-transform cost, contiguous versus indirect access, launch/decomposition overhead, and
rough crossover regimes. Record failed and deferred probes honestly. Development evidence is not
acceptance evidence.

{{SANDBOX}}

Hard prohibitions:

- Do not edit any tracked file or any manifest-declared editable root.
- Do not commit, branch, merge, rebase, reset, or alter refs.
- Do not invoke `repository_horizon.dev_eval`, the private evaluator, exact hidden shapes, private profile
  cases, or ABBA.
- Do not fetch/clone/network-search for source or use GPU/Kernel Wiki assets.
- Do not create a candidate, episode journal, handoff, or canonical memory record.

## Required JSON contract

Write only the final JSON artifact to `{{ARTIFACT}}` (plus optional ignored probing files). It must have:

- `schema_version`: `1`.
- `objective`: `metric="end_to_end_latency"`, all six `decision_variables` named above, and a non-empty
  mathematical `formulation`.
- `constraints`: non-empty `semantic`, `interface`, `policy`, and `hardware` object lists; each object has
  `statement` and workspace-local `evidence`.
- `inherited_implementation_choices`: objects with `choice`, `evidence`, `why_not_a_constraint`.
- `unverified_assumptions`: objects with unique `id`, `statement`, `consequence_if_false`, and
  `falsification_test`.
- `architecture_frontier`: route objects with `id`, `family`, `hypothesis`, `data_representation`,
  non-empty string lists for `pipeline`, `winning_regimes`, `losing_regimes`, `required_mechanisms`, and
  `risks`; `transformation_cost={status: measured|estimated|unknown, latency_us: number|null,
  evidence: string}`; `unlocked_fast_path`; and non-empty `falsification_tests`, whose objects contain
  `question`, `method`, `success_criterion`, and `failure_action`.
- `probing.experiments`: a possibly empty list. Each recorded experiment has `id`,
  `kind=static|gpu_probe|micro_prototype`, `hypothesis`, `status=measured|deferred|failed`, `evidence`, and
  `interpretation`.
- `portfolio`: `ranked_route_ids` containing every route exactly once, `primary_route_id`, one or more
  distinct `hedge_route_ids`, `selection_rationale`, and a non-empty object list `next_experiments`.

Use only public/workspace-local evidence in the artifact. Stop immediately after writing valid JSON.
