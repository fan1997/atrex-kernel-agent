# Repository Horizon V3 Preplan Enhance: end-to-end architecture frontier

This is a **plan-only macro-strategy session before episode 1**. Build an evidence-backed search
space for the end-to-end workload. You may inspect and measure, but you must not implement a candidate,
modify campaign source, commit, publish a handoff, invoke private evaluation, or begin an optimization
episode.

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

## General architecture-search laws

These laws are workload-independent. They enlarge the search space; they do not prescribe an answer.

1. **Contract over inheritance.** Separate immutable semantic, interface, policy, and hardware
   constraints from inherited implementation choices and unverified assumptions. Anything not fixed by
   the contract remains a possible decision variable.
2. **Optimize an implementation graph.** Search over the end-to-end graph from input representation,
   through optional preparation and compute stages, to the required output. Do not assume the inherited
   representation, operator boundary, stage division, algorithm, mapping, or scheduling policy is fixed.
3. **Remove structural obstacles.** Identify what prevents the contract from mapping efficiently to
   available hardware or software capabilities, then ask what legal graph change removes that obstacle
   and what new cost it introduces.
4. **Account for the whole path.** Include data movement, transformation, compute, synchronization,
   workspace, launch, selection, and postprocessing costs. A locally faster stage is not an end-to-end
   improvement unless the full graph wins.
5. **Make claims falsifiable.** Distinguish measured evidence, derived bounds, estimates, speculation,
   and unknowns. Use the cheapest decisive probe before investing in full implementation, and do not
   reject an architecture merely because an intentionally rough prototype is slow.
6. **Preserve mechanism diversity.** Avoid premature convergence on the incumbent or the first plausible
   route. Retain a materially different hedge unless public evidence rules it out.

The following are a non-exhaustive, optional, and composable thinking toolkit: changing representation
or layout; changing operator boundaries or lifecycle; preparation followed by reuse of an available
capability; direct implementation; decomposition and recombination; an equivalent algorithmic
formulation; a different parallel or data-movement mapping; numerical placement or recomputation; and
selection among already viable implementations using public runtime conditions. These are neither
required categories nor candidate answers. You may combine, reject, or go beyond them. Derive concrete
mechanisms only from this workload's public contract, source, and measurements.

Model the objective as:

`min T_end_to_end(x; G, theta)`

over the public workload domain, where `G` is an implementation graph and `theta` contains its internal
implementation parameters. If the public workload distribution is unknown, do not invent one: report
win/loss regions, worst regressions, resource costs, and uncertainty instead of a fabricated expectation.

## Required reasoning sequence

1. Normalize the public contract and identify implementation freedoms.
2. Build a workload-derived structural cost model and obstacle list.
3. Generate mechanism-distinct implementation graphs without mechanically covering a taxonomy.
4. State proof obligations and choose bounded probes that can change route ranking.
5. Produce a primary/hedge portfolio. Treat runtime dispatch only as an optional composition policy over
   concrete routes, never as a peer compute architecture or an empty placeholder.

## Bounded probing boundary

Static analysis is expected. When it materially reduces uncertainty, you may create small ignored
drivers or prototypes under `profiles/preplan/` and execute public synthetic probes through the gateway:

```bash
{{PUBLIC_DEV_COMMAND}}
```

Keep probing bounded: at most 24 files and 2 MiB under `profiles/preplan/`. Record the command, public
input description, environment, raw-output path, result, and interpretation. Failed and deferred probes
must be recorded honestly. Development evidence is not acceptance evidence.

{{SANDBOX}}

Hard prohibitions:

- Do not edit any tracked file or any manifest-declared editable root.
- Do not write outside this worktree.
- Do not commit, branch, merge, rebase, reset, or alter refs.
- Do not invoke `repository_horizon.dev_eval`, the private evaluator, exact hidden shapes, private profile
  cases, or ABBA.
- Do not fetch, clone, network-search for source, or use GPU/Kernel Wiki assets.
- Do not create a candidate, episode journal, handoff, or canonical memory record.

## Required JSON contract

Write only the final JSON artifact to `{{ARTIFACT}}` plus optional ignored probing files. It must contain:

- `schema_version=2`, `revision=1`, and `supersedes=null`.
- `objective`: `metric="end_to_end_latency"`, a non-empty mathematical `formulation`, and non-empty,
  workload-derived `decision_variables` objects with unique `name` and `description`.
- `contract_normal_form`: non-empty `semantic`, `interface`, `policy`, and `hardware` constraint object
  lists (`statement`, `evidence`), plus non-empty `implementation_freedoms` objects (`dimension`,
  `why_mutable`, `evidence`).
- `inherited_implementation_choices`: objects with `choice`, `evidence`, and `why_not_a_constraint`.
- `unverified_assumptions`: objects with unique `id`, `statement`, `consequence_if_false`, and
  `falsification_test`.
- `structural_cost_model`: non-empty `cost_terms` objects (`id`, `description`, `status`, `value`, `unit`,
  `evidence`) and non-empty `obstacles` objects (`id`, `statement`, `evidence`, `blocked_capability`,
  `removal_condition`). Evidence status is one of `measured`, `derived_bound`, `estimated`,
  `speculative`, or `unknown`; unknown values use `null`.
- `architecture_frontier`: at least two mechanism-distinct route objects. Each has `id`, `thesis`,
  `implementation_graph`, `mechanism_signature`, `addressed_obstacle_ids`, `changed_choices`,
  `prerequisites`, `cost_terms`, `evidence_level`, `supporting_evidence`, `contradicting_evidence`,
  `winning_regimes`, `losing_regimes`, `risks`, and `falsification_tests`. Optional `search_patterns` are
  descriptive only and may contain any values; no taxonomy coverage is required.
- `probing.experiments`: a possibly empty list. Each experiment records `id`, `kind`, `hypothesis`,
  `status`, `method`, `input_description`, `command`, `environment`, `evidence`, `interpretation`, and
  `decision_impact`. A measured `gpu_probe` or `micro_prototype` also records a workspace-relative
  `raw_output_path` under `profiles/preplan/` and its lowercase `sha256`; experiments without a raw
  output set both fields to `null`.
- `portfolio`: `ranked_route_ids` containing every route exactly once, a `primary_route_id`, one or more
  distinct `hedge_route_ids`, `selection_rationale`, `next_experiments`, `replan_triggers`, and optional
  `composition_policies`. A composition policy references at least two concrete route ids and states its
  public selection condition, added cost, and evidence level.

Every evidence reference must be public and workspace-local. Never present an estimate or speculation as
a measurement. Stop immediately after writing valid JSON.
