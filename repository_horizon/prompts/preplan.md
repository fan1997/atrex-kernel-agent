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
4. **Test representation bridges.** If capability `C` is known to work on representation `R2` while the
   contract supplies `R1`, explicitly evaluate a legal bridge before deciding to modify or replace `C`:
   `T_bridge = T(R1 -> R2) + T_C(R2) + T_post`. This is a generic counterfactual, not a prescribed
   implementation. Measure or bound the complete path, including temporary storage and launches.
5. **Search different graph cuts.** For each top structural obstacle, construct counterfactuals that
   remove it at materially different locations in the graph (before the blocked capability, inside it,
   after it, or at another workload-derived cut). Do not satisfy this by renaming the same mechanism.
   If only one cut is legal, justify that from the public contract or a derived bound.
6. **Keep routes atomic.** One route is exactly one connected implementation graph. Phrases such as
   “A or B”, optional mutually exclusive data paths, and unexpanded variants are separate routes, not
   one route. Runtime selection belongs only in a later composition policy.
7. **Account for the whole path.** Include data movement, transformation, compute, synchronization,
   workspace, launch, selection, and postprocessing costs. A locally faster stage is not an end-to-end
   improvement unless the full graph wins.
8. **Use evidence symmetrically.** Distinguish measured evidence, derived bounds, estimates,
   speculation,
   and unknowns. Use the cheapest decisive probe before investing in full implementation, and do not
   reject an architecture merely because an intentionally rough prototype is slow. A proxy workload may
   support a route, but cannot establish the primary route on the exact public contract. Give every
   plausible architecture-scale route its cheapest ranking-changing probe; defer one only with an
   explicit reason and keep the ranking provisional.
9. **Preserve mechanism diversity.** Avoid premature convergence on the incumbent or the first plausible
   route. Retain a materially different hedge unless public evidence rules it out.
10. **Replan after evidence.** Probe interpretation cannot lock the answer. After the bounded probes,
    reconsider every route together, record ranking before and after, and state what evidence changed or
    failed to change the ranking.

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
3. For each top obstacle, generate atomic, mechanism-distinct graphs at different graph cuts. Explicitly
   perform the representation-bridge counterfactual whenever an efficient capability and the supplied
   representation do not line up.
4. State proof obligations. Before favoring one route, allocate each plausible route its cheapest probe
   that could change the ranking; do not use evidence from one contract as if it measured another.
5. After all probes, perform a fresh adversarial ranking pass over every route.
6. Produce a portfolio that separately names the correctness bridge (when applicable), the provisional
   or evidence-complete performance primary, and hedge routes. Runtime dispatch is only an optional
   composition policy over concrete routes.

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

Use `repository_horizon/prompts/preplan_schema_v3.example.json` as the concrete shape reference. Before
stopping, run:

```bash
python -m repository_horizon.preplan validate {{ARTIFACT}}
```

Fix every reported violation without deleting substantive routes or evidence. The supervisor may start
one bounded schema-repair session if malformed JSON still escapes this check, but that session is not
allowed to change the architecture analysis.

Write only the final JSON artifact to `{{ARTIFACT}}` plus optional ignored probing files. It must contain:

- `schema_version=3`, `revision=1`, and `supersedes=null`.
- `objective`: `metric="end_to_end_latency"`, a non-empty mathematical `formulation`, and non-empty,
  workload-derived `decision_variables` objects with unique `name` and `description`.
- `contract_normal_form`: non-empty `semantic`, `interface`, `policy`, and `hardware` constraint object
  lists (`statement`, `evidence`), plus non-empty `implementation_freedoms` objects (`dimension`,
  `why_mutable`, `evidence`).
- `inherited_implementation_choices`: objects with `choice`, `evidence`, and `why_not_a_constraint`.
- `unverified_assumptions`: objects with unique `id`, `statement`, `consequence_if_false`, and
  `falsification_test`.
- `structural_cost_model`: centralized non-empty `cost_terms` objects (`id`, `description`, `status`,
  `value`, optional `formula`, `unit`, `evidence`), non-empty `obstacles`, and non-empty
  `top_obstacle_ids`. An obstacle has `id`, `statement`, `evidence`, `blocked_capability`,
  `removal_condition`, and optional `single_cut_justification`. Evidence status is one of `measured`,
  `derived_bound`, `estimated`, `speculative`, or `unknown`; unknown values use `null`, while a derived
  bound may use a numeric value or a non-empty formula.
- `representation_bridge_analysis`: `applicability` is `applicable` or `not_applicable`. When applicable,
  provide assessments with `id`, `obstacle_id`, source and target representations, enabled capability,
  legality, the complete-path cost equation, referenced centralized cost ids, evidence level/evidence,
  disposition, and decision basis. A rejected bridge requires measured evidence or a derived bound; a
  frontier bridge must be referenced by a route. When not applicable, provide a reason and no assessments.
- `architecture_frontier`: at least two mechanism-distinct route objects. Each has `id`, `thesis`,
  one connected `implementation_graph`, a structured `mechanism_signature`, `addressed_obstacle_ids`,
  `bridge_assessment_ids`, `changed_choices`, `prerequisites`, centralized `cost_term_ids`,
  `evidence_level`, `evidence_scope`, supporting/contradicting evidence, winning/losing regimes, risks,
  actionable string `falsification_tests`, and a `ranking_probe`. Top obstacles need routes covering at
  least two distinct `changed_graph_cuts`, unless the obstacle carries a public single-cut justification.
  Optional `search_patterns` are descriptive only; no named taxonomy coverage is required.
- `probing.experiments`: a possibly empty list. Each experiment records `id`, `kind`, `hypothesis`,
  `status`, `method`, `input_description`, `command`, `environment`, `evidence_level`, `evidence`,
  `interpretation`, and `decision_impact`. A completed/measured `gpu_probe` or `micro_prototype` records a workspace-relative
  `raw_output_path` under `profiles/preplan/` and its lowercase `sha256`; experiments without a raw
  output set both fields to `null`. `post_probe_replan` records rankings before/after, evidence that
  changed them, reconsideration of every route, and unresolved decisive experiment ids.
- `portfolio`: `ranked_route_ids` containing every route exactly once, a
  `performance_primary_route_id`, `correctness_bridge_route_id` (or null when bridges do not apply), one
  or more distinct `hedge_route_ids`, `ranking_status`, `selection_rationale`, structured
  `next_experiments`, `replan_triggers`, and optional `composition_policies`. A route lacking evidence on
  the exact public contract, or any deferred ranking probe, forces `ranking_status="provisional"`.

Every evidence reference must be public and workspace-local. Never present an estimate or speculation as
a measurement. Stop immediately after writing valid JSON.
