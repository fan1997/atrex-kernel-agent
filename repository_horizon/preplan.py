from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from long_horizon.git_episode import git_head, working_changes
from long_horizon.protocol import atomic_write_json
from orchestrator.session_io import run_session

from .corpus import CORPUS_RELATIVE, read_catalog, validate_source_corpus
from .manifest import RepositoryManifest
from .runtime import link_repository_runtime

PREPLAN_ARTIFACT = PurePosixPath("plans/end_to_end_architecture_frontier.json")
PREPLAN_PROFILE_ROOT = PurePosixPath("profiles/preplan")
PREPLAN_PROMPT = Path(__file__).resolve().parent / "prompts" / "preplan.md"
EVIDENCE_LEVELS = frozenset(
    {"measured", "derived_bound", "estimated", "speculative", "unknown"}
)
FORBIDDEN_EVIDENCE_MARKERS = (
    "/private_evaluator/",
    "/.atrex_private_profile_case.json",
)
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_PROFILE_FILES = 24
MAX_PROFILE_BYTES = 2 * 1024 * 1024
PREPLAN_SCHEMA_VERSION = 3


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _nonempty_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(_nonempty(item) for item in value)
    )


def _string_list(value: object) -> bool:
    return isinstance(value, list) and all(_nonempty(item) for item in value)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _object_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _require_fields(
    value: dict[str, Any], fields: tuple[str, ...], label: str
) -> list[str]:
    return [f"{label} requires {field}" for field in fields if field not in value]


def _validate_object_list(
    value: object,
    *,
    label: str,
    fields: tuple[str, ...],
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an object list" if allow_empty else "a non-empty object list"
        return [f"{label} must be {qualifier}"]
    violations: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            violations.append(f"{label}[{index}] must be an object")
            continue
        for field in fields:
            if not _nonempty(item.get(field)):
                violations.append(f"{label}[{index}] requires {field}")
    return violations


def _validate_cost_terms(value: object, *, label: str) -> tuple[list[str], set[str]]:
    violations = _validate_object_list(
        value,
        label=label,
        fields=("id", "description", "status", "unit", "evidence"),
    )
    identifiers: set[str] = set()
    if not isinstance(value, list):
        return violations, identifiers
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if isinstance(identifier, str):
            if identifier in identifiers:
                violations.append(f"duplicate {label} id: {identifier}")
            identifiers.add(identifier)
        status = item.get("status")
        if status not in EVIDENCE_LEVELS:
            violations.append(f"{label}[{index}].status is unsupported: {status}")
        numeric = isinstance(item.get("value"), (int, float)) and not isinstance(
            item.get("value"), bool
        )
        formula = item.get("formula")
        if formula is not None and not _nonempty(formula):
            violations.append(f"{label}[{index}].formula must be null or non-empty")
        if status == "measured" and not numeric:
            violations.append(
                f"{label}[{index}].value must be numeric for measured evidence"
            )
        if status == "derived_bound" and not numeric and not _nonempty(formula):
            violations.append(
                f"{label}[{index}] requires numeric value or formula for derived_bound evidence"
            )
        if status == "unknown" and item.get("value") is not None:
            violations.append(f"{label}[{index}].value must be null when unknown")
        if item.get("value") is not None and not numeric:
            violations.append(f"{label}[{index}].value must be numeric or null")
    return violations, identifiers


def _validate_implementation_graph(value: object, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    violations = _require_fields(
        value,
        ("input_representation", "stages", "output_representation"),
        label,
    )
    for field in ("input_representation", "output_representation"):
        if not _nonempty(value.get(field)):
            violations.append(f"{label}.{field} must be non-empty")
    stages = value.get("stages")
    violations.extend(
        _validate_object_list(
            stages,
            label=f"{label}.stages",
            fields=(
                "id",
                "operation",
                "input_representation",
                "output_representation",
            ),
        )
    )
    if not isinstance(stages, list) or not stages:
        return violations
    if not all(isinstance(stage, dict) for stage in stages):
        return violations
    stage_ids = [stage.get("id") for stage in stages]
    if all(isinstance(stage_id, str) for stage_id in stage_ids) and len(
        stage_ids
    ) != len(set(stage_ids)):
        violations.append(f"{label}.stages ids must be unique")
    expected = value.get("input_representation")
    for index, stage in enumerate(stages):
        if stage.get("input_representation") != expected:
            violations.append(
                f"{label}.stages[{index}] does not connect to the preceding representation"
            )
        expected = stage.get("output_representation")
    if expected != value.get("output_representation"):
        violations.append(
            f"{label}.output_representation does not match the last stage"
        )
    return violations


def _validate_mechanism_signature(
    value: object, *, label: str
) -> tuple[list[str], tuple[str, ...] | None, set[str]]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"], None, set()
    violations: list[str] = []
    fields = (
        "changed_graph_cuts",
        "representation_path",
        "compute_mechanism",
        "operator_boundary",
    )
    violations.extend(_require_fields(value, fields, label))
    for field in ("changed_graph_cuts", "representation_path"):
        if not _nonempty_list(value.get(field)):
            violations.append(f"{label}.{field} must be a non-empty string list")
    for field in ("compute_mechanism", "operator_boundary"):
        if not _nonempty(value.get(field)):
            violations.append(f"{label}.{field} must be non-empty")
    if violations:
        return violations, None, set()
    cuts = {item.casefold() for item in value["changed_graph_cuts"]}
    normalized = (
        *(f"cut:{item}" for item in sorted(cuts)),
        *(f"repr:{item.casefold()}" for item in value["representation_path"]),
        f"compute:{value['compute_mechanism'].casefold()}",
        f"boundary:{value['operator_boundary'].casefold()}",
    )
    return violations, normalized, cuts


def validate_preplan_artifact(path: Path) -> list[str]:
    """Validate the macro-strategy artifact without judging which route should win."""

    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            return [f"preplan artifact exceeds {MAX_ARTIFACT_BYTES} bytes"]
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read preplan artifact: {exc}"]
    if not isinstance(document, dict):
        return ["preplan artifact must be a JSON object"]
    violations = _require_fields(
        document,
        (
            "schema_version",
            "revision",
            "supersedes",
            "objective",
            "contract_normal_form",
            "inherited_implementation_choices",
            "unverified_assumptions",
            "structural_cost_model",
            "representation_bridge_analysis",
            "architecture_frontier",
            "probing",
            "portfolio",
        ),
        "preplan artifact",
    )
    if violations:
        return violations
    if document["schema_version"] != PREPLAN_SCHEMA_VERSION:
        violations.append(
            f"preplan artifact schema_version must be {PREPLAN_SCHEMA_VERSION}"
        )
    revision = document["revision"]
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        violations.append("preplan artifact revision must be a positive integer")
    supersedes = document["supersedes"]
    if revision == 1 and supersedes is not None:
        violations.append("initial preplan revision must set supersedes to null")
    if isinstance(revision, int) and revision > 1:
        if not isinstance(supersedes, dict):
            violations.append("revised preplan requires a supersedes object")
        else:
            if not isinstance(supersedes.get("revision"), int):
                violations.append("supersedes requires integer revision")
            for field in ("artifact_sha256", "reason"):
                if not _nonempty(supersedes.get(field)):
                    violations.append(f"supersedes requires {field}")

    objective = document["objective"]
    if not isinstance(objective, dict):
        violations.append("objective must be an object")
    else:
        if objective.get("metric") != "end_to_end_latency":
            violations.append("objective.metric must be end_to_end_latency")
        raw_variables = objective.get("decision_variables")
        violations.extend(
            _validate_object_list(
                raw_variables,
                label="objective.decision_variables",
                fields=("name", "description"),
            )
        )
        if isinstance(raw_variables, list):
            variable_names = [
                item.get("name") for item in raw_variables if isinstance(item, dict)
            ]
            if all(isinstance(name, str) for name in variable_names) and len(
                variable_names
            ) != len(set(variable_names)):
                violations.append("objective.decision_variables names must be unique")
        if not _nonempty(objective.get("formulation")):
            violations.append("objective requires a non-empty mathematical formulation")

    contract = document["contract_normal_form"]
    if not isinstance(contract, dict):
        violations.append("contract_normal_form must be an object")
    else:
        for category in ("semantic", "interface", "policy", "hardware"):
            violations.extend(
                _validate_object_list(
                    contract.get(category),
                    label=f"contract_normal_form.{category}",
                    fields=("statement", "evidence"),
                )
            )
        violations.extend(
            _validate_object_list(
                contract.get("implementation_freedoms"),
                label="contract_normal_form.implementation_freedoms",
                fields=("dimension", "why_mutable", "evidence"),
            )
        )

    choices = document["inherited_implementation_choices"]
    if not _object_list(choices):
        violations.append(
            "inherited_implementation_choices must be a non-empty object list"
        )
    else:
        for index, choice in enumerate(choices):
            for field in ("choice", "evidence", "why_not_a_constraint"):
                if not _nonempty(choice.get(field)):
                    violations.append(
                        f"inherited_implementation_choices[{index}] requires {field}"
                    )

    assumptions = document["unverified_assumptions"]
    if not _object_list(assumptions):
        violations.append("unverified_assumptions must be a non-empty object list")
    else:
        seen_assumptions: set[str] = set()
        for index, assumption in enumerate(assumptions):
            for field in (
                "id",
                "statement",
                "consequence_if_false",
                "falsification_test",
            ):
                if not _nonempty(assumption.get(field)):
                    violations.append(
                        f"unverified_assumptions[{index}] requires {field}"
                    )
            identifier = assumption.get("id")
            if isinstance(identifier, str):
                if identifier in seen_assumptions:
                    violations.append(f"duplicate assumption id: {identifier}")
                seen_assumptions.add(identifier)

    structural_model = document["structural_cost_model"]
    obstacle_ids: set[str] = set()
    top_obstacle_ids: set[str] = set()
    cost_term_ids: set[str] = set()
    obstacle_exemptions: dict[str, str] = {}
    if not isinstance(structural_model, dict):
        violations.append("structural_cost_model must be an object")
    else:
        cost_violations, cost_term_ids = _validate_cost_terms(
            structural_model.get("cost_terms"),
            label="structural_cost_model.cost_terms",
        )
        violations.extend(cost_violations)
        obstacles = structural_model.get("obstacles")
        violations.extend(
            _validate_object_list(
                obstacles,
                label="structural_cost_model.obstacles",
                fields=(
                    "id",
                    "statement",
                    "evidence",
                    "blocked_capability",
                    "removal_condition",
                ),
            )
        )
        if isinstance(obstacles, list):
            for obstacle in obstacles:
                if not isinstance(obstacle, dict):
                    continue
                identifier = obstacle.get("id")
                if isinstance(identifier, str):
                    if identifier in obstacle_ids:
                        violations.append(f"duplicate obstacle id: {identifier}")
                    obstacle_ids.add(identifier)
                    exemption = obstacle.get("single_cut_justification")
                    if _nonempty(exemption):
                        obstacle_exemptions[identifier] = exemption
                    elif exemption is not None:
                        violations.append(
                            "structural_cost_model.obstacles single_cut_justification "
                            "must be null or non-empty"
                        )
        raw_top_obstacles = structural_model.get("top_obstacle_ids")
        if not _nonempty_list(raw_top_obstacles):
            violations.append(
                "structural_cost_model.top_obstacle_ids must be a non-empty string list"
            )
        else:
            top_obstacle_ids = set(raw_top_obstacles)
            unknown = top_obstacle_ids - obstacle_ids
            if unknown:
                violations.append(
                    "structural_cost_model.top_obstacle_ids contains unknown ids: "
                    + ", ".join(sorted(unknown))
                )

    bridge_analysis = document["representation_bridge_analysis"]
    bridge_ids: set[str] = set()
    frontier_bridge_ids: set[str] = set()
    bridge_probe_ids: set[str] = set()
    if not isinstance(bridge_analysis, dict):
        violations.append("representation_bridge_analysis must be an object")
    else:
        applicability = bridge_analysis.get("applicability")
        if applicability not in {"applicable", "not_applicable"}:
            violations.append(
                "representation_bridge_analysis.applicability must be applicable or not_applicable"
            )
        assessments = bridge_analysis.get("assessments")
        if not isinstance(assessments, list):
            violations.append(
                "representation_bridge_analysis.assessments must be an object list"
            )
            assessments = []
        if applicability == "applicable" and not assessments:
            violations.append(
                "applicable representation bridge analysis requires at least one assessment"
            )
        if applicability == "not_applicable":
            if assessments:
                violations.append(
                    "not_applicable representation bridge analysis must have no assessments"
                )
            if not _nonempty(bridge_analysis.get("non_applicability_reason")):
                violations.append(
                    "not_applicable representation bridge analysis requires a reason"
                )
        for index, assessment in enumerate(assessments):
            label = f"representation_bridge_analysis.assessments[{index}]"
            if not isinstance(assessment, dict):
                violations.append(f"{label} must be an object")
                continue
            for field in (
                "id",
                "obstacle_id",
                "source_representation",
                "target_representation",
                "enabled_capability",
                "legality",
                "full_path_cost_equation",
                "cost_term_ids",
                "evidence_level",
                "evidence",
                "disposition",
                "decision_basis",
            ):
                if field not in assessment:
                    violations.append(f"{label} requires {field}")
            identifier = assessment.get("id")
            if not _nonempty(identifier):
                violations.append(f"{label}.id must be non-empty")
            elif identifier in bridge_ids:
                violations.append(f"duplicate representation bridge id: {identifier}")
            else:
                bridge_ids.add(identifier)
            for field in (
                "source_representation",
                "target_representation",
                "enabled_capability",
                "legality",
                "full_path_cost_equation",
                "evidence",
                "decision_basis",
            ):
                if not _nonempty(assessment.get(field)):
                    violations.append(f"{label}.{field} must be non-empty")
            obstacle_id = assessment.get("obstacle_id")
            if obstacle_id not in obstacle_ids:
                violations.append(f"{label}.obstacle_id must reference an obstacle")
            referenced_costs = assessment.get("cost_term_ids")
            if not _nonempty_list(referenced_costs):
                violations.append(
                    f"{label}.cost_term_ids must be a non-empty string list"
                )
            else:
                unknown_costs = set(referenced_costs) - cost_term_ids
                if unknown_costs:
                    violations.append(
                        f"{label}.cost_term_ids contains unknown ids: "
                        + ", ".join(sorted(unknown_costs))
                    )
            evidence_level = assessment.get("evidence_level")
            if evidence_level not in EVIDENCE_LEVELS:
                violations.append(f"{label}.evidence_level is unsupported")
            disposition = assessment.get("disposition")
            if disposition not in {"frontier", "rejected", "deferred"}:
                violations.append(f"{label}.disposition is unsupported")
            elif disposition == "frontier" and isinstance(identifier, str):
                frontier_bridge_ids.add(identifier)
            elif disposition == "rejected" and evidence_level not in {
                "measured",
                "derived_bound",
            }:
                violations.append(
                    f"{label} may be rejected only by measured evidence or a derived bound"
                )
            probe_id = assessment.get("probe_id")
            if probe_id is not None:
                if not _nonempty(probe_id):
                    violations.append(f"{label}.probe_id must be null or non-empty")
                else:
                    bridge_probe_ids.add(probe_id)

    routes = document["architecture_frontier"]
    route_ids: set[str] = set()
    mechanism_signatures: set[tuple[str, ...]] = set()
    obstacle_graph_cuts: dict[str, set[str]] = {}
    referenced_bridge_ids: set[str] = set()
    route_evidence_scopes: dict[str, bool] = {}
    route_probe_records: dict[str, dict[str, Any]] = {}
    if (
        not isinstance(routes, list)
        or len(routes) < 2
        or not all(isinstance(route, dict) for route in routes)
    ):
        violations.append(
            "architecture_frontier must contain at least two route objects"
        )
    else:
        for index, route in enumerate(routes):
            label = f"architecture_frontier[{index}]"
            for field in (
                "id",
                "thesis",
                "implementation_graph",
                "mechanism_signature",
                "addressed_obstacle_ids",
                "bridge_assessment_ids",
                "changed_choices",
                "prerequisites",
                "cost_term_ids",
                "evidence_level",
                "evidence_scope",
                "supporting_evidence",
                "contradicting_evidence",
                "winning_regimes",
                "losing_regimes",
                "risks",
                "falsification_tests",
                "ranking_probe",
            ):
                if field not in route:
                    violations.append(f"{label} requires {field}")
            identifier = route.get("id")
            if not _nonempty(identifier):
                violations.append(f"{label}.id must be non-empty")
            elif identifier in route_ids:
                violations.append(f"duplicate architecture route id: {identifier}")
            else:
                route_ids.add(identifier)
            if not _nonempty(route.get("thesis")):
                violations.append(f"{label}.thesis must be non-empty")
            violations.extend(
                _validate_implementation_graph(
                    route.get("implementation_graph"),
                    label=f"{label}.implementation_graph",
                )
            )
            signature_violations, normalized, graph_cuts = (
                _validate_mechanism_signature(
                    route.get("mechanism_signature"),
                    label=f"{label}.mechanism_signature",
                )
            )
            violations.extend(signature_violations)
            if normalized is not None:
                if normalized in mechanism_signatures:
                    violations.append(
                        f"{label}.mechanism_signature duplicates another route"
                    )
                mechanism_signatures.add(normalized)
            for field in ("addressed_obstacle_ids", "winning_regimes", "risks"):
                if not _nonempty_list(route.get(field)):
                    violations.append(
                        f"{label}.{field} must be a non-empty string list"
                    )
            for field in (
                "bridge_assessment_ids",
                "changed_choices",
                "prerequisites",
                "supporting_evidence",
                "contradicting_evidence",
                "losing_regimes",
            ):
                if not _string_list(route.get(field)):
                    violations.append(f"{label}.{field} must be a string list")
            addressed = route.get("addressed_obstacle_ids")
            if isinstance(addressed, list):
                unknown_obstacles = {
                    item
                    for item in addressed
                    if isinstance(item, str) and item not in obstacle_ids
                }
                if unknown_obstacles:
                    violations.append(
                        f"{label}.addressed_obstacle_ids contains unknown ids: "
                        + ", ".join(sorted(str(item) for item in unknown_obstacles))
                    )
                for obstacle_id in addressed:
                    if isinstance(obstacle_id, str):
                        obstacle_graph_cuts.setdefault(obstacle_id, set()).update(
                            graph_cuts
                        )
            referenced_bridges = route.get("bridge_assessment_ids")
            if isinstance(referenced_bridges, list):
                unknown_bridges = {
                    item
                    for item in referenced_bridges
                    if isinstance(item, str) and item not in bridge_ids
                }
                if unknown_bridges:
                    violations.append(
                        f"{label}.bridge_assessment_ids contains unknown ids: "
                        + ", ".join(sorted(unknown_bridges))
                    )
                referenced_bridge_ids.update(
                    item for item in referenced_bridges if isinstance(item, str)
                )
            referenced_costs = route.get("cost_term_ids")
            if not _nonempty_list(referenced_costs):
                violations.append(
                    f"{label}.cost_term_ids must be a non-empty string list"
                )
            else:
                unknown_costs = set(referenced_costs) - cost_term_ids
                if unknown_costs:
                    violations.append(
                        f"{label}.cost_term_ids contains unknown ids: "
                        + ", ".join(sorted(unknown_costs))
                    )
            if route.get("evidence_level") not in EVIDENCE_LEVELS:
                violations.append(f"{label}.evidence_level is unsupported")
            scope = route.get("evidence_scope")
            if not isinstance(scope, dict):
                violations.append(f"{label}.evidence_scope must be an object")
            else:
                if not _bool(scope.get("exact_public_contract")):
                    if scope.get("exact_public_contract") is not False:
                        violations.append(
                            f"{label}.evidence_scope.exact_public_contract must be boolean"
                        )
                if not _string_list(scope.get("proxy_contracts")):
                    violations.append(
                        f"{label}.evidence_scope.proxy_contracts must be a string list"
                    )
                if not _nonempty(scope.get("explanation")):
                    violations.append(f"{label}.evidence_scope requires explanation")
                if isinstance(identifier, str):
                    route_evidence_scopes[identifier] = bool(
                        scope.get("exact_public_contract")
                    )
            patterns = route.get("search_patterns")
            if patterns is not None and not _string_list(patterns):
                violations.append(f"{label}.search_patterns must be a string list")
            tests = route.get("falsification_tests")
            if not _nonempty_list(tests):
                violations.append(
                    f"{label}.falsification_tests must be a non-empty string list"
                )
            ranking_probe = route.get("ranking_probe")
            if not isinstance(ranking_probe, dict):
                violations.append(f"{label}.ranking_probe must be an object")
            else:
                for field in ("question", "cheapest_method", "status"):
                    if not _nonempty(ranking_probe.get(field)):
                        violations.append(f"{label}.ranking_probe requires {field}")
                status = ranking_probe.get("status")
                if status not in {"completed", "failed", "deferred", "not_needed"}:
                    violations.append(f"{label}.ranking_probe.status is unsupported")
                if status in {"completed", "failed"} and not _nonempty(
                    ranking_probe.get("experiment_id")
                ):
                    violations.append(
                        f"{label}.ranking_probe {status} requires experiment_id"
                    )
                if status == "deferred" and not _nonempty(
                    ranking_probe.get("non_execution_reason")
                ):
                    violations.append(
                        f"{label}.ranking_probe deferred requires non_execution_reason"
                    )
                if status == "not_needed" and not _nonempty(
                    ranking_probe.get("bound_or_contract_evidence")
                ):
                    violations.append(
                        f"{label}.ranking_probe not_needed requires bound_or_contract_evidence"
                    )
                if isinstance(identifier, str):
                    route_probe_records[identifier] = ranking_probe

        missing_frontier_bridges = frontier_bridge_ids - referenced_bridge_ids
        if missing_frontier_bridges:
            violations.append(
                "frontier representation bridges must be referenced by an architecture route: "
                + ", ".join(sorted(missing_frontier_bridges))
            )
        for obstacle_id in sorted(top_obstacle_ids):
            cuts = obstacle_graph_cuts.get(obstacle_id, set())
            if len(cuts) < 2 and obstacle_id not in obstacle_exemptions:
                violations.append(
                    f"top obstacle {obstacle_id} requires routes at two distinct graph cuts "
                    "or a single_cut_justification"
                )

    probing = document["probing"]
    experiment_ids: set[str] = set()
    post_probe_replan: dict[str, Any] = {}
    if not isinstance(probing, dict) or not isinstance(
        probing.get("experiments"), list
    ):
        violations.append("probing.experiments must be a list")
    else:
        for index, experiment in enumerate(probing["experiments"]):
            if not isinstance(experiment, dict):
                violations.append(f"probing.experiments[{index}] must be an object")
                continue
            for field in (
                "id",
                "kind",
                "hypothesis",
                "status",
                "method",
                "input_description",
                "command",
                "environment",
                "evidence_level",
                "evidence",
                "interpretation",
                "decision_impact",
            ):
                if not _nonempty(experiment.get(field)):
                    violations.append(f"probing.experiments[{index}] requires {field}")
            experiment_id = experiment.get("id")
            if isinstance(experiment_id, str):
                if experiment_id in experiment_ids:
                    violations.append(
                        f"duplicate probing experiment id: {experiment_id}"
                    )
                experiment_ids.add(experiment_id)
            if experiment.get("kind") not in {
                "static",
                "gpu_probe",
                "micro_prototype",
            }:
                violations.append(f"probing.experiments[{index}].kind is unsupported")
            if experiment.get("status") not in {
                "measured",
                "completed",
                "deferred",
                "failed",
            }:
                violations.append(f"probing.experiments[{index}].status is unsupported")
            if experiment.get("evidence_level") not in EVIDENCE_LEVELS:
                violations.append(
                    f"probing.experiments[{index}].evidence_level is unsupported"
                )
            raw_output = experiment.get("raw_output_path")
            raw_digest = experiment.get("sha256")
            requires_raw_output = experiment.get("status") in {
                "measured",
                "completed",
            } and experiment.get("kind") in {
                "gpu_probe",
                "micro_prototype",
            }
            if requires_raw_output and (
                not _nonempty(raw_output) or not _nonempty(raw_digest)
            ):
                violations.append(
                    f"probing.experiments[{index}] completed runtime evidence "
                    "requires raw_output_path and sha256"
                )
            if raw_output is not None or raw_digest is not None:
                if not _nonempty(raw_output) or not _nonempty(raw_digest):
                    violations.append(
                        f"probing.experiments[{index}] raw_output_path and sha256 "
                        "must be set together"
                    )
                    continue
                assert isinstance(raw_output, str)
                assert isinstance(raw_digest, str)
                relative = PurePosixPath(raw_output)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.parts[:2] != PREPLAN_PROFILE_ROOT.parts
                ):
                    violations.append(
                        f"probing.experiments[{index}].raw_output_path must stay under "
                        f"{PREPLAN_PROFILE_ROOT.as_posix()}/"
                    )
                    continue
                evidence_path = path.parent.parent.joinpath(*relative.parts)
                if not evidence_path.is_file() or evidence_path.is_symlink():
                    violations.append(
                        f"probing.experiments[{index}].raw_output_path is not a regular file"
                    )
                    continue
                actual_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
                if raw_digest != actual_digest:
                    violations.append(
                        f"probing.experiments[{index}].sha256 does not match raw output"
                    )

        for route_id, ranking_probe in route_probe_records.items():
            status = ranking_probe.get("status")
            experiment_id = ranking_probe.get("experiment_id")
            if (
                status in {"completed", "failed"}
                and experiment_id not in experiment_ids
            ):
                violations.append(
                    f"route {route_id} ranking_probe references unknown experiment_id"
                )
        unknown_bridge_probes = bridge_probe_ids - experiment_ids
        if unknown_bridge_probes:
            violations.append(
                "representation bridge assessments reference unknown probe ids: "
                + ", ".join(sorted(unknown_bridge_probes))
            )

        replan = probing.get("post_probe_replan")
        if not isinstance(replan, dict):
            violations.append("probing.post_probe_replan must be an object")
        else:
            post_probe_replan = replan
            for field in (
                "ranking_before",
                "ranking_after",
                "evidence_that_changed_ranking",
                "route_reconsiderations",
                "unresolved_decisive_probe_ids",
            ):
                if field not in replan:
                    violations.append(f"probing.post_probe_replan requires {field}")
            for field in ("ranking_before", "ranking_after"):
                ranking = replan.get(field)
                if (
                    not isinstance(ranking, list)
                    or len(ranking) != len(route_ids)
                    or any(not isinstance(item, str) for item in ranking)
                    or set(ranking) != route_ids
                ):
                    violations.append(
                        f"probing.post_probe_replan.{field} must rank every route exactly once"
                    )
            if not _string_list(replan.get("evidence_that_changed_ranking")):
                violations.append(
                    "probing.post_probe_replan.evidence_that_changed_ranking must be a string list"
                )
            reconsiderations = replan.get("route_reconsiderations")
            violations.extend(
                _validate_object_list(
                    reconsiderations,
                    label="probing.post_probe_replan.route_reconsiderations",
                    fields=("route_id", "evidence_reviewed", "rank_effect"),
                )
            )
            if isinstance(reconsiderations, list):
                reconsidered_ids = {
                    item.get("route_id")
                    for item in reconsiderations
                    if isinstance(item, dict) and isinstance(item.get("route_id"), str)
                }
                if reconsidered_ids != route_ids:
                    violations.append(
                        "probing.post_probe_replan.route_reconsiderations must cover every route"
                    )
            unresolved = replan.get("unresolved_decisive_probe_ids")
            if not _string_list(unresolved):
                violations.append(
                    "probing.post_probe_replan.unresolved_decisive_probe_ids must be a string list"
                )
            elif not set(unresolved).issubset(experiment_ids):
                violations.append(
                    "probing.post_probe_replan.unresolved_decisive_probe_ids contains unknown ids"
                )

    portfolio = document["portfolio"]
    if not isinstance(portfolio, dict):
        violations.append("portfolio must be an object")
    else:
        ranked = portfolio.get("ranked_route_ids")
        if (
            not isinstance(ranked, list)
            or any(not isinstance(item, str) for item in ranked)
            or set(ranked) != route_ids
            or len(ranked) != len(route_ids)
        ):
            violations.append(
                "portfolio.ranked_route_ids must rank every route exactly once"
            )
        primary = portfolio.get("performance_primary_route_id")
        if not isinstance(primary, str) or primary not in route_ids:
            violations.append(
                "portfolio.performance_primary_route_id must reference a frontier route"
            )
        correctness_bridge = portfolio.get("correctness_bridge_route_id")
        if frontier_bridge_ids:
            if (
                not isinstance(correctness_bridge, str)
                or correctness_bridge not in route_ids
            ):
                violations.append(
                    "portfolio.correctness_bridge_route_id must reference a frontier route "
                    "when representation bridges are applicable"
                )
            elif not {
                item
                for item in next(
                    (
                        route.get("bridge_assessment_ids", [])
                        for route in routes
                        if isinstance(route, dict)
                        and route.get("id") == correctness_bridge
                    ),
                    [],
                )
                if isinstance(item, str)
            }.intersection(frontier_bridge_ids):
                violations.append(
                    "portfolio.correctness_bridge_route_id must reference a frontier bridge assessment"
                )
        elif correctness_bridge is not None:
            violations.append(
                "portfolio.correctness_bridge_route_id must be null when no bridge remains on the frontier"
            )
        hedges = portfolio.get("hedge_route_ids")
        if (
            not isinstance(hedges, list)
            or not hedges
            or any(not isinstance(item, str) for item in hedges)
            or any(item not in route_ids for item in hedges)
        ):
            violations.append(
                "portfolio.hedge_route_ids must reference at least one frontier route"
            )
        if isinstance(hedges, list) and primary in hedges:
            violations.append("portfolio primary route cannot also be a hedge route")
        ranking_status = portfolio.get("ranking_status")
        if ranking_status not in {"provisional", "evidence_complete"}:
            violations.append(
                "portfolio.ranking_status must be provisional or evidence_complete"
            )
        if (
            isinstance(primary, str)
            and not route_evidence_scopes.get(primary, False)
            and ranking_status != "provisional"
        ):
            violations.append(
                "a primary route without exact-public-contract evidence must remain provisional"
            )
        deferred_route_ids = {
            route_id
            for route_id, record in route_probe_records.items()
            if record.get("status") == "deferred"
        }
        if deferred_route_ids and ranking_status != "provisional":
            violations.append(
                "deferred route-ranking probes require a provisional portfolio"
            )
        if not _nonempty(portfolio.get("selection_rationale")):
            violations.append("portfolio requires selection_rationale")
        next_experiments = portfolio.get("next_experiments")
        violations.extend(
            _validate_object_list(
                next_experiments,
                label="portfolio.next_experiments",
                fields=("id", "purpose"),
            )
        )
        deferred_covered: set[str] = set()
        if isinstance(next_experiments, list):
            for index, experiment in enumerate(next_experiments):
                if not isinstance(experiment, dict):
                    continue
                covered_routes = experiment.get("route_ids")
                if not _nonempty_list(covered_routes):
                    violations.append(
                        f"portfolio.next_experiments[{index}].route_ids must be a non-empty string list"
                    )
                    continue
                unknown_routes = set(covered_routes) - route_ids
                if unknown_routes:
                    violations.append(
                        f"portfolio.next_experiments[{index}].route_ids contains unknown ids"
                    )
                deferred_covered.update(covered_routes)
        if not deferred_route_ids.issubset(deferred_covered):
            violations.append(
                "every deferred route-ranking probe must be covered by portfolio.next_experiments"
            )
        if not _nonempty_list(portfolio.get("replan_triggers")):
            violations.append(
                "portfolio.replan_triggers must be a non-empty string list"
            )
        policies = portfolio.get("composition_policies", [])
        if not isinstance(policies, list):
            violations.append("portfolio.composition_policies must be an object list")
        else:
            for index, policy in enumerate(policies):
                label = f"portfolio.composition_policies[{index}]"
                if not isinstance(policy, dict):
                    violations.append(f"{label} must be an object")
                    continue
                for field in ("id", "public_condition", "added_cost"):
                    if not _nonempty(policy.get(field)):
                        violations.append(f"{label} requires {field}")
                policy_routes = policy.get("route_ids")
                if (
                    not isinstance(policy_routes, list)
                    or len(policy_routes) < 2
                    or any(not isinstance(item, str) for item in policy_routes)
                    or len(set(policy_routes)) != len(policy_routes)
                    or any(item not in route_ids for item in policy_routes)
                ):
                    violations.append(
                        f"{label}.route_ids must reference at least two distinct routes"
                    )
                if policy.get("evidence_level") not in EVIDENCE_LEVELS:
                    violations.append(f"{label}.evidence_level is unsupported")
        if post_probe_replan:
            if post_probe_replan.get("ranking_after") != ranked:
                violations.append(
                    "probing.post_probe_replan.ranking_after must equal portfolio.ranked_route_ids"
                )

    serialized = json.dumps(document, sort_keys=True).casefold()
    for marker in FORBIDDEN_EVIDENCE_MARKERS:
        if marker in serialized:
            violations.append(
                "preplan evidence contains forbidden private-evaluator marker: "
                + marker
            )
    return violations


def _profile_violations(workspace: Path) -> list[str]:
    root = workspace / PREPLAN_PROFILE_ROOT
    if not root.exists():
        return []
    files = [path for path in root.rglob("*") if path.is_file() or path.is_symlink()]
    violations: list[str] = []
    if len(files) > MAX_PROFILE_FILES:
        violations.append(f"preplan prototypes exceed {MAX_PROFILE_FILES} files")
    total = 0
    for path in files:
        if path.is_symlink():
            violations.append(
                "preplan prototype may not be a symlink: "
                + str(path.relative_to(workspace))
            )
            continue
        total += path.stat().st_size
        try:
            content = path.read_text(encoding="utf-8").casefold()
        except (UnicodeError, OSError):
            content = ""
        for marker in FORBIDDEN_EVIDENCE_MARKERS:
            if marker in content:
                violations.append(
                    "preplan prototype contains forbidden private-evaluator marker: "
                    + marker
                )
    if total > MAX_PROFILE_BYTES:
        violations.append(f"preplan prototypes exceed {MAX_PROFILE_BYTES} bytes")
    return violations


def render_preplan_prompt(
    campaign: Any, manifest: RepositoryManifest, workspace: Path
) -> str:
    public_dev = (
        "python tools/sandbox.py --kind dev " f"--hardware {campaign.sandbox_hardware}"
    )
    if campaign.sandbox_profile:
        public_dev += f" --gateway-profile {campaign.sandbox_profile}"
    if campaign.sandbox_url:
        public_dev += f" --url {campaign.sandbox_url}"
    public_dev += (
        " --no-sync --input "
        + shlex.quote(manifest.vendor_root)
        + " --input kernel.py "
        "--input input.py --input reference.py -- "
        "python profiles/preplan/<driver>.py"
    )
    values = {
        "WORKSPACE": workspace,
        "PLATFORM": campaign.platform,
        "FRAMEWORK": campaign.framework,
        "SOURCE_NAME": manifest.source_name,
        "SOURCE_REVISION": manifest.revision,
        "EDITABLE_ROOTS": ", ".join(
            f"`{value}`" for value in manifest.editable_workspace_roots
        ),
        "SOURCE_CORPUS": (
            CORPUS_RELATIVE if read_catalog(workspace) is not None else "unavailable"
        ),
        "ARTIFACT": PREPLAN_ARTIFACT.as_posix(),
        "PUBLIC_DEV_COMMAND": public_dev,
        "SANDBOX": campaign._sandbox_directive(),
    }
    prompt = PREPLAN_PROMPT.read_text(encoding="utf-8")
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    unresolved = [
        part.split("}}", 1)[0] for part in prompt.split("{{")[1:] if "}}" in part
    ]
    if unresolved:
        raise RuntimeError(
            "unresolved preplan prompt placeholders: " + ", ".join(unresolved)
        )
    return prompt


def publish_validated_preplan(
    *,
    canonical: Path,
    worktree: Path,
    manifest: RepositoryManifest,
    session_id: str,
    tokens: int | None,
    schema_repair_session_id: str | None = None,
) -> Path:
    """Recheck the plan-only boundary and publish one already-authored artifact."""

    base_commit = git_head(canonical)
    violations: list[str] = []
    if git_head(worktree) != base_commit:
        violations.append(
            "preplan changed Git HEAD; implementation commits are forbidden"
        )
    dirty = working_changes(worktree)
    if dirty:
        violations.append("preplan modified campaign source: " + ", ".join(dirty[:12]))
    if (worktree / ".atrex_long_horizon").exists():
        violations.append("preplan created a formal episode runtime directory")
    artifact = worktree / PREPLAN_ARTIFACT
    violations.extend(validate_preplan_artifact(artifact))
    violations.extend(_profile_violations(worktree))
    corpus = read_catalog(worktree)
    if corpus is not None:
        violations.extend(
            validate_source_corpus(
                worktree,
                corpus,
                expected_runtime=canonical / CORPUS_RELATIVE,
            )
        )
    if violations:
        raise RuntimeError("PREPLAN REJECTED: " + "; ".join(violations))

    canonical_artifact = canonical / PREPLAN_ARTIFACT
    canonical_artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, canonical_artifact)
    digest = hashlib.sha256(canonical_artifact.read_bytes()).hexdigest()
    document = json.loads(canonical_artifact.read_text(encoding="utf-8"))
    evidence_manifest: list[dict[str, str]] = []
    for experiment in document["probing"]["experiments"]:
        raw_output = experiment.get("raw_output_path")
        raw_digest = experiment.get("sha256")
        if not isinstance(raw_output, str) or not isinstance(raw_digest, str):
            continue
        relative = PurePosixPath(raw_output)
        source = worktree.joinpath(*relative.parts)
        destination = canonical.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        evidence_manifest.append({"path": relative.as_posix(), "sha256": raw_digest})
    atomic_write_json(
        canonical / "plans" / "preplan_run.json",
        {
            "schema_version": 2,
            "status": "PASS",
            "source_revision": manifest.revision,
            "incumbent_commit": base_commit,
            "artifact": PREPLAN_ARTIFACT.as_posix(),
            "artifact_sha256": digest,
            "artifact_revision": document["revision"],
            "probe_evidence": evidence_manifest,
            "session_id": session_id,
            "schema_repair_session_id": schema_repair_session_id,
            "tokens": tokens,
            "worktree": str(worktree),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "formal_episode_started": False,
            "candidate_created": False,
        },
    )
    print(
        f"[repository-horizon] PREPLAN PASS artifact={canonical_artifact} "
        f"sha256={digest} formal_episodes=0",
        flush=True,
    )
    return canonical_artifact


@dataclass
class PreplanRunner:
    campaign: Any
    manifest: RepositoryManifest
    timeout: int = 7200

    def _repair_schema(self, worktree: Path, violations: list[str]):
        rendered = "\n".join(f"- {item}" for item in violations[:80])
        prompt = f"""# Repair a Preplan JSON artifact

This is a bounded format-repair session, not a new architecture-planning session. The prior session
authored `{PREPLAN_ARTIFACT.as_posix()}` but deterministic validation rejected its shape.

Preserve every substantive route, experiment, measurement, uncertainty, and ranking conclusion. Do not
research, probe, benchmark, edit tracked files, or introduce a new architecture. Only translate the
existing authored content into the schema shown in
`repository_horizon/prompts/preplan_schema_v3.example.json`, repair references, and run:

`python -m repository_horizon.preplan validate {PREPLAN_ARTIFACT.as_posix()}`

Repeat until it prints `PREPLAN_SCHEMA_VALID`, then stop. Current violations:

{rendered}
"""
        print(
            "[repository-horizon] starting one bounded PREPLAN schema-repair session",
            flush=True,
        )
        return run_session(
            worktree,
            prompt,
            timeout=min(self.timeout, 1200),
            agent_cli=self.campaign.agent_cli,
            sandbox_hardware=self.campaign.sandbox_hardware,
            sandbox_profile=self.campaign.sandbox_profile,
            sandbox_url=self.campaign.sandbox_url,
            sandbox_timeout=self.campaign.sandbox_timeout,
            reasoning_effort="max",
            extra_environment=self.campaign.agent_environment(),
        )

    def run(self) -> Path:
        canonical = self.campaign.workspace
        base_commit = git_head(canonical)
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        worktree = (
            canonical.parent / ".atrex_preplan_worktrees" / canonical.name / run_id
        )
        worktree.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), base_commit],
            cwd=str(canonical),
            check=True,
            capture_output=True,
            text=True,
        )
        link_repository_runtime(self.campaign, worktree, self.manifest)
        corpus = read_catalog(worktree)
        if corpus is not None:
            corpus_violations = validate_source_corpus(
                worktree,
                corpus,
                expected_runtime=canonical / CORPUS_RELATIVE,
            )
            if corpus_violations:
                raise RuntimeError(
                    "preplan source corpus is invalid: " + "; ".join(corpus_violations)
                )

        prompt = render_preplan_prompt(self.campaign, self.manifest, worktree)
        print(
            "[repository-horizon] starting PREPLAN-only session "
            f"worktree={worktree}",
            flush=True,
        )
        result = run_session(
            worktree,
            prompt,
            timeout=self.timeout,
            agent_cli=self.campaign.agent_cli,
            sandbox_hardware=self.campaign.sandbox_hardware,
            sandbox_profile=self.campaign.sandbox_profile,
            sandbox_url=self.campaign.sandbox_url,
            sandbox_timeout=self.campaign.sandbox_timeout,
            reasoning_effort="max",
            extra_environment=self.campaign.agent_environment(),
        )

        invocation_violations: list[str] = []
        if result.timed_out:
            invocation_violations.append("preplan session timed out")
        if result.exit_status != 0:
            invocation_violations.append(
                f"preplan session exited with status {result.exit_status}"
            )
        if invocation_violations:
            raise RuntimeError(
                "PREPLAN REJECTED: "
                + "; ".join(invocation_violations)
                + f"; forensic_worktree={worktree}; "
                + f"stderr={result.stderr_tail[-1000:]}"
            )
        repair = None
        artifact = worktree / PREPLAN_ARTIFACT
        if artifact.is_file():
            schema_violations = validate_preplan_artifact(artifact)
            if schema_violations:
                repair = self._repair_schema(worktree, schema_violations)
                if repair.timed_out or repair.exit_status != 0:
                    raise RuntimeError(
                        "PREPLAN REJECTED: schema-repair session failed; "
                        f"forensic_worktree={worktree}; "
                        f"stderr={repair.stderr_tail[-1000:]}"
                    )
        total_tokens = result.tokens
        if repair is not None and repair.tokens is not None:
            total_tokens = (total_tokens or 0) + repair.tokens
        return publish_validated_preplan(
            canonical=canonical,
            worktree=worktree,
            manifest=self.manifest,
            session_id=result.session_id,
            tokens=total_tokens,
            schema_repair_session_id=(
                repair.session_id if repair is not None else None
            ),
        )


def _main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "validate":
        print("usage: python -m repository_horizon.preplan validate <artifact.json>")
        return 2
    violations = validate_preplan_artifact(Path(argv[1]))
    if violations:
        for violation in violations:
            print(f"PREPLAN_SCHEMA_ERROR: {violation}")
        return 1
    print("PREPLAN_SCHEMA_VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
