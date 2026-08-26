from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from repository_horizon.preplan import render_preplan_prompt, validate_preplan_artifact


def cost_term(identifier: str, *, status: str = "speculative") -> dict:
    return {
        "id": identifier,
        "description": f"Cost represented by {identifier}.",
        "status": status,
        "value": None if status in {"speculative", "unknown"} else 1.0,
        "formula": None,
        "unit": "microseconds",
        "evidence": "Public source analysis or a pending public probe.",
    }


def route(identifier: str, mechanism: str) -> dict:
    return {
        "id": identifier,
        "thesis": f"Use {mechanism} to remove a documented obstacle.",
        "implementation_graph": {
            "input_representation": "public input",
            "stages": [
                {
                    "id": f"{identifier}-stage",
                    "operation": mechanism,
                    "input_representation": "public input",
                    "output_representation": "required output",
                }
            ],
            "output_representation": "required output",
        },
        "mechanism_signature": {
            "changed_graph_cuts": [mechanism],
            "representation_path": ["public input", "required output"],
            "compute_mechanism": mechanism,
            "operator_boundary": mechanism,
        },
        "addressed_obstacle_ids": ["obstacle-1"],
        "bridge_assessment_ids": [],
        "changed_choices": ["Inherited boundary"],
        "prerequisites": [],
        "cost_term_ids": ["baseline-cost"],
        "evidence_level": "speculative",
        "evidence_scope": {
            "exact_public_contract": False,
            "proxy_contracts": [],
            "explanation": "No exact-contract measurement yet.",
        },
        "supporting_evidence": ["Public contract and locked source."],
        "contradicting_evidence": [],
        "winning_regimes": ["The obstacle dominates end-to-end cost."],
        "losing_regimes": [],
        "risks": ["The added cost may dominate."],
        "falsification_tests": ["Run a bounded public full-path probe."],
        "ranking_probe": {
            "question": "Can the graph repay its added cost?",
            "cheapest_method": "Run a bounded public probe.",
            "status": "deferred",
            "experiment_id": None,
            "non_execution_reason": "Unit-test fixture has no runtime.",
            "bound_or_contract_evidence": None,
        },
        "search_patterns": ["an optional, non-binding pattern"],
    }


def valid_document() -> dict:
    routes = [route("route-a", "mechanism-a"), route("route-b", "mechanism-b")]
    ids = [item["id"] for item in routes]
    return {
        "schema_version": 3,
        "revision": 1,
        "supersedes": None,
        "objective": {
            "metric": "end_to_end_latency",
            "decision_variables": [
                {
                    "name": "workload-derived-axis",
                    "description": "A mutable choice justified by this contract.",
                }
            ],
            "formulation": "minimize the sum of all end-to-end graph costs",
        },
        "contract_normal_form": {
            category: [
                {
                    "statement": f"Public {category} constraint.",
                    "evidence": "agent_problem.json",
                }
            ]
            for category in ("semantic", "interface", "policy", "hardware")
        }
        | {
            "implementation_freedoms": [
                {
                    "dimension": "operator boundary",
                    "why_mutable": "The public contract does not fix it.",
                    "evidence": "agent_problem.json",
                }
            ]
        },
        "inherited_implementation_choices": [
            {
                "choice": "Current decomposition",
                "evidence": "kernel.py",
                "why_not_a_constraint": "The public contract does not require it.",
            }
        ],
        "unverified_assumptions": [
            {
                "id": "assumption-1",
                "statement": "An added cost can be amortized.",
                "consequence_if_false": "The route loses.",
                "falsification_test": "Measure the complete graph.",
            }
        ],
        "structural_cost_model": {
            "cost_terms": [cost_term("baseline-cost", status="derived_bound")],
            "obstacles": [
                {
                    "id": "obstacle-1",
                    "statement": "A public structural property blocks efficient mapping.",
                    "evidence": "locked_source.py",
                    "blocked_capability": "An otherwise legal efficient mechanism.",
                    "removal_condition": "Change a contract-permitted graph decision.",
                    "single_cut_justification": None,
                }
            ],
            "top_obstacle_ids": ["obstacle-1"],
        },
        "representation_bridge_analysis": {
            "applicability": "not_applicable",
            "non_applicability_reason": "The test fixture declares no representation mismatch.",
            "assessments": [],
        },
        "architecture_frontier": routes,
        "probing": {
            "experiments": [],
            "post_probe_replan": {
                "ranking_before": ids,
                "ranking_after": ids,
                "evidence_that_changed_ranking": [],
                "route_reconsiderations": [
                    {
                        "route_id": item,
                        "evidence_reviewed": "No probe in unit fixture.",
                        "rank_effect": "Remain provisional.",
                    }
                    for item in ids
                ],
                "unresolved_decisive_probe_ids": [],
            },
        },
        "portfolio": {
            "ranked_route_ids": ids,
            "performance_primary_route_id": ids[0],
            "correctness_bridge_route_id": None,
            "hedge_route_ids": [ids[1]],
            "ranking_status": "provisional",
            "selection_rationale": "Best evidence-adjusted end-to-end outlook.",
            "next_experiments": [
                {"id": "experiment-1", "purpose": "Rank routes.", "route_ids": ids}
            ],
            "replan_triggers": ["A required mechanism is falsified."],
            "composition_policies": [],
        },
    }


class FakeCampaign:
    sandbox_hardware = "accelerator"
    sandbox_profile = "public-dev"
    sandbox_url = None
    platform = "platform"
    framework = "framework"

    def _sandbox_directive(self) -> str:
        return "Use only the declared public sandbox."


class PreplanValidationTests(unittest.TestCase):
    def validate(self, document: dict) -> list[str]:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plans" / "frontier.json"
            path.parent.mkdir()
            path.write_text(json.dumps(document), encoding="utf-8")
            return validate_preplan_artifact(path)

    def test_valid_frontier_passes(self) -> None:
        self.assertEqual(self.validate(valid_document()), [])

    def test_taxonomy_is_not_required(self) -> None:
        document = valid_document()
        for item in document["architecture_frontier"]:
            item.pop("search_patterns")
        self.assertEqual(self.validate(document), [])

    def test_requires_mechanism_distinct_routes(self) -> None:
        document = valid_document()
        document["architecture_frontier"][1]["mechanism_signature"] = document[
            "architecture_frontier"
        ][0]["mechanism_signature"]
        violations = self.validate(document)
        self.assertTrue(
            any("duplicates another route" in value for value in violations)
        )

    def test_unknown_obstacle_reference_fails(self) -> None:
        document = valid_document()
        document["architecture_frontier"][0]["addressed_obstacle_ids"] = ["missing"]
        violations = self.validate(document)
        self.assertTrue(any("unknown ids" in value for value in violations))

    def test_measured_cost_requires_numeric_value(self) -> None:
        document = valid_document()
        term = document["structural_cost_model"]["cost_terms"][0]
        term["status"] = "measured"
        term["value"] = None
        violations = self.validate(document)
        self.assertTrue(any("must be numeric" in value for value in violations))

    def test_composition_is_separate_from_base_routes(self) -> None:
        document = valid_document()
        document["portfolio"]["composition_policies"] = [
            {
                "id": "policy-1",
                "route_ids": ["route-a", "route-b"],
                "public_condition": "A public runtime predicate.",
                "added_cost": "One measured selection step.",
                "evidence_level": "estimated",
            }
        ]
        self.assertEqual(self.validate(document), [])

    def test_measured_runtime_probe_requires_persisted_raw_output(self) -> None:
        document = valid_document()
        document["probing"]["experiments"] = [
            {
                "id": "probe-1",
                "kind": "gpu_probe",
                "hypothesis": "A public measurement changes route ranking.",
                "status": "measured",
                "method": "Run a bounded driver.",
                "input_description": "A public synthetic input.",
                "command": "python profiles/preplan/driver.py",
                "environment": "Public development gateway.",
                "evidence_level": "measured",
                "evidence": "The raw output should be persisted.",
                "interpretation": "One route becomes less uncertain.",
                "decision_impact": "Keep the route.",
                "raw_output_path": None,
                "sha256": None,
            }
        ]
        violations = self.validate(document)
        self.assertTrue(
            any("requires raw_output_path" in value for value in violations)
        )

    def test_measured_runtime_probe_accepts_matching_raw_output(self) -> None:
        document = valid_document()
        raw_content = b'{"latency_us": 1.0}\n'
        document["probing"]["experiments"] = [
            {
                "id": "probe-1",
                "kind": "gpu_probe",
                "hypothesis": "A public measurement changes route ranking.",
                "status": "measured",
                "method": "Run a bounded driver.",
                "input_description": "A public synthetic input.",
                "command": "python profiles/preplan/driver.py",
                "environment": "Public development gateway.",
                "evidence_level": "measured",
                "evidence": "profiles/preplan/probe-1.json",
                "interpretation": "One route becomes less uncertain.",
                "decision_impact": "Keep the route.",
                "raw_output_path": "profiles/preplan/probe-1.json",
                "sha256": hashlib.sha256(raw_content).hexdigest(),
            }
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw_path = root / "profiles" / "preplan" / "probe-1.json"
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(raw_content)
            artifact = root / "plans" / "frontier.json"
            artifact.parent.mkdir()
            artifact.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(validate_preplan_artifact(artifact), [])

    def test_private_evaluator_marker_fails(self) -> None:
        document = valid_document()
        document["portfolio"][
            "selection_rationale"
        ] = "Read /private_evaluator/output.json."
        violations = self.validate(document)
        self.assertTrue(any("forbidden" in value for value in violations))

    def test_malformed_route_is_reported_not_raised(self) -> None:
        document = valid_document()
        document["architecture_frontier"][0]["mechanism_signature"] = [{"bad": "type"}]
        document["architecture_frontier"][0]["addressed_obstacle_ids"] = [
            {"bad": "type"}
        ]
        violations = self.validate(document)
        self.assertTrue(violations)

    def test_top_obstacle_requires_distinct_graph_cuts(self) -> None:
        document = valid_document()
        document["architecture_frontier"][1]["mechanism_signature"][
            "changed_graph_cuts"
        ] = ["mechanism-a"]
        violations = self.validate(document)
        self.assertTrue(any("two distinct graph cuts" in value for value in violations))

    def test_frontier_bridge_requires_atomic_route_reference(self) -> None:
        document = valid_document()
        document["representation_bridge_analysis"] = {
            "applicability": "applicable",
            "non_applicability_reason": None,
            "assessments": [
                {
                    "id": "bridge-1",
                    "obstacle_id": "obstacle-1",
                    "source_representation": "public input",
                    "target_representation": "compatible input",
                    "enabled_capability": "efficient capability",
                    "legality": "Allowed by the public contract.",
                    "full_path_cost_equation": "T_transform + T_compute",
                    "cost_term_ids": ["baseline-cost"],
                    "evidence_level": "speculative",
                    "evidence": "Public source.",
                    "disposition": "frontier",
                    "decision_basis": "Needs a bounded probe.",
                    "probe_id": None,
                }
            ],
        }
        document["portfolio"]["correctness_bridge_route_id"] = "route-b"
        violations = self.validate(document)
        self.assertTrue(
            any("frontier representation bridges" in value for value in violations)
        )

    def test_exact_contract_gap_forces_provisional_ranking(self) -> None:
        document = valid_document()
        document["portfolio"]["ranking_status"] = "evidence_complete"
        violations = self.validate(document)
        self.assertTrue(any("must remain provisional" in value for value in violations))

    def test_implementation_graph_must_connect(self) -> None:
        document = valid_document()
        document["architecture_frontier"][0]["implementation_graph"]["stages"][0][
            "input_representation"
        ] = "disconnected"
        violations = self.validate(document)
        self.assertTrue(any("does not connect" in value for value in violations))

    def test_probe_command_uses_manifest_source_not_a_fixed_repository(self) -> None:
        manifest = SimpleNamespace(
            source_name="example_source",
            revision="a" * 40,
            vendor_root="vendor/example_source",
            editable_workspace_roots=("vendor/example_source/src",),
        )
        with tempfile.TemporaryDirectory() as temp:
            prompt = render_preplan_prompt(FakeCampaign(), manifest, Path(temp))
        self.assertIn("--input vendor/example_source", prompt)
        self.assertIn("T_bridge = T(R1 -> R2) + T_C(R2) + T_post", prompt)
        self.assertIn("python -m repository_horizon.preplan validate", prompt)
        self.assertNotIn("vendor/flash_attention", prompt)
        self.assertNotIn("paged representation", prompt.casefold())
        self.assertNotIn("mudi", prompt.casefold())
        self.assertNotIn("index_select", prompt.casefold())


if __name__ == "__main__":
    unittest.main()
