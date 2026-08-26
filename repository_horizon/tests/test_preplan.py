from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from repository_horizon.preplan import REQUIRED_FAMILIES, validate_preplan_artifact


def valid_document() -> dict:
    routes = []
    for index, family in enumerate(sorted(REQUIRED_FAMILIES)):
        routes.append(
            {
                "id": f"route-{index}",
                "family": family,
                "hypothesis": "A falsifiable latency hypothesis.",
                "data_representation": "Consume or transform the public input.",
                "pipeline": ["stage one", "stage two"],
                "transformation_cost": {
                    "status": "unknown",
                    "latency_us": None,
                    "evidence": "Not measured; first next experiment.",
                },
                "unlocked_fast_path": "A simpler memory path.",
                "winning_regimes": ["Conversion is amortized."],
                "losing_regimes": ["Conversion dominates."],
                "required_mechanisms": ["Public-contract dispatch."],
                "risks": ["Crossover may be narrow."],
                "falsification_tests": [
                    {
                        "question": "Does cost amortize?",
                        "method": "Run a public synthetic probe.",
                        "success_criterion": "Total latency is lower.",
                        "failure_action": "Drop or narrow this route.",
                    }
                ],
            }
        )
    ids = [route["id"] for route in routes]
    return {
        "schema_version": 1,
        "objective": {
            "metric": "end_to_end_latency",
            "decision_variables": [
                "representation",
                "preprocessing",
                "decomposition",
                "algorithm",
                "schedule",
                "dispatch",
            ],
            "formulation": (
                "min over decisions of transform plus compute plus launch latency"
            ),
        },
        "constraints": {
            name: [
                {
                    "statement": f"{name} constraint",
                    "evidence": "agent_problem.json",
                }
            ]
            for name in ("semantic", "interface", "policy", "hardware")
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
                "id": "a1",
                "statement": "Transform cost amortizes.",
                "consequence_if_false": "The route loses.",
                "falsification_test": "Measure transform plus compute.",
            }
        ],
        "architecture_frontier": routes,
        "probing": {"experiments": []},
        "portfolio": {
            "ranked_route_ids": ids,
            "primary_route_id": ids[0],
            "hedge_route_ids": [ids[1]],
            "selection_rationale": "Best information-adjusted latency.",
            "next_experiments": [
                {"id": "e1", "purpose": "Measure crossover."}
            ],
        },
    }


class PreplanValidationTests(unittest.TestCase):
    def test_valid_frontier_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frontier.json"
            path.write_text(json.dumps(valid_document()), encoding="utf-8")
            self.assertEqual(validate_preplan_artifact(path), [])

    def test_missing_architecture_family_fails(self) -> None:
        document = valid_document()
        removed = document["architecture_frontier"].pop()
        document["portfolio"]["ranked_route_ids"].remove(removed["id"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frontier.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            violations = validate_preplan_artifact(path)
        self.assertTrue(
            any("misses required families" in value for value in violations)
        )

    def test_private_evaluator_marker_fails(self) -> None:
        document = valid_document()
        document["portfolio"]["selection_rationale"] = (
            "Read private_evaluator output."
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frontier.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            violations = validate_preplan_artifact(path)
        self.assertTrue(any("forbidden" in value for value in violations))


if __name__ == "__main__":
    unittest.main()
