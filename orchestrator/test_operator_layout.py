from __future__ import annotations

import json
from pathlib import Path

from orchestrator.operator_layout import (
    GENERALIZED_AGENT_VISIBLE_FILES,
    GENERALIZED_EVALUATOR_ONLY_ARTIFACTS,
    agent_visible_operator_files,
)


def test_generalized_workspace_excludes_provider_audit_inputs(tmp_path: Path) -> None:
    operator = tmp_path / "operator"
    operator.mkdir()
    (operator / "reference.py").write_text("class Model: pass\n", encoding="utf-8")
    (operator / "input.py").write_text("def _make_inputs(): return {}\n", encoding="utf-8")
    (operator / "shapes.json").write_text(
        json.dumps({"0": {"init_kwargs": None, "input_kwargs": {"x": 1}}}),
        encoding="utf-8",
    )
    (operator / "agent_problem.json").write_text(
        json.dumps(
            {
                "schema_version": "atrex.agent_problem.v1",
                "objective": "Implement the operator.",
                "evaluation": {
                    "exact_cases": "private",
                    "development_cases_are_evaluation_cases": False,
                },
                "operator_contract": {"kind": "test"},
                "shape_domain": {"x": {"values": [1]}},
                "invariants": ["x is positive"],
                "coverage_regimes": [{"name": "test"}],
                "development_cases": [
                    {"init_kwargs": None, "input_kwargs": {"x": 2}}
                ],
            }
        ),
        encoding="utf-8",
    )
    (operator / "coverage.json").write_text("{}\n", encoding="utf-8")
    (operator / "providers").mkdir()

    visible = agent_visible_operator_files(operator, generalized=True)

    assert visible == GENERALIZED_AGENT_VISIBLE_FILES
    assert set(visible).isdisjoint(GENERALIZED_EVALUATOR_ONLY_ARTIFACTS)
