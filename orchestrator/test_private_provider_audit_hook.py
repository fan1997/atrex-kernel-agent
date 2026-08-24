from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from orchestrator.campaign import Campaign


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_private_provider_audit_runs_outside_workspace_and_is_reused(
    tmp_path: Path, monkeypatch
) -> None:
    benchmark = tmp_path / "atrex-bench"
    operator = benchmark / "data" / "operator"
    provider = operator / "providers" / "provider-a"
    provider.mkdir(parents=True)
    (benchmark / "src").mkdir()
    script = benchmark / "scripts" / "audit_candidate_providers.py"
    script.parent.mkdir()
    script.write_text("# fixture\n", encoding="utf-8")
    (operator / "reference.py").write_text("# reference\n", encoding="utf-8")
    (operator / "input.py").write_text("# input\n", encoding="utf-8")
    (provider / "provider.py").write_text("# provider\n", encoding="utf-8")
    _write_json(provider / "provider.json", {"id": "provider-a"})
    _write_json(
        operator / "shapes.json",
        {"0": {"init_kwargs": None, "input_kwargs": {"seed": 1}}},
    )
    _write_json(operator / "metadata.json", {"shapes": {"0": {}}})
    _write_json(
        operator / "coverage.json",
        {
            "private_provider_audit": {
                "required_provider_ids": ["provider-a"],
                "min_providers_per_shape": 1,
            }
        },
    )

    output_root = tmp_path / "evaluator-private"
    campaign = Campaign(
        name="operator",
        kernel_demo=str(operator / "reference.py"),
        platform="B300",
        framework="Cuda",
        optimization_mode="production",
        atrex_bench_root=str(benchmark),
        work_dir=str(tmp_path / "workspaces"),
        private_provider_audit_root=str(output_root),
        private_provider_audit_cuda_visible_devices="1",
    )
    campaign.workspace.mkdir(parents=True)
    candidate = campaign.workspace / "kernel.py"
    candidate.write_text("# candidate\n", encoding="utf-8")
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    evidence_inputs = campaign._private_provider_evidence_inputs()
    calls: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        calls.append(kwargs["env"])
        artifact = output_root / "operator" / "contract" / "candidate" / "evidence"
        artifact.mkdir(parents=True)
        _write_json(
            artifact / "audit.json",
            {
                "scope": "candidate_human_audit",
                "execution_status": "complete",
                "candidate": {"sha256": candidate_sha256},
                "evidence": {"operator_inputs": evidence_inputs},
                "summary": {
                    "provider_set_complete": True,
                    "provider_coverage_complete": True,
                },
                "findings": {
                    "pair_groups": {
                        "candidate_vs_provider": {"comparison_count": 2}
                    }
                },
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="private", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    first = campaign.ensure_private_provider_audit(version=9)
    second = campaign.ensure_private_provider_audit(version=9)

    assert first == second
    assert first is not None
    assert len(calls) == 1
    assert calls[0]["CUDA_VISIBLE_DEVICES"] == "1"
    assert "ATREX_PRIVATE_REFERENCE_DIR" not in calls[0]
    assert not first.is_relative_to(campaign.workspace)
