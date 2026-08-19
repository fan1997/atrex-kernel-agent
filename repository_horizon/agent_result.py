from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from long_horizon.models import VerificationResult
from long_horizon.protocol import atomic_write_json

DEFAULT_MAX_BYTES = 16 * 1024


def result_classification(result: VerificationResult) -> str:
    if result.error:
        lowered = result.error.lower()
        if "timeout" in lowered or "timed out" in lowered:
            return "TIMEOUT"
        if "correct" in lowered:
            return "CORRECTNESS_FAIL"
        return "INFRA_FAILURE"
    if result.gate == "PASS":
        return "PASS"
    improvement = result.improvement_pct
    if isinstance(improvement, (int, float)) and improvement < 0:
        return "REGRESSION"
    return "FAIL"


def write_agent_result(
    directory: Path,
    result: VerificationResult,
    *,
    evaluation_id: str,
    backend: str,
    request_digest: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    profile: dict[str, Any] | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "backend": backend,
        "classification": result_classification(result),
        "gate": result.gate,
        "candidate_latency_us": result.candidate_latency_us,
        "incumbent_latency_us": result.incumbent_latency_us,
        "improvement_pct": result.improvement_pct,
        "error": result.error or None,
        "request_digest": request_digest,
        "evidence_directory": str(directory),
        "evidence_index": {
            "full_result": "result.json",
            "transport_evidence": "transport_result.json",
            "artifact_directory": "artifacts/",
        },
    }
    if profile:
        payload["profile"] = profile
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if len(encoded) > max_bytes:
        payload["error"] = str(payload.get("error") or "")[:1000] or None
        payload["profile"] = {
            "summary": "profile detail omitted; inspect evidence selectively"
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError(f"agent result exceeds {max_bytes} bytes")
    path = directory / "agent_result.json"
    atomic_write_json(path, payload)
    return path
