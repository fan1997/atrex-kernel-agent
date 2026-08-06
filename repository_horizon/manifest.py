from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from long_horizon.integration import normalize_relative_path


@dataclass(frozen=True)
class MeasurementConfig:
    warmup: int = 10
    timed_runs: int = 100
    repeats: int = 2
    per_run_timeout: int = 120
    min_improvement_pct: float = 0.0


@dataclass(frozen=True)
class RuntimeRequirement:
    distribution: str
    import_name: str
    version: str


@dataclass(frozen=True)
class RepositoryManifest:
    path: Path
    name: str
    source_name: str
    revision: str
    archive_paths: tuple[str, ...]
    editable_roots: tuple[str, ...]
    adapter: Path
    package_root: str
    measurement: MeasurementConfig
    runtime_requirements: tuple[RuntimeRequirement, ...]

    @property
    def vendor_root(self) -> str:
        return f"vendor/{self.source_name}"

    @property
    def editable_workspace_roots(self) -> tuple[str, ...]:
        return tuple(f"{self.vendor_root}/{path}" for path in self.editable_roots)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _paths(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return tuple(
        normalize_relative_path(_nonempty_string(item, field)) for item in value
    )


def load_manifest(path: str | Path) -> RepositoryManifest:
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("repository manifest schema_version must be 1")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    measurement_payload = payload.get("measurement") or {}
    if not isinstance(measurement_payload, dict):
        raise ValueError("measurement must be an object")
    measurement = MeasurementConfig(
        warmup=int(measurement_payload.get("warmup", 10)),
        timed_runs=int(measurement_payload.get("timed_runs", 100)),
        repeats=int(measurement_payload.get("repeats", 2)),
        per_run_timeout=int(measurement_payload.get("per_run_timeout", 120)),
        min_improvement_pct=float(measurement_payload.get("min_improvement_pct", 0.0)),
    )
    if (
        min(
            measurement.warmup,
            measurement.timed_runs,
            measurement.repeats,
            measurement.per_run_timeout,
        )
        <= 0
    ):
        raise ValueError("measurement counts and timeouts must be positive")
    requirements_payload = payload.get("runtime_requirements") or []
    if not isinstance(requirements_payload, list):
        raise ValueError("runtime_requirements must be a list")
    requirements = []
    for index, item in enumerate(requirements_payload):
        if not isinstance(item, dict):
            raise ValueError(f"runtime_requirements[{index}] must be an object")
        requirements.append(
            RuntimeRequirement(
                distribution=_nonempty_string(item.get("distribution"), "distribution"),
                import_name=_nonempty_string(item.get("import"), "import"),
                version=_nonempty_string(item.get("version"), "version"),
            )
        )
    adapter_value = _nonempty_string(payload.get("adapter"), "adapter")
    adapter = (manifest_path.parent / adapter_value).resolve()
    if not adapter.is_file():
        seeded_adapter = manifest_path.parent / "kernel.py"
        if manifest_path.name == "source_manifest.json" and seeded_adapter.is_file():
            adapter = seeded_adapter
        else:
            raise FileNotFoundError(f"repository adapter not found: {adapter}")
    return RepositoryManifest(
        path=manifest_path,
        name=_nonempty_string(payload.get("name"), "name"),
        source_name=_nonempty_string(source.get("name"), "source.name"),
        revision=_nonempty_string(source.get("revision"), "source.revision"),
        archive_paths=_paths(source.get("archive_paths"), "source.archive_paths"),
        editable_roots=_paths(payload.get("editable_roots"), "editable_roots"),
        adapter=adapter,
        package_root=normalize_relative_path(
            _nonempty_string(source.get("package_root", "."), "source.package_root")
        )
        if source.get("package_root", ".") != "."
        else ".",
        measurement=measurement,
        runtime_requirements=tuple(requirements),
    )
