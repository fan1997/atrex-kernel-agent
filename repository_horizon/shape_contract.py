from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from orchestrator.operator_layout import validate_private_shapes

SHAPE_TRAIN_FILENAME = "shape_train.json"
SHAPE_VALID_FILENAME = "shape_valid.json"
LEGACY_SHAPES_FILENAME = "shapes.json"
SHAPE_TRAIN_SCHEMA_VERSION = "atrex.shape_train.v1"


def has_split_shape_contract(directory: Path) -> bool:
    """Return whether *directory* contains the complete train/valid contract."""
    train = (directory / SHAPE_TRAIN_FILENAME).is_file()
    valid = (directory / SHAPE_VALID_FILENAME).is_file()
    if train != valid:
        missing = SHAPE_VALID_FILENAME if train else SHAPE_TRAIN_FILENAME
        raise ValueError(
            f"incomplete split shape contract in {directory}: missing {missing}"
        )
    return train


def exact_shapes_path(directory: Path) -> Path:
    """Resolve evaluator-owned cases, preferring the modern split contract."""
    if (directory / SHAPE_VALID_FILENAME).is_file():
        return directory / SHAPE_VALID_FILENAME
    return directory / LEGACY_SHAPES_FILENAME


def load_exact_shapes(directory: Path) -> dict[str, dict[str, Any]]:
    return validate_private_shapes(exact_shapes_path(directory))


def _case_signature(value: dict[str, Any]) -> str:
    return json.dumps(
        {
            "init_kwargs": value.get("init_kwargs"),
            "input_kwargs": value.get("input_kwargs"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_shape_train(path: Path, *, private_shapes_path: Path) -> dict[str, Any]:
    """Validate the public half of an Atrex-Bench train/valid shape contract."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {SHAPE_TRAIN_FILENAME}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if payload.get("schema_version") != SHAPE_TRAIN_SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema_version must be {SHAPE_TRAIN_SCHEMA_VERSION!r}"
        )
    if not isinstance(payload.get("objective"), str) or not payload["objective"].strip():
        raise ValueError(f"{path}.objective must be a non-empty string")
    for field in ("operator_contract", "shape_domain"):
        if not isinstance(payload.get(field), dict) or not payload[field]:
            raise ValueError(f"{path}.{field} must be a non-empty JSON object")
    invariants = payload.get("invariants")
    if not isinstance(invariants, list) or not all(
        isinstance(value, str) and value.strip() for value in invariants
    ):
        raise ValueError(f"{path}.invariants must be a list of non-empty strings")
    regimes = payload.get("coverage_regimes")
    if not isinstance(regimes, list) or not all(isinstance(value, dict) for value in regimes):
        raise ValueError(f"{path}.coverage_regimes must be a list of objects")
    development_cases = payload.get("development_cases", [])
    if not isinstance(development_cases, list) or not all(
        isinstance(value, dict)
        and isinstance(value.get("init_kwargs"), (dict, type(None)))
        and isinstance(value.get("input_kwargs"), dict)
        for value in development_cases
    ):
        raise ValueError(
            f"{path}.development_cases must contain init_kwargs/input_kwargs objects"
        )
    private_shapes = validate_private_shapes(private_shapes_path)
    hidden = {_case_signature(value) for value in private_shapes.values()}
    duplicates = [
        str(value.get("name", index))
        for index, value in enumerate(development_cases)
        if _case_signature(value) in hidden
    ]
    if duplicates:
        raise ValueError(
            f"{path}.development_cases must not duplicate private validation cases: "
            + ", ".join(duplicates[:8])
        )
    return payload
