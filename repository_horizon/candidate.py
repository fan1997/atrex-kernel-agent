from __future__ import annotations

from pathlib import Path
from typing import Any

from long_horizon.integration import EVIDENCE_PREFIXES, normalize_relative_path

from .manifest import RepositoryManifest

PROTECTED = frozenset(
    {
        "kernel.py",
        "test_kernel.py",
        "reference.py",
        "input.py",
        "shapes.json",
        "metadata.json",
        "roofline.json",
        "valid.py",
        "source_manifest.json",
        "source.lock.json",
        ".gitignore",
    }
)
PROTECTED_PREFIXES = (
    "memory/",
    ".atrex_long_horizon/",
    "atrex-bench/",
    "vendor_support/",
)


class RepositoryCandidateContract:
    def __init__(self, manifest: RepositoryManifest):
        self.manifest = manifest

    def _editable(self, path: str) -> bool:
        normalized = normalize_relative_path(path)
        return any(
            normalized == root or normalized.startswith(root.rstrip("/") + "/")
            for root in self.manifest.editable_workspace_roots
        )

    def validate_changed_paths(self, paths: list[str]) -> str:
        source_changes = []
        for raw in paths:
            path = normalize_relative_path(raw)
            if path in PROTECTED or path.startswith(PROTECTED_PREFIXES):
                return f"candidate modified protected path: {path}"
            if path.startswith(EVIDENCE_PREFIXES):
                continue
            if not self._editable(path):
                return f"candidate modified undeclared repository path: {path}"
            source_changes.append(path)
        if not source_changes:
            roots = ", ".join(self.manifest.editable_workspace_roots)
            return (
                f"candidate must change at least one editable repository path: {roots}"
            )
        return ""

    def verification_paths(self, paths: list[str]) -> list[str]:
        return [path for path in paths if self._editable(path)]

    def workspace_violations(self, campaign: Any, workspace: Path) -> list[str]:
        return []
