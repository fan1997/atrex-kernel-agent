from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .compat import EVIDENCE_PREFIXES, normalize_relative_path

from .corpus import (
    CORPUS_RELATIVE,
    corpus_has_commit,
    corpus_has_path,
    read_catalog,
    validate_source_corpus,
)
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
        "source_corpus.json",
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
        if not self.manifest.repository_search.require_report:
            return []
        if (workspace / "memory" / "v0.json").is_file():
            return []
        report_path = workspace / "plans" / "repository_search.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ["bring-up candidate requires plans/repository_search.json"]
        if not isinstance(report, dict) or report.get("schema_version") != 1:
            return ["repository search report has unsupported schema"]
        if report.get("source_revision") != self.manifest.revision:
            return ["repository search report source_revision differs from R0"]
        queries = report.get("queries")
        candidates = report.get("candidates")
        selected = report.get("selected")
        if (
            not isinstance(queries, list)
            or not queries
            or any(not isinstance(value, str) or not value.strip() for value in queries)
        ):
            return ["repository search report must record non-empty queries"]
        if not isinstance(candidates, list) or not candidates:
            return ["repository search report must record candidate findings"]
        if not isinstance(selected, dict) or not selected:
            return ["repository search report must record the selected source path"]
        catalog = read_catalog(workspace)
        if catalog is None:
            return ["repository search report requires a bounded source corpus"]
        expected_runtime = Path(campaign.workspace) / CORPUS_RELATIVE
        corpus_violations = validate_source_corpus(
            workspace, catalog, expected_runtime=expected_runtime
        )
        if corpus_violations:
            return corpus_violations
        reported: set[tuple[str, str]] = set()
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                return [f"repository search candidate {index} must be an object"]
            commit = candidate.get("commit")
            path = candidate.get("path")
            if (
                not isinstance(commit, str)
                or re.fullmatch(r"[0-9a-f]{40}", commit) is None
                or not isinstance(path, str)
                or not path.strip()
            ):
                return [
                    f"repository search candidate {index} requires full commit and path"
                ]
            try:
                path = normalize_relative_path(path)
            except ValueError as exc:
                return [f"repository search candidate {index} has unsafe path: {exc}"]
            if not corpus_has_commit(workspace, commit):
                return [
                    f"repository search candidate commit is outside bounded corpus: {commit}"
                ]
            if not corpus_has_path(workspace, commit, path):
                return [
                    "repository search candidate path is absent from its corpus commit: "
                    f"{commit}:{path}"
                ]
            reported.add((commit, path))
        selected_commit = selected.get("commit")
        selected_path = selected.get("path")
        selected_gap = selected.get("gap")
        if (
            not isinstance(selected_commit, str)
            or not isinstance(selected_path, str)
            or (selected_commit, selected_path) not in reported
            or not isinstance(selected_gap, str)
            or not selected_gap.strip()
        ):
            return [
                "repository search selected entry must reference a reported commit/path and explain the gap"
            ]
        return []
