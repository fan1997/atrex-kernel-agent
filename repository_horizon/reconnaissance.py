from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from long_horizon.git_episode import git_head
from long_horizon.protocol import atomic_write_json

from .compat import normalize_relative_path
from .corpus import (
    CORPUS_RELATIVE,
    corpus_has_commit,
    corpus_has_path,
    read_catalog,
    validate_source_corpus,
)
from .manifest import RepositoryManifest, load_manifest

REPORT_RELATIVE = Path("plans/repository_search.json")
SEAL_RELATIVE = Path(
    ".repository_horizon_runtime/repository_reconnaissance_seal.json"
)


def reconnaissance_required(workspace: Path, manifest: RepositoryManifest) -> bool:
    config = manifest.repository_search
    return (
        config.require_report
        and config.seal_before_first_eval
        and not (workspace / "memory" / "v0.json").is_file()
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _git_lines(workspace: Path, *args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _editable_changes(
    workspace: Path, manifest: RepositoryManifest, base_commit: str
) -> list[str]:
    paths = set(_git_lines(workspace, "diff", "--name-only", base_commit, "--"))
    paths.update(
        _git_lines(workspace, "ls-files", "--others", "--exclude-standard", "--")
    )
    editable = manifest.editable_workspace_roots
    return sorted(
        path
        for path in paths
        if any(
            path == root or path.startswith(root.rstrip("/") + "/")
            for root in editable
        )
    )


def validate_reconnaissance_report(
    workspace: Path,
    manifest: RepositoryManifest,
    *,
    expected_runtime: Path | None = None,
) -> list[str]:
    report_path = workspace / REPORT_RELATIVE
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [f"bring-up requires {REPORT_RELATIVE.as_posix()}"]
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        return ["repository reconnaissance report has unsupported schema"]
    if report.get("source_revision") != manifest.revision:
        return ["repository reconnaissance source_revision differs from R0"]

    queries = report.get("queries")
    candidates = report.get("candidates")
    selected = report.get("selected")
    if (
        not isinstance(queries, list)
        or not queries
        or any(not _nonempty(value) for value in queries)
    ):
        return ["repository reconnaissance must record non-empty queries"]
    if not isinstance(candidates, list):
        return ["repository reconnaissance must record candidate findings"]
    minimum = manifest.repository_search.min_candidates
    if len(candidates) < minimum:
        return [
            "repository reconnaissance requires at least "
            f"{minimum} candidate findings"
        ]
    if not isinstance(selected, dict) or not selected:
        return ["repository reconnaissance must record the selected source finding"]

    catalog = read_catalog(workspace)
    if catalog is None:
        return ["repository reconnaissance requires a bounded source corpus"]
    corpus_violations = validate_source_corpus(
        workspace, catalog, expected_runtime=expected_runtime
    )
    if corpus_violations:
        return corpus_violations

    reported: set[tuple[str, str]] = set()
    narrative_fields = ("mechanism", "workload_relevance", "transfer_gap", "risks")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            return [f"repository reconnaissance candidate {index} must be an object"]
        commit = candidate.get("commit")
        path = candidate.get("path")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or not _nonempty(path)
        ):
            return [
                f"repository reconnaissance candidate {index} requires full commit and path"
            ]
        try:
            normalized_path = normalize_relative_path(str(path))
        except ValueError as exc:
            return [
                f"repository reconnaissance candidate {index} has unsafe path: {exc}"
            ]
        if manifest.repository_search.seal_before_first_eval:
            missing = [
                field
                for field in narrative_fields
                if not _nonempty(candidate.get(field))
            ]
            if missing:
                return [
                    f"repository reconnaissance candidate {index} requires "
                    + ", ".join(missing)
                ]
        if not corpus_has_commit(workspace, commit):
            return [
                f"repository reconnaissance commit is outside bounded corpus: {commit}"
            ]
        if not corpus_has_path(workspace, commit, normalized_path):
            return [
                "repository reconnaissance path is absent from its corpus commit: "
                f"{commit}:{normalized_path}"
            ]
        reported.add((commit, normalized_path))
    if len(reported) < minimum:
        return [
            "repository reconnaissance requires at least "
            f"{minimum} distinct commit/path findings"
        ]

    selected_commit = selected.get("commit")
    selected_path = selected.get("path")
    try:
        normalized_selected_path = normalize_relative_path(str(selected_path))
    except ValueError:
        normalized_selected_path = ""
    selected_narrative_valid = _nonempty(selected.get("gap")) and (
        not manifest.repository_search.seal_before_first_eval
        or _nonempty(selected.get("rationale"))
    )
    if (
        not isinstance(selected_commit, str)
        or (selected_commit, normalized_selected_path) not in reported
        or not selected_narrative_valid
    ):
        explanation = (
            "rationale and gap"
            if manifest.repository_search.seal_before_first_eval
            else "gap"
        )
        return [
            "repository reconnaissance selected entry must reference a reported "
            f"commit/path and explain {explanation}"
        ]
    return []


def _episode_base_commit(workspace: Path) -> str:
    journal_path = workspace / ".atrex_long_horizon/journal.json"
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("episode journal is unavailable") from exc
    base_commit = journal.get("base_commit") if isinstance(journal, dict) else None
    if not isinstance(base_commit, str) or len(base_commit) != 40:
        raise ValueError("episode journal has no full base_commit")
    return base_commit


def seal_reconnaissance(workspace: Path, manifest: RepositoryManifest) -> Path:
    if not reconnaissance_required(workspace, manifest):
        raise ValueError("reconnaissance seal is not required for this episode")
    base_commit = _episode_base_commit(workspace)
    head = git_head(workspace)
    if head != base_commit:
        raise ValueError(
            "reconnaissance must be sealed before any episode commit; "
            f"HEAD is {head}, expected {base_commit}"
        )
    editable_changes = _editable_changes(workspace, manifest, base_commit)
    if editable_changes:
        raise ValueError(
            "reconnaissance must be sealed before editable source changes: "
            + ", ".join(editable_changes[:8])
        )
    violations = validate_reconnaissance_report(workspace, manifest)
    if violations:
        raise ValueError("; ".join(violations))
    catalog = read_catalog(workspace)
    if catalog is None:
        raise ValueError("bounded source corpus catalog is unavailable")
    report_path = workspace / REPORT_RELATIVE
    seal_path = workspace / SEAL_RELATIVE
    atomic_write_json(
        seal_path,
        {
            "schema_version": 1,
            "kind": "repository_reconnaissance",
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "base_commit": base_commit,
            "sealed_head": head,
            "source_revision": manifest.revision,
            "report_path": REPORT_RELATIVE.as_posix(),
            "report_sha256": _sha256(report_path),
            "corpus_commit_set_sha256": catalog.get("commit_set_sha256"),
            "corpus_object_set_sha256": catalog.get("object_set_sha256"),
            "editable_roots": list(manifest.editable_workspace_roots),
        },
    )
    return seal_path


def reconnaissance_gate_violations(
    workspace: Path,
    manifest: RepositoryManifest,
    *,
    expected_runtime: Path | None = None,
) -> list[str]:
    if not reconnaissance_required(workspace, manifest):
        return []
    report_violations = validate_reconnaissance_report(
        workspace, manifest, expected_runtime=expected_runtime
    )
    if report_violations:
        return report_violations
    seal_path = workspace / SEAL_RELATIVE
    try:
        seal: Any = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return [
            "bring-up evaluation requires a clean repository reconnaissance seal"
        ]
    try:
        base_commit = _episode_base_commit(workspace)
    except ValueError as exc:
        return [str(exc)]
    catalog = read_catalog(workspace)
    report_path = workspace / REPORT_RELATIVE
    expected = {
        "schema_version": 1,
        "kind": "repository_reconnaissance",
        "base_commit": base_commit,
        "sealed_head": base_commit,
        "source_revision": manifest.revision,
        "report_path": REPORT_RELATIVE.as_posix(),
        "report_sha256": _sha256(report_path),
        "corpus_commit_set_sha256": (
            catalog.get("commit_set_sha256") if isinstance(catalog, dict) else None
        ),
        "corpus_object_set_sha256": (
            catalog.get("object_set_sha256") if isinstance(catalog, dict) else None
        ),
        "editable_roots": list(manifest.editable_workspace_roots),
    }
    if not isinstance(seal, dict):
        return ["repository reconnaissance seal is malformed"]
    mismatches = [key for key, value in expected.items() if seal.get(key) != value]
    if mismatches:
        return [
            "repository reconnaissance seal does not match " + ", ".join(mismatches)
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seal workload-directed bounded repository reconnaissance"
    )
    parser.add_argument("command", choices=("seal",))
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    try:
        manifest = load_manifest(workspace / "source_manifest.json")
        path = seal_reconnaissance(workspace, manifest)
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            f"[repository-horizon] reconnaissance seal rejected: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"status": "SEALED", "path": str(path.relative_to(workspace))},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
