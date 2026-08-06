from __future__ import annotations

from pathlib import Path

from orchestrator.optimization_policy import POLICY_BEGIN, POLICY_END

from .manifest import RepositoryManifest


def _directive(manifest: RepositoryManifest) -> str:
    roots = ", ".join(f"`{value}`" for value in manifest.editable_workspace_roots)
    return (
        "## Repository-assisted production mode (hard gate)\n\n"
        "This generated section is the Repository Horizon specialization of "
        "production mode. It overrides generic single-file guidance elsewhere "
        "in `CLAUDE.md`.\n\n"
        "- The implementation framework is exactly **CuteDSL**.\n"
        f"- The locked `{manifest.source_name}` source under `vendor/` is the "
        "implementation being developed, not a forbidden prebuilt-operator "
        "shortcut. Its manifest-declared internal modules and runtime requirements "
        "are allowed. Do not dispatch to a separately installed implementation.\n"
        f"- Source changes are limited to {roots}. The adapter, evaluator, workload, "
        "source lock, corpus catalog, and vendored runtime support are immutable.\n"
        "- The repository candidate contract, bounded-corpus integrity check, and "
        "independent Agate verifier are the mechanical acceptance authority.\n"
    )


def install_repository_policy(workspace: Path, manifest: RepositoryManifest) -> None:
    """Replace generic single-file production prose with repository semantics."""

    path = workspace / "CLAUDE.md"
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    generated = f"{POLICY_BEGIN}\n\n{_directive(manifest).rstrip()}\n\n{POLICY_END}\n"
    if POLICY_BEGIN in current and POLICY_END in current:
        before, remainder = current.split(POLICY_BEGIN, 1)
        _, after = remainder.split(POLICY_END, 1)
        prefix = before.rstrip()
        current = (prefix + "\n\n" if prefix else "") + generated + after.lstrip("\n")
    else:
        current = current.rstrip() + ("\n\n" if current.strip() else "") + generated
    path.write_text(current, encoding="utf-8")
