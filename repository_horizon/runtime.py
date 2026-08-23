from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from long_horizon.store import CampaignStore
from orchestrator.constants import REPO_ROOT, STALL_STATE_FILE
from orchestrator.optimization_policy import install_workspace_policy
from orchestrator.workspace_runtime import _install_atrex_bench_runtime

from .capabilities import install_capabilities
from .corpus import CORPUS_RELATIVE
from .manifest import RepositoryManifest
from .policy import install_repository_policy


def _link_directory(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            raise RuntimeError(
                f"runtime path points at {target.resolve()}, expected {source.resolve()}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def _link_if_present(source: Path, destination: Path) -> None:
    if source.exists() and not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, destination)


def _append_runtime_ignores(workspace: Path, *, has_evaluator: bool) -> None:
    path = workspace / ".gitignore"
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = set(current.splitlines())
    requested = [
        "/tools",
        "/reference",
        "/skills",
        "/reference-projects",
        "/.claude",
        "/.qoder",
        "/.agents",
        "/.repository_horizon_runtime",
        "/" + STALL_STATE_FILE,
    ]
    if has_evaluator:
        requested.append("/atrex-bench")
    missing = [entry for entry in requested if entry not in lines]
    if not missing:
        return
    with path.open("a", encoding="utf-8") as handle:
        if current and not current.endswith("\n"):
            handle.write("\n")
        handle.write("\n# repository horizon runtime (not campaign source)\n")
        handle.writelines(f"{entry}\n" for entry in missing)


def link_repository_runtime(
    campaign: Any,
    workspace: Path,
    manifest: RepositoryManifest,
) -> None:
    """Install main's runtime surface except GPU Wiki/KernelWiki.

    The repository-local skills, including ``gen-plan``, remain discoverable so the
    backend surface stays aligned with main.  The v4 prompt treats planning as an
    optional engineering tool and never invokes ``gen-plan`` as a required stage.
    GPU Wiki and KernelWiki are deliberately absent from every discovery root.
    """

    CampaignStore.ensure_excluded(workspace)
    install_workspace_policy(
        workspace,
        campaign.optimization_mode,
        campaign.framework,
        agent_runtime=getattr(campaign, "agent_cli", None),
    )
    install_repository_policy(workspace, manifest)

    for name in ("tools", "reference", "skills", "reference-projects"):
        _link_if_present(REPO_ROOT / name, workspace / name)

    atrex_root = Path(campaign.atrex_bench_root) if campaign.atrex_bench_root else None
    if atrex_root is not None:
        _install_atrex_bench_runtime(workspace, atrex_root)

    ncu_skill = REPO_ROOT / "3rdparty" / "ncu-report-skill"
    plan_skill = REPO_ROOT / "skills" / "gen-plan"
    agents = REPO_ROOT / "agents"
    for runtime_name in (".claude", ".qoder"):
        root = workspace / runtime_name
        _link_if_present(ncu_skill, root / "skills" / "ncu-report-skill")
        _link_if_present(plan_skill, root / "skills" / "gen-plan")
        _link_if_present(agents, root / "agents")

    agent_skills = workspace / ".agents" / "skills"
    agent_skills.mkdir(parents=True, exist_ok=True)
    project_skills = REPO_ROOT / "skills"
    if project_skills.is_dir():
        for source in project_skills.iterdir():
            if (source / "SKILL.md").is_file():
                _link_if_present(source, agent_skills / source.name)
    _link_if_present(ncu_skill, agent_skills / "ncu-report-skill")

    corpus = Path(campaign.workspace) / CORPUS_RELATIVE
    if corpus.is_dir():
        _link_directory(corpus, workspace / CORPUS_RELATIVE)

    runtime = workspace / ".repository_horizon_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    capabilities = getattr(campaign, "repository_capabilities", None)
    if isinstance(capabilities, dict):
        install_capabilities(workspace, capabilities)

    forbidden = (
        workspace / "gpu-wiki",
        workspace / ".claude" / "skills" / "KernelWiki",
        workspace / ".qoder" / "skills" / "KernelWiki",
        workspace / ".agents" / "skills" / "KernelWiki",
    )
    exposed = [str(path) for path in forbidden if path.exists() or path.is_symlink()]
    if exposed:
        raise RuntimeError(
            "repository horizon exposed forbidden Wiki assets: " + ", ".join(exposed)
        )

    _append_runtime_ignores(workspace, has_evaluator=atrex_root is not None)


# Backward-compatible name for manifests/tools that imported the v2 helper.  Its
# behavior is v4's main-aligned runtime, not the old asset-free runtime.
install_minimal_runtime = link_repository_runtime


def repository_agent_runtime_directive(agent_cli: str) -> str:
    if agent_cli in {"codex", "pi"}:
        root = ".agents/skills/"
    elif agent_cli == "qodercli":
        root = ".qoder/skills/"
    else:
        root = ".claude/skills/"
    return (
        f"- `{root}` contains the repository-local main skills. `gen-plan` is available "
        "but optional; do not treat it as a required phase. GPU Wiki and KernelWiki are "
        "intentionally not installed for Repository Horizon."
    )
