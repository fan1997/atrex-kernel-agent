from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

from long_horizon.store import CampaignStore

from .corpus import CORPUS_RELATIVE
from .manifest import RepositoryManifest
from .policy import install_repository_policy
from .capabilities import install_capabilities


def _link_directory(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        if target.resolve() != source.resolve():
            raise RuntimeError(
                f"runtime path points at {target.resolve()}, expected {source.resolve()}"
            )
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def install_minimal_runtime(
    campaign: Any,
    workspace: Path,
    manifest: RepositoryManifest,
) -> None:
    """Install only repository boundaries and the bounded source corpus."""

    if (workspace / ".git").exists():
        CampaignStore.ensure_excluded(workspace)
    install_repository_policy(workspace, manifest)
    source = Path(campaign.workspace) / CORPUS_RELATIVE
    if source.is_dir():
        _link_directory(source, workspace / CORPUS_RELATIVE)

    runtime = workspace / ".repository_horizon_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    capabilities = getattr(campaign, "repository_capabilities", None)
    if isinstance(capabilities, dict):
        install_capabilities(workspace, capabilities)

    for forbidden in (
        workspace / ".claude" / "agents",
        workspace / ".claude" / "skills",
        workspace / ".qoder" / "agents",
        workspace / ".qoder" / "skills",
        workspace / ".agents" / "skills",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise RuntimeError(f"autonomous runtime exposed forbidden Agent assets: {forbidden}")


def autonomous_environment() -> dict[str, str]:
    environment = dict(os.environ)
    raw = environment.get("ATREX_CODEX_SESSION_SETTINGS", "")
    try:
        settings = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise ValueError("ATREX_CODEX_SESSION_SETTINGS must be valid JSON") from exc
    if isinstance(settings, dict):
        settings["features.multi_agent"] = False
    elif isinstance(settings, list):
        settings = [
            value
            for value in settings
            if not (
                isinstance(value, str)
                and value.split("=", 1)[0] == "features.multi_agent"
            )
        ]
        settings.append("features.multi_agent=false")
    else:
        raise ValueError(
            "ATREX_CODEX_SESSION_SETTINGS must be a JSON object or key=value list"
        )
    environment["ATREX_CODEX_SESSION_SETTINGS"] = json.dumps(settings, separators=(",", ":"))
    return environment
