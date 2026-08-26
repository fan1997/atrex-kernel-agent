from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from long_horizon import main_adapter

from .compat import module_root
from .config import EvaluationPolicy
from .corpus import CORPUS_RELATIVE, read_catalog
from .manifest import RepositoryManifest
from .policy import repository_mode_directive
from .reconnaissance import reconnaissance_required
from .runtime import repository_agent_runtime_directive

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "episode.md"
MAX_PROMPT_BYTES = 24 * 1024


def _render(template: str, values: dict[str, object]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def render_prompt(
    *,
    campaign: Any,
    manifest: RepositoryManifest,
    episode: int,
    version: int,
    worktree: Any,
    journal_path: Path,
    handoff_path: Path,
    live_memory_path: Path,
    evaluation_policy: EvaluationPolicy | None = None,
    route_directive: str = "",
) -> str:
    """Render the main-compatible episode contract with two explicit opt-outs.

    Repository evaluations run inline because current main's ``LongSessionRunner`` owns
    session recovery and has no external-evaluation suspension protocol.  This keeps
    session, telemetry, and recovery behavior identical to main.
    """

    root = shlex.quote(str(module_root()))
    workspace = shlex.quote(str(worktree.path))
    public_dev = (
        "python tools/sandbox.py --kind dev "
        f"--hardware {shlex.quote(campaign.sandbox_hardware)}"
    )
    if campaign.sandbox_profile:
        public_dev += f" --gateway-profile {shlex.quote(campaign.sandbox_profile)}"
    if campaign.sandbox_url:
        public_dev += f" --url {shlex.quote(campaign.sandbox_url)}"
    public_dev += (
        " --no-sync --input vendor/flash_attention --input kernel.py "
        "--input input.py --input reference.py -- "
        "python profiles/<episode>/public_driver.py"
    )
    journal_command = (
        f"PYTHONPATH={root} python -m long_horizon.journal "
        f"--live-path {shlex.quote(str(live_memory_path))}"
    )
    editable_roots = " ".join(
        shlex.quote(value) for value in manifest.editable_workspace_roots
    )
    require_report = ""
    if manifest.repository_search.require_report:
        if reconnaissance_required(worktree.path, manifest):
            seal_command = (
                f"PYTHONPATH={root} python -m repository_horizon.reconnaissance "
                f"seal --workspace {workspace}"
            )
            require_report = (
                "## Mandatory pre-bring-up repository reconnaissance\n\n"
                "Before modifying an editable root, making an episode commit, or invoking "
                "development evaluation/profile, inspect the locked R0 snapshot and its "
                "bounded source corpus. Use only workspace-local repository evidence; do not "
                "fetch, clone, or consult remote GitHub/PR material. Search specifically for "
                "mechanisms relevant to the public workload contract, then write the findings "
                "best-first to `plans/repository_search.json`. The report uses schema_version "
                "1, the locked source_revision, non-empty queries, at least "
                f"{manifest.repository_search.min_candidates} distinct candidates, and one "
                "selected entry. Every candidate requires `commit`, `path`, `mechanism`, "
                "`workload_relevance`, `transfer_gap`, and `risks`; selected requires a "
                "reported `commit`/`path`, `rationale`, and `gap`. This is bounded source "
                "archaeology, not a mandatory `gen-plan`.\n\n"
                "Seal the report while HEAD still equals the episode base and editable roots "
                "are unchanged:\n\n```bash\n"
                + seal_command
                + "\n```\n\nDevelopment evaluation rejects the episode until this seal passes. "
                "After sealing, choose one coherent direction and begin implementation."
            )
        else:
            require_report = (
                "Before a bring-up candidate is terminal, create "
                "`plans/repository_search.json` using the locked bounded corpus. "
                "The repository candidate policy validates its queries, full commit ids, "
                f"at least {manifest.repository_search.min_candidates} distinct paths, "
                "selection, and stated gap."
            )
    prompt = _render(
        PROMPT_PATH.read_text(encoding="utf-8"),
        {
            "EPISODE": episode,
            "VERSION": version,
            "WORKSPACE": worktree.path,
            "PLATFORM": campaign.platform,
            "FRAMEWORK": campaign.framework,
            "BASE_COMMIT": worktree.base_commit,
            "EPISODE_BRANCH": worktree.branch,
            "SOURCE_NAME": manifest.source_name,
            "SOURCE_REVISION": manifest.revision,
            "EDITABLE_ROOTS": ", ".join(
                f"`{value}`" for value in manifest.editable_workspace_roots
            ),
            "EDITABLE_ROOTS_SHELL": editable_roots,
            "SOURCE_CORPUS": (
                CORPUS_RELATIVE
                if read_catalog(worktree.path) is not None
                else "unavailable"
            ),
            "JOURNAL_PATH": journal_path,
            "JOURNAL_PATH_SHELL": json.dumps(str(journal_path)),
            "HANDOFF_PATH": handoff_path,
            "NOTES": campaign.notes,
            "MODE_POLICY": repository_mode_directive(manifest),
            "HARDWARE": main_adapter.hardware_directive(
                campaign.platform, campaign.arch
            ),
            "SANDBOX": campaign._sandbox_directive(),
            "AGENT_RUNTIME": repository_agent_runtime_directive(
                getattr(campaign, "agent_cli", "claude")
            ),
            "PUBLIC_DEV_COMMAND": public_dev,
            "REPOSITORY_SEARCH_REQUIREMENT": require_report,
            "ROUTE_DIRECTIVE": route_directive,
            "JOURNAL_COMMAND": journal_command,
        },
    )
    unresolved = sorted(
        part.split("}}", 1)[0] for part in prompt.split("{{")[1:] if "}}" in part
    )
    if unresolved:
        raise RuntimeError(
            "unresolved repository prompt placeholders: " + ", ".join(unresolved)
        )
    if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
        raise RuntimeError(
            f"repository episode prompt exceeds {MAX_PROMPT_BYTES} bytes"
        )
    return prompt
