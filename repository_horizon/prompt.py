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
) -> str:
    """Render the main-compatible episode contract with two explicit opt-outs.

    Repository evaluations run inline because current main's ``LongSessionRunner`` owns
    session recovery and has no external-evaluation suspension protocol.  This keeps
    session, telemetry, and recovery behavior identical to main.
    """

    policy = evaluation_policy or EvaluationPolicy(wait_mode="inline")
    root = shlex.quote(str(module_root()))
    workspace = shlex.quote(str(worktree.path))
    common = (
        f"PYTHONPATH={root} python -m repository_horizon.dev_eval "
        f"--workspace {workspace} --hardware {shlex.quote(campaign.sandbox_hardware)} "
        f"--backend {shlex.quote(policy.backend)} --wait-mode inline "
        f"--wait-timeout {policy.wait_timeout} --agent-cli "
        f"{shlex.quote(getattr(campaign, 'agent_cli', 'claude'))} "
        f"--agent-result-max-bytes {policy.agent_result_max_bytes}"
    )
    endpoint = ""
    if campaign.sandbox_profile:
        endpoint += f" --profile {shlex.quote(campaign.sandbox_profile)}"
    if campaign.sandbox_url:
        endpoint += f" --url {shlex.quote(campaign.sandbox_url)}"
    journal_command = (
        f"PYTHONPATH={root} python -m long_horizon.journal "
        f"--live-path {shlex.quote(str(live_memory_path))}"
    )
    editable_roots = " ".join(
        shlex.quote(value) for value in manifest.editable_workspace_roots
    )
    require_report = ""
    if manifest.repository_search.require_report:
        require_report = (
            "Before a bring-up candidate is terminal, create `plans/repository_search.json` "
            "using the locked bounded corpus. The repository candidate policy validates its "
            "queries, full commit ids, paths, selection, and stated gap."
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
            "DEV_EVAL_COMMAND": common.replace(
                "repository_horizon.dev_eval ",
                "repository_horizon.dev_eval submit ",
                1,
            )
            + endpoint,
            "PROFILE_COMMAND": common.replace(
                "repository_horizon.dev_eval ",
                "repository_horizon.dev_eval profile ",
                1,
            )
            + endpoint
            + " --route auto",
            "REPOSITORY_SEARCH_REQUIREMENT": require_report,
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
