from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from long_horizon.models import SupervisorState

from .compat import module_root
from .corpus import CORPUS_RELATIVE, read_catalog
from .manifest import RepositoryManifest


PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "autonomous_episode.md"
MAX_PROMPT_BYTES = 16 * 1024
MAX_HISTORY_BYTES = 8 * 1024
MAX_HISTORY_ATTEMPTS = 5


def compact_history(state: SupervisorState) -> str:
    attempts: list[dict[str, Any]] = []
    for attempt in state.attempts[-MAX_HISTORY_ATTEMPTS:]:
        verification = attempt.get("verification")
        item = {
            key: attempt.get(key)
            for key in (
                "episode",
                "status",
                "accepted",
                "summary",
                "candidate_commit",
                "violation",
            )
            if attempt.get(key) is not None
        }
        if isinstance(verification, dict):
            item["verification"] = {
                key: verification.get(key)
                for key in (
                    "gate",
                    "candidate_latency_us",
                    "incumbent_latency_us",
                    "improvement_pct",
                    "error",
                )
                if verification.get(key) is not None
            }
        directions = attempt.get("next_directions")
        if isinstance(directions, list):
            item["next_directions"] = [
                value[:600]
                for value in directions[:3]
                if isinstance(value, str) and value.strip()
            ]
        summary = item.get("summary")
        if isinstance(summary, str):
            item["summary"] = summary[:1200]
        attempts.append(item)
    while attempts:
        rendered = json.dumps(attempts, ensure_ascii=False, indent=2)
        if len(rendered.encode("utf-8")) <= MAX_HISTORY_BYTES:
            return rendered
        attempts.pop(0)
    return "[]"


def render_prompt(
    *,
    campaign: Any,
    manifest: RepositoryManifest,
    episode: int,
    worktree: Any,
    journal_path: Path,
    handoff_path: Path,
    state: SupervisorState,
) -> str:
    command = (
        f"PYTHONPATH={module_root()} python -m repository_horizon.dev_eval "
        f"--workspace {worktree.path} --hardware {campaign.sandbox_hardware}"
    )
    if campaign.sandbox_profile:
        command += f" --profile {campaign.sandbox_profile}"
    if campaign.sandbox_url:
        command += f" --url {campaign.sandbox_url}"
    journal_command = f"PYTHONPATH={module_root()} python -m long_horizon.journal"
    values: dict[str, object] = {
        "EPISODE": episode,
        "WORKSPACE": worktree.path,
        "BASE_COMMIT": worktree.base_commit,
        "EPISODE_BRANCH": worktree.branch,
        "SOURCE_NAME": manifest.source_name,
        "SOURCE_REVISION": manifest.revision,
        "EDITABLE_ROOTS": ", ".join(
            f"`{value}`" for value in manifest.editable_workspace_roots
        ),
        "SOURCE_CORPUS": (
            CORPUS_RELATIVE if read_catalog(worktree.path) is not None else "unavailable"
        ),
        "JOURNAL_PATH": journal_path,
        "JOURNAL_PATH_SHELL": json.dumps(str(journal_path)),
        "HANDOFF_PATH": handoff_path,
        "NOTES": campaign.notes,
        "DEV_EVAL_COMMAND": command,
        "HISTORY": compact_history(state),
        "JOURNAL_COMMAND": journal_command,
        "STALL_SIGNAL": (
            "The incumbent has not improved for "
            f"{state.consecutive_without_promotion} completed attempts. Consider a materially "
            "different strategy, but choose the engineering process yourself."
            if state.consecutive_without_promotion >= 3
            else ""
        ),
    }
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", str(value))
    size = len(prompt.encode("utf-8"))
    if size > MAX_PROMPT_BYTES:
        raise RuntimeError(
            f"autonomous episode prompt is {size} bytes, limit is {MAX_PROMPT_BYTES}"
        )
    return prompt
