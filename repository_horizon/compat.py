from __future__ import annotations

import inspect
from pathlib import Path, PurePosixPath

from long_horizon.git_episode import (
    EpisodeWorktree,
    changed_paths,
    git_head,
    git_text,
    promote_candidate,
    working_changes,
)
from long_horizon.journal import initialize as initialize_journal
from long_horizon.journal import load as load_journal
from long_horizon.journal import validate_terminal
from long_horizon.models import SupervisorState, VerificationResult
from long_horizon.protocol import atomic_write_json, read_handoff
from long_horizon.session import LongSessionRunner
from long_horizon.campaign import LongHorizonCampaign
from long_horizon.store import CampaignStore, RUNTIME_DIR
from long_horizon.telemetry import summarize_episode
from orchestrator.optimize import Campaign, latest_version

UPSTREAM_BASELINE = "71b16928579474c93039053d2facfeaf7134e268"
EVIDENCE_PREFIXES = ("plans/", "profiles/")


def normalize_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() in {"", "."}
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def assert_upstream_compatible() -> None:
    """Fail early when the intentionally small upstream API surface changes."""

    required = {
        "LongSessionRunner.run": (
            inspect.signature(LongSessionRunner.run).parameters,
            {
                "workspace",
                "prompt",
                "handoff_path",
                "handoff_resumes",
                "completion_check",
                "telemetry_environment",
            },
        ),
        "EpisodeWorktree.plan": (
            inspect.signature(EpisodeWorktree.plan).parameters,
            {"incumbent_workspace", "episode", "base_commit"},
        ),
        "promote_candidate": (
            inspect.signature(promote_candidate).parameters,
            {"incumbent_workspace", "base_commit", "candidate_commit", "episode"},
        ),
        "LongHorizonCampaign._link_episode_runtime": (
            inspect.signature(LongHorizonCampaign._link_episode_runtime).parameters,
            {"self", "workspace"},
        ),
        "LongHorizonCampaign._validate_candidate": (
            inspect.signature(LongHorizonCampaign._validate_candidate).parameters,
            {"self", "worktree", "candidate_commit"},
        ),
    }
    failures = []
    for label, (parameters, expected) in required.items():
        missing = expected - set(parameters)
        if missing:
            failures.append(f"{label} missing {sorted(missing)}")
    if failures:
        raise RuntimeError(
            "repository_horizon is incompatible with this upstream checkout: "
            + "; ".join(failures)
        )


def module_root() -> Path:
    return Path(__file__).resolve().parent.parent
