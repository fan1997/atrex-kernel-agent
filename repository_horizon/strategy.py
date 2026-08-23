from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from long_horizon.protocol import atomic_write_json

RUNTIME_DIR = ".repository_horizon_runtime"
STRATEGY_FILE = "strategy_state.json"
ARCHITECTURE_MAP_FILE = "architecture_map.json"
WIP_PATCH_FILE = "architecture_wip.patch"
ARCHITECTURE_MAP_WORKTREE_PATH = Path("plans/architecture_map.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArchitectureStrategyState:
    schema_version: int = 1
    mode: str = "normal"
    cycle: int = 0
    entered_episode: int | None = None
    last_review_episode: int = 0
    commitment_remaining: int = 0
    active_direction_id: str = ""
    active_thesis: str = ""
    wip_base_commit: str = ""
    wip_source_commit: str = ""
    wip_patch_sha256: str = ""
    review_required: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: object) -> "ArchitectureStrategyState":
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            return cls()
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


class RepositoryStrategyStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / RUNTIME_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.root / STRATEGY_FILE

    @property
    def architecture_map_path(self) -> Path:
        return self.root / ARCHITECTURE_MAP_FILE

    @property
    def wip_patch_path(self) -> Path:
        return self.root / WIP_PATCH_FILE

    def load(self) -> ArchitectureStrategyState:
        try:
            return ArchitectureStrategyState.from_dict(
                json.loads(self.state_path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            return ArchitectureStrategyState()

    def save(self, state: ArchitectureStrategyState) -> None:
        atomic_write_json(self.state_path, asdict(state))

    def enter_if_needed(
        self,
        *,
        episode: int,
        consecutive_without_promotion: int,
        escape_after: int,
        review_interval: int,
        commitment_episodes: int,
    ) -> ArchitectureStrategyState:
        state = self.load()
        stalled = escape_after > 0 and consecutive_without_promotion >= escape_after
        periodic = (
            review_interval > 0
            and episode - state.last_review_episode >= review_interval
        )
        if state.mode != "architecture" and (stalled or periodic):
            state.mode = "architecture"
            state.cycle += 1
            state.entered_episode = episode
            state.commitment_remaining = max(1, commitment_episodes)
            state.review_required = False
            state.history.append(
                {
                    "event": "architecture_escape_entered",
                    "episode": episode,
                    "reason": "promotion_stall" if stalled else "periodic_review",
                    "consecutive_without_promotion": consecutive_without_promotion,
                    "timestamp": _utc_now(),
                }
            )
            self.save(state)
        return state

    def stage_runtime_snapshot(self, episode_workspace: Path) -> None:
        destination = episode_workspace / RUNTIME_DIR
        destination.mkdir(parents=True, exist_ok=True)
        for source in (self.state_path, self.architecture_map_path):
            if source.is_file():
                (destination / source.name).write_bytes(source.read_bytes())
        if self.architecture_map_path.is_file():
            worktree_map = episode_workspace / ARCHITECTURE_MAP_WORKTREE_PATH
            worktree_map.parent.mkdir(parents=True, exist_ok=True)
            worktree_map.write_bytes(self.architecture_map_path.read_bytes())

    def apply_wip(
        self, episode_workspace: Path, state: ArchitectureStrategyState
    ) -> bool:
        if state.mode != "architecture" or not self.wip_patch_path.is_file():
            return False
        patch_bytes = self.wip_patch_path.read_bytes()
        if (
            state.wip_patch_sha256
            and hashlib.sha256(patch_bytes).hexdigest() != state.wip_patch_sha256
        ):
            raise RuntimeError(
                "architecture WIP patch checksum does not match strategy state"
            )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(episode_workspace),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if state.wip_base_commit and head != state.wip_base_commit:
            raise RuntimeError(
                "architecture WIP base no longer matches the incumbent; suspend or rebase it "
                "explicitly instead of applying it implicitly"
            )
        completed = subprocess.run(
            ["git", "apply", "--3way", str(self.wip_patch_path)],
            cwd=str(episode_workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "cannot restore architecture WIP onto incumbent: "
                + (completed.stderr.strip() or completed.stdout.strip())[-1200:]
            )
        return True


def validate_architecture_map(value: object) -> str:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return "architecture map must be a schema_version 1 object"
    directions = value.get("directions")
    if not isinstance(directions, list) or len(directions) < 3:
        return "architecture map must contain at least three directions"
    ids: set[str] = set()
    required = (
        "id",
        "thesis",
        "required_mechanisms",
        "evidence_for",
        "evidence_against",
        "falsification_test",
        "status",
    )
    for direction in directions:
        if not isinstance(direction, dict):
            return "architecture map directions must be objects"
        for key in required:
            if key not in direction:
                return f"architecture direction is missing {key}"
        direction_id = direction.get("id")
        if not isinstance(direction_id, str) or not direction_id.strip():
            return "architecture direction id must be non-empty"
        if direction_id in ids:
            return f"duplicate architecture direction id: {direction_id}"
        ids.add(direction_id)
        if not isinstance(direction.get("required_mechanisms"), list):
            return f"architecture direction {direction_id} required_mechanisms must be a list"
    selected = value.get("selected_direction_id")
    if not isinstance(selected, str) or selected not in ids:
        return "architecture map selected_direction_id must name a reported direction"
    return ""


def load_architecture_map(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"architecture map is missing or invalid: {exc}"
    diagnosis = validate_architecture_map(value)
    return (value if not diagnosis else None), diagnosis


def validate_architecture_outcome(
    outcome: object,
    *,
    terminal_status: str,
    has_last_trial_commit: bool,
) -> str:
    if not isinstance(outcome, dict):
        return "architecture episode requires an outcome object"
    architecture = outcome.get("architecture")
    if not isinstance(architecture, dict):
        return "architecture episode outcome requires an architecture object"
    for key in ("direction_id", "thesis", "disposition"):
        if not str(architecture.get(key, "")).strip():
            return f"architecture outcome requires non-empty {key}"
    disposition = architecture.get("disposition")
    allowed = {
        "continue",
        "implementation_refuted",
        "suspend",
        "architecture_refuted",
        "promote",
    }
    if disposition not in allowed:
        return "architecture disposition is invalid"
    options = architecture.get("next_implementation_options", [])
    if not isinstance(options, list) or any(
        not isinstance(item, str) for item in options
    ):
        return "architecture next_implementation_options must be a list of strings"
    if (
        disposition == "continue"
        and terminal_status != "candidate_ready"
        and not has_last_trial_commit
    ):
        return "continuing architecture work requires last_trial_commit"
    if disposition == "architecture_refuted":
        if architecture.get("feature_parity_complete") is not True:
            return "architecture refutation requires feature_parity_complete=true"
        variants = architecture.get("tested_implementation_variants", 0)
        if not isinstance(variants, int) or isinstance(variants, bool) or variants < 2:
            return (
                "architecture refutation requires at least two implementation variants"
            )
        review = architecture.get("independent_review")
        if not isinstance(review, dict) or review.get("status") != "completed":
            return "architecture refutation requires a completed independent review"
    if terminal_status == "candidate_ready" and disposition != "promote":
        return "architecture candidate_ready outcome must use disposition=promote"
    return ""


def render_strategy_directive(
    state: ArchitectureStrategyState,
    *,
    escape_after: int,
    review_interval: int,
    commitment_episodes: int,
    wip_applied: bool,
) -> str:
    if state.mode != "architecture":
        return f"""## Search-horizon policy

The supervisor tracks local-optimum risk independently of ordinary candidate promotion. After
{escape_after} consecutive unpromoted episodes, or at the periodic {review_interval}-episode
architecture review, it enters architecture-escape mode. Until then, preserve a distinction between
an implementation failure and an architecture refutation. Do not describe a family as exhausted
unless its required mechanisms were present in an equivalent test.
"""
    remaining = max(1, state.commitment_remaining)
    active = state.active_direction_id or "to be selected from the architecture map"
    budget_note = (
        f"protected commitment remaining: {remaining} episode(s)"
        if state.commitment_remaining > 0
        else "the initial commitment budget is exhausted; select a materially different implementation or suspend the direction"
    )
    review_note = (
        "A strategy reset is required now: do not repeat the last implementation."
        if state.review_required
        else ""
    )
    return f"""## Architecture-escape mode (binding)

The campaign is escaping a possible local optimum. Cycle {state.cycle} is active; direction:
`{active}`; {budget_note} of an initial
{commitment_episodes}-episode budget. Architecture work may be temporarily slower, incomplete, or
unable to run. Promotion latency is not a valid reason to abandon it before feature parity.
{review_note}

First read `.repository_horizon_runtime/strategy_state.json` and the architecture map if present.
Create or update `plans/architecture_map.json` by exhaustively revisiting all architecture-scale
directions visible in R0, the bounded source corpus, and canonical campaign memory, then add
first-principles hardware alternatives. Do not consult material outside the isolated workspace. The
map must use schema_version 1, contain at least three directions, and give every
direction `id`, `thesis`, `required_mechanisms`, `evidence_for`, `evidence_against`,
`falsification_test`, and `status`, plus a valid `selected_direction_id`.

Commit to the selected architectural thesis, not to one patch. Preserve belief while a target
mechanism remains plausible, but stay flexible about loaders, layouts, schedules, pipelines, and
staging. After every failed implementation, state which implementation was falsified, which required
mechanism was absent, and at least two materially different next implementation options.

Do not permanently refute an architecture using an out-of-domain proxy or a build missing one of its
required mechanisms. `architecture_refuted` requires feature parity, at least two materially distinct
implementation variants, and a completed independent reviewer challenge. If a reviewer is unavailable,
use `suspend`, not `architecture_refuted`. `gen-plan` remains optional; reviewer consultation is needed
only for permanent architecture refutation.

For a non-promotable but valuable checkpoint, commit the WIP and return its HEAD as
`last_trial_commit`; the supervisor will carry the full patch into the next architecture episode.
WIP restored for this episode: {str(wip_applied).lower()}.

Every terminal outcome in this mode must include:

```json
"architecture": {{
  "direction_id": "stable-id",
  "thesis": "hardware/algorithm reason for upside",
  "disposition": "continue | implementation_refuted | suspend | architecture_refuted | promote",
  "feature_parity_complete": false,
  "tested_implementation_variants": 1,
  "missing_mechanisms": ["..."],
  "next_implementation_options": ["materially different option A", "option B"],
  "independent_review": {{"status": "not_required | unavailable | completed", "evidence": "plans/<review-file>"}}
}}
```
"""
