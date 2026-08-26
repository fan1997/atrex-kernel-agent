from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from long_horizon.git_episode import git_head
from long_horizon.protocol import atomic_write_json

from .manifest import RepositoryManifest
from .preplan import (
    FORBIDDEN_EVIDENCE_MARKERS,
    MAX_PROFILE_BYTES,
    MAX_PROFILE_FILES,
    PREPLAN_ARTIFACT,
    validate_preplan_artifact,
)

RUNTIME_ROOT = Path(".atrex_long_horizon/fixed_preplan_route")
EPISODE_BUNDLE = Path(".repository_horizon_runtime/fixed_preplan_route")
STATE_FILE = "state.json"
BUNDLE_DIR = "bundle"
WIP_PATCH_FILE = "wip.patch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_relative(value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"unsafe Preplan evidence path: {value!r}")
    return Path(*relative.parts)


@dataclass
class FixedRouteState:
    schema_version: int = 1
    route_id: str = ""
    policy: str = "fixed_deep_optimization"
    artifact_sha256: str = ""
    artifact_revision: int = 0
    source_revision: str = ""
    preplan_incumbent_commit: str = ""
    total_episode_target: int = 0
    selected_at: str = ""
    wip_base_commit: str = ""
    wip_source_commit: str = ""
    wip_patch_sha256: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: object) -> "FixedRouteState":
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("fixed Preplan route state is missing or incompatible")
        fields = cls.__dataclass_fields__
        return cls(**{key: value[key] for key in fields if key in value})


class FixedPreplanRouteStore:
    """Persist and stage one immutable Preplan route for a whole campaign."""

    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / RUNTIME_ROOT

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def bundle_path(self) -> Path:
        return self.root / BUNDLE_DIR

    @property
    def wip_patch_path(self) -> Path:
        return self.root / WIP_PATCH_FILE

    def load(self) -> FixedRouteState:
        try:
            return FixedRouteState.from_dict(
                json.loads(self.state_path.read_text(encoding="utf-8"))
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot load fixed Preplan route state: {exc}") from exc

    def save(self, state: FixedRouteState) -> None:
        atomic_write_json(self.state_path, asdict(state))

    def _validated_inputs(
        self, route_id: str, manifest: RepositoryManifest
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        artifact = self.workspace / PREPLAN_ARTIFACT
        violations = validate_preplan_artifact(artifact)
        if violations:
            raise RuntimeError(
                "selected route requires a valid Preplan: " + "; ".join(violations)
            )
        run_path = self.workspace / "plans" / "preplan_run.json"
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            document = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read validated Preplan state: {exc}") from exc
        if run.get("status") != "PASS":
            raise RuntimeError("selected route requires PREPLAN PASS")
        digest = _sha256(artifact)
        if run.get("artifact_sha256") != digest:
            raise RuntimeError(
                "Preplan artifact digest no longer matches preplan_run.json"
            )
        if run.get("source_revision") != manifest.revision:
            raise RuntimeError(
                "Preplan source revision does not match the repository manifest"
            )
        preplan_head = str(run.get("incumbent_commit") or "")
        if not preplan_head:
            raise RuntimeError("preplan_run.json has no incumbent commit")
        ancestor = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                preplan_head,
                git_head(self.workspace),
            ],
            cwd=str(self.workspace),
            check=False,
            capture_output=True,
        )
        if ancestor.returncode:
            raise RuntimeError(
                "Preplan incumbent is not an ancestor of the campaign HEAD"
            )
        routes = document.get("architecture_frontier", [])
        selected = next(
            (
                item
                for item in routes
                if isinstance(item, dict) and item.get("id") == route_id
            ),
            None,
        )
        if selected is None:
            available = ", ".join(
                str(item.get("id")) for item in routes if isinstance(item, dict)
            )
            raise RuntimeError(
                f"Preplan route {route_id!r} does not exist; available: {available}"
            )
        return document, run, selected

    def configure(
        self,
        *,
        route_id: str,
        manifest: RepositoryManifest,
        total_episode_target: int,
    ) -> FixedRouteState:
        if total_episode_target <= 0:
            raise RuntimeError("fixed-route total episode target must be positive")
        document, run, selected = self._validated_inputs(route_id, manifest)
        artifact = self.workspace / PREPLAN_ARTIFACT
        expected = {
            "route_id": route_id,
            "artifact_sha256": _sha256(artifact),
            "artifact_revision": int(document["revision"]),
            "source_revision": manifest.revision,
            "preplan_incumbent_commit": str(run["incumbent_commit"]),
            "total_episode_target": total_episode_target,
        }
        if self.state_path.is_file():
            state = self.load()
            for key, value in expected.items():
                if getattr(state, key) != value:
                    raise RuntimeError(
                        f"fixed Preplan route cannot change {key}: "
                        f"recorded={getattr(state, key)!r} requested={value!r}"
                    )
            self._verify_bundle()
            return state

        self.root.mkdir(parents=True, exist_ok=True)
        if self.bundle_path.exists():
            raise RuntimeError("fixed Preplan route bundle exists without state")
        self.bundle_path.mkdir()
        shutil.copy2(artifact, self.bundle_path / "architecture_frontier.json")
        shutil.copy2(
            self.workspace / "plans" / "preplan_run.json",
            self.bundle_path / "preplan_run.json",
        )
        atomic_write_json(self.bundle_path / "selected_route.json", selected)
        self._copy_preplan_evidence(run)
        files = {
            str(path.relative_to(self.bundle_path)): _sha256(path)
            for path in sorted(self.bundle_path.rglob("*"))
            if path.is_file() and not path.is_symlink()
        }
        atomic_write_json(
            self.bundle_path / "manifest.json", {"schema_version": 1, "files": files}
        )
        state = FixedRouteState(
            **expected,
            selected_at=_utc_now(),
            history=[
                {
                    "event": "fixed_preplan_route_selected",
                    "route_id": route_id,
                    "timestamp": _utc_now(),
                }
            ],
        )
        self.save(state)
        self._verify_bundle()
        return state

    def _copy_preplan_evidence(self, run: dict[str, Any]) -> None:
        destination = self.bundle_path / "profiles" / "preplan"
        copied = 0
        total = 0
        forensic = Path(str(run.get("worktree") or ""))
        source_root = forensic / "profiles" / "preplan"
        if source_root.is_dir():
            for source in sorted(source_root.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(source_root)
                size = source.stat().st_size
                copied += 1
                total += size
                if copied > MAX_PROFILE_FILES or total > MAX_PROFILE_BYTES:
                    raise RuntimeError(
                        "Preplan evidence exceeds validated publication bounds"
                    )
                try:
                    folded = source.read_text(encoding="utf-8").casefold()
                except (OSError, UnicodeError):
                    folded = ""
                if any(marker in folded for marker in FORBIDDEN_EVIDENCE_MARKERS):
                    raise RuntimeError(
                        f"forbidden private marker in Preplan evidence: {relative}"
                    )
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        for evidence in run.get("probe_evidence", []):
            if not isinstance(evidence, dict):
                continue
            relative = _safe_relative(str(evidence.get("path") or ""))
            source = self.workspace / relative
            if not source.is_file() or _sha256(source) != evidence.get("sha256"):
                raise RuntimeError(
                    f"canonical Preplan evidence is missing or changed: {relative}"
                )
            target = self.bundle_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _verify_bundle(self) -> None:
        try:
            manifest = json.loads(
                (self.bundle_path / "manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"fixed Preplan route bundle is invalid: {exc}") from exc
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise RuntimeError("fixed Preplan route bundle manifest has no files")
        actual: set[str] = set()
        for path in self.bundle_path.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(
                    "fixed Preplan route bundle contains a symlink: "
                    + str(path.relative_to(self.bundle_path))
                )
            if path.is_file() and path.name != "manifest.json":
                actual.add(str(path.relative_to(self.bundle_path)))
        if actual != set(files):
            raise RuntimeError("fixed Preplan route bundle file set changed")
        for value, digest in files.items():
            relative = _safe_relative(str(value))
            path = self.bundle_path / relative
            if not path.is_file() or path.is_symlink() or _sha256(path) != digest:
                raise RuntimeError(f"fixed Preplan route bundle changed: {relative}")

    def stage_episode(self, episode_workspace: Path) -> bool:
        state = self.load()
        self._verify_bundle()
        target = episode_workspace / EPISODE_BUNDLE
        if target.exists():
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.bundle_path, target)
        atomic_write_json(target / STATE_FILE, asdict(state))
        return self.apply_wip(episode_workspace, state)

    def apply_wip(self, episode_workspace: Path, state: FixedRouteState) -> bool:
        if not self.wip_patch_path.is_file():
            return False
        patch = self.wip_patch_path.read_bytes()
        if (
            not state.wip_patch_sha256
            or hashlib.sha256(patch).hexdigest() != state.wip_patch_sha256
        ):
            raise RuntimeError("fixed-route WIP patch checksum mismatch")
        if git_head(episode_workspace) != state.wip_base_commit:
            raise RuntimeError("fixed-route WIP base does not match the incumbent")
        completed = subprocess.run(
            ["git", "apply", "--3way", str(self.wip_patch_path)],
            cwd=str(episode_workspace),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "cannot restore fixed-route WIP: "
                + (completed.stderr.strip() or completed.stdout.strip())[-1200:]
            )
        return True

    def record_episode(
        self,
        *,
        worktree: Path,
        base_commit: str,
        checkpoint: str,
        accepted: bool,
        disposition: str,
        episode: int,
    ) -> None:
        state = self.load()
        if accepted or disposition == "implementation_refuted":
            self.wip_patch_path.unlink(missing_ok=True)
            state.wip_base_commit = ""
            state.wip_source_commit = ""
            state.wip_patch_sha256 = ""
        elif checkpoint:
            completed = subprocess.run(
                ["git", "diff", "--binary", base_commit, checkpoint, "--"],
                cwd=str(worktree),
                capture_output=True,
                check=True,
            )
            temporary = self.wip_patch_path.with_suffix(".patch.tmp")
            temporary.write_bytes(completed.stdout)
            temporary.replace(self.wip_patch_path)
            state.wip_base_commit = git_head(self.workspace)
            state.wip_source_commit = checkpoint
            state.wip_patch_sha256 = hashlib.sha256(completed.stdout).hexdigest()
        state.history.append(
            {
                "event": "fixed_route_episode_recorded",
                "episode": episode,
                "accepted": accepted,
                "disposition": disposition,
                "checkpoint": checkpoint or None,
                "timestamp": _utc_now(),
            }
        )
        self.save(state)

    def reanchor_after_recovery(
        self,
        *,
        episode: int,
        base_commit: str,
        outcome_commit: str,
        editable_roots: tuple[str, ...],
    ) -> None:
        state = self.load()
        if not self.wip_patch_path.is_file() or state.wip_base_commit == outcome_commit:
            return
        if state.wip_base_commit != base_commit:
            raise RuntimeError("cannot re-anchor fixed-route WIP: base commit mismatch")
        unchanged = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                base_commit,
                outcome_commit,
                "--",
                *editable_roots,
            ],
            cwd=str(self.workspace),
            check=False,
        )
        if unchanged.returncode:
            raise RuntimeError(
                "cannot re-anchor fixed-route WIP after editable source changed"
            )
        state.wip_base_commit = outcome_commit
        state.history.append(
            {
                "event": "fixed_route_wip_reanchored_after_recovery",
                "episode": episode,
                "old_base_commit": base_commit,
                "new_base_commit": outcome_commit,
                "timestamp": _utc_now(),
            }
        )
        self.save(state)

    def selected_route(self) -> dict[str, Any]:
        return json.loads(
            (self.bundle_path / "selected_route.json").read_text(encoding="utf-8")
        )


def validate_fixed_route_outcome(
    outcome: object,
    *,
    route_id: str,
    terminal_status: str,
    has_last_trial_commit: bool,
) -> str:
    if not isinstance(outcome, dict):
        return "fixed-route episode requires an outcome object"
    route = outcome.get("preplan_route")
    if not isinstance(route, dict):
        return "fixed-route outcome requires preplan_route"
    if route.get("route_id") != route_id:
        return f"preplan_route.route_id must remain {route_id}"
    disposition = route.get("disposition")
    if disposition not in {"continue", "implementation_refuted", "candidate"}:
        return "preplan_route.disposition must be continue, implementation_refuted, or candidate"
    if not str(route.get("implementation_variant") or "").strip():
        return "preplan_route requires implementation_variant"
    mechanisms = route.get("preserved_mechanisms")
    if (
        not isinstance(mechanisms, list)
        or not mechanisms
        or any(not isinstance(item, str) or not item.strip() for item in mechanisms)
    ):
        return "preplan_route requires non-empty preserved_mechanisms"
    options = route.get("next_implementation_options")
    if not isinstance(options, list) or any(
        not isinstance(item, str) for item in options
    ):
        return "preplan_route.next_implementation_options must be a string list"
    if terminal_status == "candidate_ready" and disposition != "candidate":
        return "candidate_ready fixed-route outcome requires disposition=candidate"
    if terminal_status != "candidate_ready" and disposition == "candidate":
        return "non-candidate fixed-route outcome cannot use disposition=candidate"
    if disposition == "continue" and not has_last_trial_commit:
        return "continuing fixed-route work requires last_trial_commit"
    return ""


def render_fixed_route_directive(
    state: FixedRouteState, route: dict[str, Any], *, wip_applied: bool
) -> str:
    thesis = str(route.get("thesis") or "")
    return f"""## Fixed Preplan architecture route (binding)

The campaign has selected `{state.route_id}` from a validated Preplan frontier for all
{state.total_episode_target} formal episodes. Read the immutable bundle under
`.repository_horizon_runtime/fixed_preplan_route/`, especially `selected_route.json`,
`architecture_frontier.json`, and the captured `profiles/preplan/` probes. The route thesis is:

> {thesis}

Do not switch to another frontier route, including one that was ranked faster by the Preplan. This
is a deep optimization campaign for the selected architectural family, not a reranking exercise.
Preserve its defining representation/compute bridge, while treating the recorded implementation
graph and prototypes only as evidence. They are not a prescription for a naive materialized gather.
Within the selected route you may move graph cuts, fuse or make conversion lazy/on-chip, eliminate
intermediate materialization and launches, change layouts, schedules, pipelines, dispatch, memory
movement, and add missing compute mechanisms such as grouped-query reuse. Measure the complete
end-to-end path: preprocessing, allocation, conversion, compute, launches, and postprocessing all
count. Never hide route cost outside the measured operator.

An implementation failure does not authorize changing route. Try a materially different
implementation of the same thesis. For useful non-promotable work, commit the checkpoint and return
it as `last_trial_commit`; the supervisor will restore it in the next isolated episode. WIP restored
for this episode: {str(wip_applied).lower()}.

Every terminal journal outcome must include:

```json
"preplan_route": {{
  "route_id": "{state.route_id}",
  "disposition": "continue | implementation_refuted | candidate",
  "implementation_variant": "specific implementation attempted in this episode",
  "preserved_mechanisms": ["selected-route mechanism retained"],
  "next_implementation_options": ["different implementation within the same route"]
}}
```

Use `candidate` only with `candidate_ready`. Use `continue` with `pivot` or `blocked` only when a
clean committed `last_trial_commit` preserves useful work. `implementation_refuted` rejects only the
current implementation variant; the next episode must remain on `{state.route_id}`.
"""
