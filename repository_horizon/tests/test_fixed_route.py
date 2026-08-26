from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from repository_horizon.cli import build_parser
from repository_horizon.fixed_route import (
    EPISODE_BUNDLE,
    FixedPreplanRouteStore,
    render_fixed_route_directive,
    validate_fixed_route_outcome,
)
from repository_horizon.manifest import load_manifest
from repository_horizon.tests.helpers import init_repo, run_git
from repository_horizon.tests.test_preplan import valid_document
from long_horizon.git_episode import EpisodeWorktree, git_head

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "recipes" / "fa4_fp8_paged_sm100.example.json"


class FixedPreplanRouteTests(unittest.TestCase):
    def _workspace(self, root: Path) -> tuple[Path, object]:
        workspace = root / "workspace"
        head = init_repo(workspace)
        manifest = load_manifest(MANIFEST)
        plans = workspace / "plans"
        plans.mkdir()
        document = valid_document()
        artifact = plans / "end_to_end_architecture_frontier.json"
        artifact.write_text(json.dumps(document), encoding="utf-8")
        forensic = root / "forensic"
        evidence = forensic / "profiles" / "preplan"
        evidence.mkdir(parents=True)
        (evidence / "route_probe.py").write_text("# bounded prototype\n")
        raw = workspace / "profiles" / "preplan" / "raw"
        raw.mkdir(parents=True)
        (raw / "probe.txt").write_text("public evidence\n")
        raw_digest = hashlib.sha256((raw / "probe.txt").read_bytes()).hexdigest()
        (plans / "preplan_run.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "source_revision": manifest.revision,
                    "incumbent_commit": head,
                    "artifact_sha256": hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest(),
                    "worktree": str(forensic),
                    "probe_evidence": [
                        {
                            "path": "profiles/preplan/raw/probe.txt",
                            "sha256": raw_digest,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return workspace, manifest

    def test_cli_has_exact_episode_and_fixed_route_controls(self) -> None:
        options = {action.dest for action in build_parser()._actions}
        self.assertIn("episode_count", options)
        self.assertIn("preplan_route_id", options)

    def test_selection_is_frozen_and_staged_with_prototype(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace, manifest = self._workspace(Path(temp))
            store = FixedPreplanRouteStore(workspace)
            state = store.configure(
                route_id="route-b", manifest=manifest, total_episode_target=100
            )
            self.assertEqual(state.route_id, "route-b")
            self.assertEqual(state.total_episode_target, 100)
            self.assertTrue(
                (store.bundle_path / "profiles/preplan/route_probe.py").is_file()
            )
            episode = Path(temp) / "episode"
            episode.mkdir()
            self.assertFalse(store.stage_episode(episode))
            selected = json.loads(
                (episode / EPISODE_BUNDLE / "selected_route.json").read_text()
            )
            self.assertEqual(selected["id"], "route-b")
            with self.assertRaisesRegex(RuntimeError, "cannot change route_id"):
                store.configure(
                    route_id="route-a", manifest=manifest, total_episode_target=100
                )
            (store.bundle_path / "injected.txt").write_text("unexpected\n")
            with self.assertRaisesRegex(RuntimeError, "file set changed"):
                store.stage_episode(episode)

    def test_route_directive_frees_implementation_but_forbids_route_switch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace, manifest = self._workspace(Path(temp))
            store = FixedPreplanRouteStore(workspace)
            state = store.configure(
                route_id="route-b", manifest=manifest, total_episode_target=100
            )
            directive = render_fixed_route_directive(
                state, store.selected_route(), wip_applied=False
            )
            self.assertIn("Do not switch to another frontier route", directive)
            self.assertIn(
                "not a prescription for a naive materialized gather", directive
            )
            self.assertIn("fuse or make conversion lazy/on-chip", directive)

    def test_terminal_outcome_must_keep_selected_route(self) -> None:
        outcome = {
            "preplan_route": {
                "route_id": "route-b",
                "disposition": "continue",
                "implementation_variant": "fused bridge",
                "preserved_mechanisms": ["representation bridge"],
                "next_implementation_options": ["lazy bridge"],
            }
        }
        self.assertEqual(
            validate_fixed_route_outcome(
                outcome,
                route_id="route-b",
                terminal_status="pivot",
                has_last_trial_commit=True,
            ),
            "",
        )
        self.assertIn(
            "must remain route-a",
            validate_fixed_route_outcome(
                outcome,
                route_id="route-a",
                terminal_status="pivot",
                has_last_trial_commit=True,
            ),
        )
        self.assertIn(
            "requires last_trial_commit",
            validate_fixed_route_outcome(
                outcome,
                route_id="route-b",
                terminal_status="pivot",
                has_last_trial_commit=False,
            ),
        )

    def test_nonpromoted_checkpoint_is_restored_in_next_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace, manifest = self._workspace(root)
            editable = workspace / manifest.editable_workspace_roots[0] / "route.py"
            editable.parent.mkdir(parents=True)
            editable.write_text("VALUE = 1\n", encoding="utf-8")
            run_git(workspace, "add", str(editable.relative_to(workspace)))
            run_git(workspace, "commit", "-m", "add editable source")
            base = git_head(workspace)
            store = FixedPreplanRouteStore(workspace)
            store.configure(
                route_id="route-b", manifest=manifest, total_episode_target=100
            )

            trial = EpisodeWorktree.create(
                workspace, 1, base, root=root / "episode-worktrees"
            )
            trial_file = trial.path / editable.relative_to(workspace)
            trial_file.write_text("VALUE = 2\n", encoding="utf-8")
            run_git(trial.path, "add", str(editable.relative_to(workspace)))
            run_git(trial.path, "commit", "-m", "fixed-route checkpoint")
            checkpoint = git_head(trial.path)
            store.record_episode(
                worktree=trial.path,
                base_commit=base,
                checkpoint=checkpoint,
                accepted=False,
                disposition="continue",
                episode=1,
            )

            next_trial = EpisodeWorktree.create(
                workspace, 2, base, root=root / "episode-worktrees"
            )
            self.assertTrue(store.stage_episode(next_trial.path))
            self.assertEqual(
                (next_trial.path / editable.relative_to(workspace)).read_text(),
                "VALUE = 2\n",
            )
            next_trial.remove(workspace)
            trial.remove(workspace)


if __name__ == "__main__":
    unittest.main()
