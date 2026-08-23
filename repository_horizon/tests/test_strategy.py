from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from repository_horizon.strategy import (
    ARCHITECTURE_MAP_WORKTREE_PATH,
    ArchitectureStrategyState,
    RepositoryStrategyStore,
    render_strategy_directive,
    validate_architecture_map,
    validate_architecture_outcome,
)


def _run(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=str(repo), text=True, capture_output=True, check=True
    )
    return completed.stdout.strip()


def _architecture_map() -> dict:
    return {
        "schema_version": 1,
        "selected_direction_id": "pair-cta",
        "directions": [
            {
                "id": value,
                "thesis": f"thesis {value}",
                "required_mechanisms": ["mechanism"],
                "evidence_for": ["source"],
                "evidence_against": [],
                "falsification_test": "equivalent implementation test",
                "status": "open",
            }
            for value in ("pair-cta", "tile-family", "scheduler-family")
        ],
    }


class RepositoryStrategyTests(unittest.TestCase):
    def test_stall_enters_architecture_mode_and_does_not_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = RepositoryStrategyStore(Path(temp))
            state = store.enter_if_needed(
                episode=6,
                consecutive_without_promotion=5,
                escape_after=5,
                review_interval=20,
                commitment_episodes=3,
            )
            self.assertEqual(state.mode, "architecture")
            self.assertEqual(state.commitment_remaining, 3)
            self.assertEqual(state.history[-1]["reason"], "promotion_stall")
            self.assertEqual(store.load().cycle, 1)

    def test_periodic_review_survives_ordinary_promotions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = RepositoryStrategyStore(Path(temp))
            state = store.enter_if_needed(
                episode=8,
                consecutive_without_promotion=0,
                escape_after=5,
                review_interval=8,
                commitment_episodes=3,
            )
            self.assertEqual(state.mode, "architecture")
            self.assertEqual(state.history[-1]["reason"], "periodic_review")

    def test_architecture_map_requires_a_portfolio(self) -> None:
        self.assertEqual(validate_architecture_map(_architecture_map()), "")
        invalid = _architecture_map()
        invalid["directions"] = invalid["directions"][:2]
        self.assertIn("at least three", validate_architecture_map(invalid))

    def test_proxy_failure_cannot_refute_an_architecture(self) -> None:
        outcome = {
            "architecture": {
                "direction_id": "pair-cta",
                "thesis": "pair execution raises the compute ceiling",
                "disposition": "architecture_refuted",
                "feature_parity_complete": False,
                "tested_implementation_variants": 1,
                "next_implementation_options": [],
                "independent_review": {"status": "completed"},
            }
        }
        self.assertIn(
            "feature_parity_complete",
            validate_architecture_outcome(
                outcome, terminal_status="pivot", has_last_trial_commit=False
            ),
        )
        outcome["architecture"]["feature_parity_complete"] = True
        outcome["architecture"]["tested_implementation_variants"] = 2
        self.assertEqual(
            validate_architecture_outcome(
                outcome, terminal_status="pivot", has_last_trial_commit=False
            ),
            "",
        )

    def test_continuing_work_requires_a_checkpoint(self) -> None:
        outcome = {
            "architecture": {
                "direction_id": "pair-cta",
                "thesis": "pair execution raises the compute ceiling",
                "disposition": "continue",
                "next_implementation_options": ["loader A", "loader B"],
            }
        }
        self.assertIn(
            "last_trial_commit",
            validate_architecture_outcome(
                outcome, terminal_status="pivot", has_last_trial_commit=False
            ),
        )
        self.assertEqual(
            validate_architecture_outcome(
                outcome, terminal_status="pivot", has_last_trial_commit=True
            ),
            "",
        )

    def test_wip_patch_is_restored_on_a_fresh_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            _run(source, "init")
            _run(source, "config", "user.name", "test")
            _run(source, "config", "user.email", "test@example.com")
            tracked = source / "kernel.py"
            tracked.write_text("VALUE = 1\n", encoding="utf-8")
            _run(source, "add", "kernel.py")
            _run(source, "commit", "-m", "base")
            base = _run(source, "rev-parse", "HEAD")
            tracked.write_text("VALUE = 2\n", encoding="utf-8")
            _run(source, "add", "kernel.py")
            _run(source, "commit", "-m", "wip")
            wip = _run(source, "rev-parse", "HEAD")

            campaign = root / "campaign"
            subprocess.run(
                ["git", "clone", "--quiet", str(source), str(campaign)], check=True
            )
            _run(campaign, "checkout", "--detach", base)
            store = RepositoryStrategyStore(campaign)
            patch = subprocess.run(
                ["git", "diff", "--binary", base, wip, "--"],
                cwd=str(source),
                capture_output=True,
                check=True,
            ).stdout
            store.wip_patch_path.write_bytes(patch)
            restored = store.apply_wip(
                campaign, ArchitectureStrategyState(mode="architecture")
            )
            self.assertTrue(restored)
            self.assertEqual((campaign / "kernel.py").read_text(), "VALUE = 2\n")

    def test_escape_prompt_encodes_belief_and_flexibility(self) -> None:
        state = ArchitectureStrategyState(
            mode="architecture",
            cycle=2,
            commitment_remaining=2,
            active_direction_id="pair-cta",
        )
        directive = render_strategy_directive(
            state,
            escape_after=5,
            review_interval=8,
            commitment_episodes=3,
            wip_applied=True,
        )
        self.assertIn("temporarily slower", directive)
        self.assertIn("thesis, not to one patch", directive)
        self.assertIn("out-of-domain proxy", directive)
        self.assertIn(str(ARCHITECTURE_MAP_WORKTREE_PATH), directive)


if __name__ == "__main__":
    unittest.main()
