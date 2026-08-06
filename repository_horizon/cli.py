from __future__ import annotations

import argparse
import sys
from pathlib import Path

from long_horizon import cli as long_cli

from .integration import RepositoryIntegration
from .manifest import load_manifest


def _repository_parser(*, add_help: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description="Repository source-assisted Long Horizon overlay",
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-checkout", required=True)
    return parser


def _extract(argv: list[str]):
    parser = _repository_parser(add_help=False)
    values, remaining = parser.parse_known_args(argv)
    return values, remaining


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "-h" in raw or "--help" in raw:
        _repository_parser(add_help=True).print_help()
        print("\nInherited optimize.py and Long Horizon options:")
        return long_cli.main(["--help"])
    values, remaining = _extract(raw)
    if "--layer" in remaining:
        raise SystemExit("repository horizon v1 does not support --layer")
    if "--no-workload-bucketing" not in remaining:
        raise SystemExit("repository horizon v1 requires --no-workload-bucketing")
    try:
        framework = remaining[remaining.index("--framework") + 1]
    except (ValueError, IndexError):
        raise SystemExit("repository horizon v1 requires explicit --framework CuteDSL")
    if framework != "CuteDSL":
        raise SystemExit("repository horizon v1 requires explicit --framework CuteDSL")
    if "--framework-baseline" not in remaining:
        remaining += ["--framework-baseline", "never"]
    manifest = load_manifest(values.source_manifest)
    source_checkout = Path(values.source_checkout).resolve()
    if not (source_checkout / ".git").exists():
        # Worktrees use a .git file, normal clones use a directory.
        marker = source_checkout / ".git"
        if not marker.is_file():
            raise SystemExit(
                f"source checkout is not a Git worktree: {source_checkout}"
            )

    def factory(campaign, options):
        return RepositoryIntegration(manifest, source_checkout)

    return long_cli.main(remaining, integration_factory=factory)


if __name__ == "__main__":
    raise SystemExit(main())
