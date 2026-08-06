from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from long_horizon.git_episode import changed_paths, git_head

from .manifest import load_manifest
from .verifier import RepositoryABBAValidator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run repository candidate development ABBA on Agate"
    )
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--profile", choices=("pre", "prod"), default="")
    parser.add_argument("--url", default="")
    args = parser.parse_args(argv)
    workspace = Path(args.workspace).resolve()
    if (
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=str(workspace), check=False
        ).returncode
        != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(workspace), check=False
        ).returncode
        != 0
    ):
        parser.error("development evaluation requires a clean checkpoint commit")
    journal = json.loads(
        (workspace / ".atrex_long_horizon" / "journal.json").read_text(encoding="utf-8")
    )
    base_commit = str(journal["base_commit"])
    candidate_commit = git_head(workspace)
    paths = changed_paths(workspace, base_commit, candidate_commit)
    manifest = load_manifest(workspace / "source_manifest.json")
    lock = json.loads((workspace / "source.lock.json").read_text(encoding="utf-8"))
    verifier = RepositoryABBAValidator(
        manifest=manifest,
        atrex_bench_root=Path(lock["atrex_bench_root"]),
        hardware=args.hardware,
        profile=args.profile,
        url=args.url,
        timeout=600,
        repeats=1,
        min_improvement_pct=-100.0,
    )
    result = verifier.verify(
        workspace,
        base_commit=base_commit,
        candidate_commit=candidate_commit,
        changed_paths=paths,
    )
    print(json.dumps(result.as_dict(), indent=2), flush=True)
    return 0 if result.status in {"PASS", "FAIL"} and result.candidate_latency_us else 1


if __name__ == "__main__":
    raise SystemExit(main())
