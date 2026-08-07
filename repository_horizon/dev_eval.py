from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from long_horizon.git_episode import changed_paths, git_head

from .evaluation import (
    evaluation_handoff_path,
    load_pending,
    write_evaluation_handoff,
)
from .manifest import load_manifest
from .transport import get_agate_job
from .verifier import RepositoryABBAValidator, collect_pending_verification


def _clean_checkpoint(parser: argparse.ArgumentParser, workspace: Path) -> None:
    if (
        subprocess.run(
            ["git", "diff", "--quiet"], cwd=str(workspace), check=False
        ).returncode
        != 0
        or subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(workspace),
            check=False,
        ).returncode
        != 0
    ):
        parser.error("development evaluation requires a clean checkpoint commit")


def _verifier(workspace: Path, args: argparse.Namespace) -> tuple[RepositoryABBAValidator, str, str, list[str]]:
    journal = json.loads(
        (workspace / ".atrex_long_horizon" / "journal.json").read_text(
            encoding="utf-8"
        )
    )
    base_commit = str(journal["base_commit"])
    candidate_commit = git_head(workspace)
    paths = changed_paths(workspace, base_commit, candidate_commit)
    manifest = load_manifest(workspace / "source_manifest.json")
    lock = json.loads((workspace / "source.lock.json").read_text(encoding="utf-8"))
    bringup = not (workspace / "memory" / "v0.json").is_file()
    repeats = manifest.bringup.probe_repeats if bringup else 1
    schedule_runs = repeats if bringup else repeats * 2
    per_run_timeout = max(1, (600 - 30) // schedule_runs)
    verifier = RepositoryABBAValidator(
        manifest=manifest,
        atrex_bench_root=Path(lock["atrex_bench_root"]),
        hardware=args.hardware,
        profile=args.profile,
        url=args.url,
        timeout=600,
        repeats=repeats,
        per_run_timeout=per_run_timeout,
        min_improvement_pct=(
            -100.0 if bringup else manifest.measurement.min_improvement_pct
        ),
        candidate_only=bringup,
    )
    return verifier, base_commit, candidate_commit, paths


def _submit(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    _clean_checkpoint(parser, workspace)
    handoff_path = evaluation_handoff_path(workspace)
    if handoff_path.exists():
        value = json.loads(handoff_path.read_text(encoding="utf-8"))
        pending_path = Path(str(value.get("pending_path", "")))
        if pending_path.is_file() and not pending_path.with_name("result.json").is_file():
            print(json.dumps(value, indent=2), flush=True)
            return 0
        parser.error(
            "stale evaluation_handoff.json exists; the repository supervisor must consume it"
        )
    verifier, base_commit, candidate_commit, paths = _verifier(workspace, args)
    pending = verifier.submit(
        workspace,
        base_commit=base_commit,
        candidate_commit=candidate_commit,
        changed_paths=paths,
    )
    handoff = write_evaluation_handoff(workspace, pending)
    print(
        json.dumps(
            {
                "status": "submitted",
                "evaluation_id": pending.evaluation_id,
                "job_id": pending.job_id,
                "candidate_commit": candidate_commit,
                "pending_path": str(pending.pending_path),
                "evaluation_handoff": str(handoff),
                "next_action": "end this Agent invocation immediately; the supervisor will resume it",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def _status(args: argparse.Namespace) -> int:
    pending = load_pending(Path(args.pending).resolve())
    snapshot = get_agate_job(
        pending.job_id,
        profile=pending.profile,
        url=pending.url,
    )
    print(json.dumps(snapshot.response, indent=2), flush=True)
    return 0


def _collect(args: argparse.Namespace) -> int:
    result = collect_pending_verification(Path(args.pending).resolve())
    print(json.dumps(result.as_dict(), indent=2), flush=True)
    return 0 if result.gate in {"PASS", "FAIL"} else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit and collect repository candidate ABBA jobs on Agate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="submit one job and return immediately")
    submit.add_argument("--workspace", required=True)
    submit.add_argument("--hardware", required=True)
    submit.add_argument("--profile", choices=("pre", "prod"), default="")
    submit.add_argument("--url", default="")

    status = subparsers.add_parser("status", help="perform one non-waiting status request")
    status.add_argument("--pending", required=True)

    collect = subparsers.add_parser("collect", help="wait and persist one terminal result")
    collect.add_argument("--pending", required=True)

    args = parser.parse_args(argv)
    if args.command == "submit":
        return _submit(parser, args)
    if args.command == "status":
        return _status(args)
    return _collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
