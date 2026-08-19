from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from long_horizon.git_episode import git_head

from .evaluation import (
    append_evaluation_experiment,
    load_pending,
)
from .manifest import load_manifest
from .profile_eval import collect_profile, submit_profile
from .reconnaissance import reconnaissance_gate_violations
from .transport import get_agate_job, get_local_job
from .verifier import RepositoryABBAValidator, collect_pending_verification


def _lines(workspace: Path, *args: str) -> list[str]:
    process = subprocess.run(
        ["git", *args], cwd=str(workspace), check=True, capture_output=True, text=True
    )
    return [value for value in process.stdout.splitlines() if value]


def _working_changed_paths(workspace: Path, base_commit: str) -> list[str]:
    tracked = _lines(workspace, "diff", "--name-only", base_commit, "--")
    untracked = _lines(workspace, "ls-files", "--others", "--exclude-standard")
    return sorted(set(tracked + untracked))


def _require_reconnaissance(
    parser: argparse.ArgumentParser, workspace: Path
) -> None:
    manifest = load_manifest(workspace / "source_manifest.json")
    violations = reconnaissance_gate_violations(workspace, manifest)
    if violations:
        parser.error("repository reconnaissance gate: " + "; ".join(violations))


def _verifier(
    workspace: Path, args: argparse.Namespace
) -> tuple[RepositoryABBAValidator, str, str, list[str]]:
    journal = json.loads(
        (workspace / ".atrex_long_horizon" / "journal.json").read_text(encoding="utf-8")
    )
    base_commit = str(journal["base_commit"])
    candidate_commit = git_head(workspace)
    paths = _working_changed_paths(workspace, base_commit)
    manifest = load_manifest(workspace / "source_manifest.json")
    lock = json.loads((workspace / "source.lock.json").read_text(encoding="utf-8"))
    from orchestrator.workspace_state import latest_version, read_memory

    version = latest_version(workspace)
    memory = read_memory(workspace, version) if version >= 0 else None
    bringup = not (
        isinstance(memory, dict)
        and (memory.get("quality_gate") or {}).get("result") == "PASS"
    )
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
        backend=args.backend,
        wait_mode=args.wait_mode,
        wait_timeout=args.wait_timeout,
        agent_result_max_bytes=args.agent_result_max_bytes,
        private_reference_dir=(
            Path(os.environ["ATREX_PRIVATE_REFERENCE_DIR"]).resolve()
            if os.environ.get("ATREX_PRIVATE_REFERENCE_DIR")
            else None
        ),
    )
    return verifier, base_commit, candidate_commit, paths


def _submit(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    _require_reconnaissance(parser, workspace)
    verifier, base_commit, candidate_commit, paths = _verifier(workspace, args)
    pending = verifier.submit(
        workspace,
        base_commit=base_commit,
        candidate_commit=candidate_commit,
        changed_paths=paths,
        working_snapshot=True,
    )
    result = collect_pending_verification(pending)
    append_evaluation_experiment(workspace, pending, result)
    card = json.loads(
        (pending.directory / "agent_result.json").read_text(encoding="utf-8")
    )
    print(json.dumps(card, indent=2), flush=True)
    return 0


def _status(args: argparse.Namespace) -> int:
    pending = load_pending(Path(args.pending).resolve())
    snapshot = (
        get_local_job(Path(pending.stage), pending.job_id)
        if pending.backend == "local"
        else get_agate_job(
            pending.job_id,
            profile=pending.profile,
            url=pending.url,
        )
    )
    print(json.dumps(snapshot.response, indent=2), flush=True)
    return 0


def _collect(args: argparse.Namespace) -> int:
    pending = load_pending(Path(args.pending).resolve())
    result = (
        collect_profile(pending)
        if pending.kind == "profile"
        else collect_pending_verification(pending)
    )
    print(json.dumps(result.as_dict(), indent=2), flush=True)
    return 0 if result.gate in {"PASS", "FAIL"} else 1


def _profile(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    _require_reconnaissance(parser, workspace)
    candidate = (workspace / args.candidate).resolve()
    reference_dir = (workspace / args.reference_dir).resolve()
    for path, label in ((candidate, "candidate"), (reference_dir, "reference-dir")):
        try:
            path.relative_to(workspace)
        except ValueError:
            parser.error(f"{label} escaped the workspace: {path}")
    if not candidate.is_file() or not reference_dir.is_dir():
        parser.error("profile candidate/reference-dir is missing")
    pending = submit_profile(
        workspace,
        candidate=candidate,
        reference_dir=reference_dir,
        hardware=args.hardware,
        profile=args.profile,
        url=args.url,
        level=args.level,
        wait_mode=args.wait_mode,
        wait_timeout=args.wait_timeout,
        job_timeout=args.job_timeout,
        kernel_name=args.kernel_name,
        kernel_regex=args.kernel_regex,
        source=args.source,
        launch_skip=args.launch_skip,
        launch_count=args.launch_count,
        agent_result_max_bytes=args.agent_result_max_bytes,
        backend=args.backend,
        route=args.route,
    )
    result = collect_profile(pending)
    append_evaluation_experiment(workspace, pending, result)
    print((pending.directory / "agent_result.json").read_text(encoding="utf-8"), end="")
    return 0


def _show(args: argparse.Namespace) -> int:
    directory = Path(args.evaluation).resolve()
    if directory.is_file():
        directory = directory.parent
    card_path = directory / "agent_result.json"
    if args.section == "summary":
        value = json.loads(card_path.read_text(encoding="utf-8"))
    else:
        transport = json.loads(
            (directory / "transport_result.json").read_text(encoding="utf-8")
        )
        if isinstance(transport.get("profile"), dict):
            transport = transport["profile"]
        aliases = {
            "hotspots": ("hotspots", "top_kernels", "bottlenecks"),
            "source-lines": ("source_lines", "source_evidence", "localize"),
            "sass": ("sass", "disassembly"),
            "artifacts": ("artifacts",),
        }
        value = {
            key: transport.get(key)
            for key in aliases[args.section]
            if transport.get(key) is not None
        }
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    encoded = rendered.encode("utf-8")
    if len(encoded) > args.max_bytes:
        rendered = encoded[: args.max_bytes].decode("utf-8", errors="ignore")
        rendered += "\n... [truncated; inspect the persisted artifact directly]\n"
    print(rendered)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Submit and collect repository candidate ABBA jobs on Agate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser(
        "submit", help="submit one job and return immediately"
    )
    submit.add_argument("--workspace", required=True)
    submit.add_argument("--hardware", required=True)
    submit.add_argument("--profile", choices=("pre", "prod"), default="")
    submit.add_argument("--url", default="")
    submit.add_argument("--backend", choices=("agate", "local"), default="agate")
    submit.add_argument("--wait-mode", choices=("inline",), default="inline")
    submit.add_argument("--wait-timeout", type=int, default=14_400)
    submit.add_argument("--agent-result-max-bytes", type=int, default=16 * 1024)
    submit.add_argument(
        "--agent-cli", choices=("claude", "qodercli", "codex", "pi"), default="claude"
    )

    status = subparsers.add_parser(
        "status", help="perform one non-waiting status request"
    )
    status.add_argument("--pending", required=True)

    collect = subparsers.add_parser(
        "collect", help="wait and persist one terminal result"
    )
    collect.add_argument("--pending", required=True)

    profile = subparsers.add_parser(
        "profile", help="optional typed NCU/rocprof profile"
    )
    profile.add_argument("--workspace", required=True)
    profile.add_argument("--hardware", required=True)
    profile.add_argument("--profile", choices=("pre", "prod"), default="")
    profile.add_argument("--url", default="")
    profile.add_argument("--backend", choices=("agate", "local"), default="agate")
    profile.add_argument("--wait-mode", choices=("inline",), default="inline")
    profile.add_argument("--wait-timeout", type=int, default=14_400)
    profile.add_argument("--job-timeout", type=int, default=600)
    profile.add_argument("--candidate", default="kernel.py")
    profile.add_argument("--reference-dir", default=".")
    profile.add_argument("--level", choices=("survey", "sol", "deep"), default="sol")
    profile.add_argument("--kernel-name", default="")
    profile.add_argument("--kernel-regex", default="")
    profile.add_argument("--source", action="store_true")
    profile.add_argument("--launch-skip", type=int, default=0)
    profile.add_argument("--launch-count", type=int, default=0)
    profile.add_argument(
        "--route", choices=("auto", "typed", "repository"), default="auto"
    )
    profile.add_argument("--agent-result-max-bytes", type=int, default=16 * 1024)
    profile.add_argument(
        "--agent-cli", choices=("claude", "qodercli", "codex", "pi"), default="claude"
    )

    show = subparsers.add_parser("show", help="read one bounded evidence section")
    show.add_argument("--evaluation", required=True)
    show.add_argument(
        "--section",
        choices=("summary", "hotspots", "source-lines", "sass", "artifacts"),
        default="summary",
    )
    show.add_argument("--max-bytes", type=int, default=16 * 1024)

    args = parser.parse_args(argv)
    if args.command == "submit":
        return _submit(parser, args)
    if args.command == "status":
        return _status(args)
    if args.command == "profile":
        return _profile(parser, args)
    if args.command == "show":
        return _show(args)
    return _collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
