from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from orchestrator.optimize import framework_workspace_suffix

from .baseline import RepositoryBaselineManager
from .capabilities import probe_capabilities
from .campaign import RepositoryCampaign, RepositoryHorizonCampaign
from .compat import (
    assert_upstream_compatible,
    git_head,
    latest_version,
    working_changes,
)
from .config import EvaluationPolicy, endpoint_is_local
from .manifest import load_manifest
from .preplan import PreplanRunner
from .support_wheel import canonical_distribution
from .verifier import RepositoryABBAValidator, RepositoryPhaseValidator


def _atrex_bench_root(op_dir: Path) -> Path:
    for candidate in (op_dir, *op_dir.parents):
        if (candidate / "scripts" / "run_eval.py").is_file() and (
            candidate / "src" / "atrex_bench"
        ).is_dir():
            return candidate
    raise ValueError(
        f"cannot find Atrex-Bench root above {op_dir}; missing scripts/run_eval.py/src/atrex_bench"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Main-based repository-backed Long Horizon optimizer (v3)"
    )
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-checkout", required=True)
    parser.add_argument(
        "--support-wheel",
        action="append",
        default=[],
        metavar="DISTRIBUTION=PATH",
    )
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--sandbox-hardware", default="")
    parser.add_argument("--sandbox-profile", choices=("pre", "prod"), default="")
    parser.add_argument("--sandbox-url", default="")
    parser.add_argument("--sandbox-timeout", type=int, default=600)
    parser.add_argument(
        "--evaluation-backend", choices=("agate", "local"), default="agate"
    )
    parser.add_argument("--evaluation-wait-mode", choices=("inline",), default="inline")
    parser.add_argument("--evaluation-wait-timeout", type=int, default=14_400)
    parser.add_argument("--agent-result-max-bytes", type=int, default=16 * 1024)
    parser.add_argument(
        "--agent-cli",
        choices=("claude", "qodercli", "codex", "pi"),
        default="claude",
    )
    parser.add_argument(
        "--optimization-mode", choices=("production",), default="production"
    )
    parser.add_argument("--framework", choices=("CuteDSL",), default="CuteDSL")
    parser.add_argument("--framework-baseline", choices=("never",), default="never")
    parser.add_argument("--no-workload-bucketing", action="store_true")
    parser.add_argument("--episode-policy", choices=("main-v3",), default="main-v3")
    parser.add_argument(
        "--max-iters", "--max-episodes", dest="max_episodes", type=int, default=20
    )
    parser.add_argument("--token-budget", type=int, default=0)
    parser.add_argument("--max-stall", type=int, default=0)
    parser.add_argument("--handoff-resumes", type=int, default=1)
    parser.add_argument("--verify-repeats", type=int, default=2)
    parser.add_argument("--verify-run-timeout", type=int, default=120)
    parser.add_argument("--min-improvement-pct", type=float, default=0.0)
    parser.add_argument("--arch", default="")
    parser.add_argument("--notes", default="none")
    parser.add_argument("--workspace", default="")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate/prepare the locked incumbent and exit before starting an Agent episode",
    )
    parser.add_argument(
        "--preplan-only",
        action="store_true",
        help=(
            "prepare the locked incumbent, run one isolated end-to-end architecture "
            "Preplan session, validate its artifact, and exit with zero formal episodes"
        ),
    )
    parser.add_argument(
        "--preplan-timeout",
        type=int,
        default=7200,
        help="wall-clock timeout in seconds for the single Preplan session",
    )
    return parser


def _parse_support_wheels(values: list[str], manifest) -> dict[str, Path]:
    support_wheels: dict[str, Path] = {}
    for value in values:
        distribution, separator, raw_path = value.partition("=")
        if not separator or not distribution or not raw_path:
            raise ValueError("--support-wheel must be DISTRIBUTION=PATH")
        key = canonical_distribution(distribution)
        if key in support_wheels:
            raise ValueError(f"duplicate --support-wheel distribution: {distribution}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"support wheel not found: {path}")
        support_wheels[key] = path
    declared = {
        canonical_distribution(item.distribution) for item in manifest.runtime_support
    }
    unexpected = sorted(set(support_wheels) - declared)
    if unexpected:
        raise ValueError(
            "--support-wheel not declared by manifest: " + ", ".join(unexpected)
        )
    return support_wheels


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        assert_upstream_compatible()
        if not args.no_workload_bucketing:
            raise ValueError("repository horizon requires --no-workload-bucketing")
        if args.preflight_only and args.preplan_only:
            raise ValueError(
                "--preflight-only and --preplan-only are mutually exclusive"
            )
        if args.preplan_timeout <= 0:
            raise ValueError("--preplan-timeout must be positive")
        if args.sandbox_profile and args.sandbox_url:
            raise ValueError(
                "--sandbox-profile and --sandbox-url are mutually exclusive"
            )
        if args.max_episodes <= 0:
            raise ValueError("--max-episodes must be positive")
        if args.handoff_resumes < 0:
            raise ValueError("--handoff-resumes must be non-negative")
        if args.verify_repeats <= 0 or args.verify_run_timeout <= 0:
            raise ValueError("verification repeats/timeouts must be positive")
        if (
            2 * args.verify_repeats * args.verify_run_timeout + 30
            > args.sandbox_timeout
        ):
            raise ValueError(
                "2 * --verify-repeats * --verify-run-timeout + 30 must fit --sandbox-timeout"
            )
        if shutil.which(args.agent_cli) is None:
            raise ValueError(f"agent CLI not found on PATH: {args.agent_cli}")

        manifest = load_manifest(args.source_manifest)
        source_checkout = Path(args.source_checkout).expanduser().resolve()
        if not (source_checkout / ".git").exists():
            raise ValueError(
                f"source checkout is not a Git checkout: {source_checkout}"
            )
        support_wheels = _parse_support_wheels(args.support_wheel, manifest)
        op_dir = Path(args.op_dir).expanduser().resolve()
        if not (op_dir / "reference.py").is_file():
            raise ValueError(f"operator is missing reference.py: {op_dir}")
        atrex_root = _atrex_bench_root(op_dir)
        workspace_root = (
            Path(args.workspace).expanduser().resolve()
            if args.workspace
            else Path.cwd()
        )
        hardware = args.sandbox_hardware or args.platform
        policy = EvaluationPolicy(
            backend=args.evaluation_backend,
            wait_mode=args.evaluation_wait_mode,
            wait_timeout=args.evaluation_wait_timeout,
            agent_result_max_bytes=args.agent_result_max_bytes,
        )
        resolved_wait_mode = policy.resolved_wait_mode(
            args.agent_cli,
            endpoint_is_local=endpoint_is_local(args.sandbox_url, hardware),
        )
        suffix = framework_workspace_suffix(
            args.framework, args.platform, args.optimization_mode
        )
        campaign = RepositoryCampaign(
            name=op_dir.name,
            kernel_demo=str(op_dir / "reference.py"),
            platform=args.platform,
            framework=args.framework,
            notes=args.notes,
            arch=args.arch,
            work_dir=str(workspace_root),
            workspace_suffix=suffix,
            max_iters=args.max_episodes,
            token_budget=args.token_budget,
            max_stall=args.max_stall,
            sandbox_hardware=hardware,
            sandbox_profile=args.sandbox_profile,
            sandbox_url=args.sandbox_url,
            sandbox_timeout=args.sandbox_timeout,
            atrex_bench_root=str(atrex_root),
            agent_cli=args.agent_cli,
            optimization_mode=args.optimization_mode,
            framework_baseline=args.framework_baseline,
            handoff_resumes=args.handoff_resumes,
            verify_repeats=args.verify_repeats,
            verify_run_timeout=args.verify_run_timeout,
            min_improvement_pct=args.min_improvement_pct,
            repository_manifest=manifest,
        )
        campaign.repository_evaluation_policy = policy
        campaign.repository_capabilities = probe_capabilities(
            policy,
            hardware=hardware,
            profile=args.sandbox_profile,
            url=args.sandbox_url,
        )
        baseline = RepositoryBaselineManager(
            manifest,
            source_checkout,
            support_wheels,
        )
        normal = RepositoryABBAValidator(
            manifest=manifest,
            atrex_bench_root=atrex_root,
            hardware=hardware,
            profile=args.sandbox_profile,
            url=args.sandbox_url,
            timeout=args.sandbox_timeout,
            repeats=args.verify_repeats,
            per_run_timeout=args.verify_run_timeout,
            min_improvement_pct=args.min_improvement_pct,
            backend=policy.backend,
            wait_mode=resolved_wait_mode,
            wait_timeout=policy.wait_timeout,
            agent_result_max_bytes=policy.agent_result_max_bytes,
            private_reference_dir=campaign.private_reference_dir,
        )
        bringup_repeats = manifest.bringup.probe_repeats
        bringup = RepositoryABBAValidator(
            manifest=manifest,
            atrex_bench_root=atrex_root,
            hardware=hardware,
            profile=args.sandbox_profile,
            url=args.sandbox_url,
            timeout=args.sandbox_timeout,
            repeats=bringup_repeats,
            per_run_timeout=max(1, (args.sandbox_timeout - 30) // bringup_repeats),
            min_improvement_pct=-100.0,
            candidate_only=True,
            backend=policy.backend,
            wait_mode=resolved_wait_mode,
            wait_timeout=policy.wait_timeout,
            agent_result_max_bytes=policy.agent_result_max_bytes,
            private_reference_dir=campaign.private_reference_dir,
        )
        controller = RepositoryHorizonCampaign(
            base_campaign=campaign,
            manifest=manifest,
            baseline=baseline,
            verifier=RepositoryPhaseValidator(normal, bringup),
            max_version=args.max_episodes,
            token_budget=args.token_budget,
            handoff_resumes=args.handoff_resumes,
            max_stall=args.max_stall,
            evaluation_policy=policy,
        )
        print(
            f"[repository-horizon] source={manifest.source_name}@{manifest.revision[:12]} "
            f"agent={args.agent_cli} policy=main-v3 platform={args.platform} "
            f"evaluation={policy.backend}/{resolved_wait_mode} "
            f"hardware={hardware} endpoint={args.sandbox_profile or args.sandbox_url or 'default'} "
            f"workspace={campaign.workspace}",
            flush=True,
        )
        if args.preflight_only:
            baseline.prepare(campaign)
            dirty = working_changes(campaign.workspace)
            if dirty:
                raise RuntimeError(
                    "preflight left a dirty incumbent: " + ", ".join(dirty[:12])
                )
            print(
                f"[repository-horizon] PREFLIGHT PASS head={git_head(campaign.workspace)} "
                f"latest_version=v{latest_version(campaign.workspace)}",
                flush=True,
            )
            return 0
        if args.preplan_only:
            baseline.prepare(campaign)
            PreplanRunner(
                campaign=campaign,
                manifest=manifest,
                timeout=args.preplan_timeout,
            ).run()
            return 0
        controller.run()
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
