#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from supervisor.bridge import SupervisorConfig, SupervisorRuntime
    from supervisor.runtime_adapter import install_supervised_runtime
else:
    from .bridge import SupervisorConfig, SupervisorRuntime
    from .runtime_adapter import install_supervised_runtime


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPOSITORY_ROOT.parent / "atrex-supervisor-data"


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--supervisor-help", action="store_true")
    parser.add_argument(
        "--supervisor-data-root",
        default=os.environ.get("ATREX_SUPERVISOR_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
    )
    parser.add_argument("--supervisor-cli", default=os.environ.get("ATREX_SUPERVISOR_CLI", "codex"))
    parser.add_argument(
        "--supervisor-model",
        default=os.environ.get("ATREX_SUPERVISOR_MODEL", "gpt-5.6-sol"),
    )
    parser.add_argument(
        "--supervisor-reasoning-effort",
        default=os.environ.get("ATREX_SUPERVISOR_REASONING_EFFORT", "max"),
    )
    parser.add_argument(
        "--supervisor-settings",
        default=os.environ.get("ATREX_SUPERVISOR_SESSION_SETTINGS", ""),
    )
    parser.add_argument(
        "--supervisor-activation-timeout",
        type=int,
        default=int(os.environ.get("ATREX_SUPERVISOR_ACTIVATION_TIMEOUT", "900")),
    )
    parser.add_argument(
        "--supervisor-every-iterations",
        type=int,
        default=int(os.environ.get("ATREX_SUPERVISOR_EVERY_ITERATIONS", "5")),
    )
    parser.add_argument(
        "--supervisor-every-events",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--supervisor-every-sessions",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--supervisor-max-activations",
        type=int,
        default=int(os.environ.get("ATREX_SUPERVISOR_MAX_ACTIVATIONS", "100")),
    )
    parser.add_argument(
        "--supervisor-max-restarts",
        type=int,
        default=int(os.environ.get("ATREX_SUPERVISOR_MAX_RESTARTS", "2")),
    )
    parser.add_argument(
        "--supervisor-required",
        action="store_true",
        default=os.environ.get("ATREX_SUPERVISOR_REQUIRED", "0") == "1",
    )
    parser.add_argument(
        "--supervisor-activate-now",
        metavar="WORKSPACE",
        help="Run one explicit manual Supervisor review for an existing workspace, then exit.",
    )
    parser.add_argument(
        "--supervisor-manual-reason",
        default="operator requested an immediate macro review",
    )
    args, optimize_args = parser.parse_known_args(argv)
    if args.supervisor_help:
        parser.print_help()
        print("\nAll remaining arguments are passed unchanged to orchestrator/optimize.py.")
    return args, optimize_args


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args, optimize_args = _parse_args(raw_argv)
    if args.supervisor_help:
        return 0
    if args.supervisor_activation_timeout < 1:
        raise SystemExit("--supervisor-activation-timeout must be positive")
    if args.supervisor_every_events is not None:
        raise SystemExit(
            "--supervisor-every-events was removed: V1 strong Supervisor activations are "
            "never event-triggered"
        )
    if args.supervisor_every_sessions is not None:
        raise SystemExit(
            "--supervisor-every-sessions was renamed to --supervisor-every-iterations; "
            "attempts/restarts are no longer schedule units"
        )
    if args.supervisor_every_iterations < 1:
        raise SystemExit("--supervisor-every-iterations must be positive")
    if args.supervisor_max_restarts < 0:
        raise SystemExit("--supervisor-max-restarts cannot be negative")
    if args.supervisor_settings:
        try:
            settings = json.loads(args.supervisor_settings)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid --supervisor-settings JSON: {exc}") from exc
        if not isinstance(settings, dict):
            raise SystemExit("--supervisor-settings must be a JSON object")

    config = SupervisorConfig(
        data_root=Path(args.supervisor_data_root),
        repository_root=REPOSITORY_ROOT,
        cli=args.supervisor_cli,
        model=args.supervisor_model,
        reasoning_effort=args.supervisor_reasoning_effort,
        settings=args.supervisor_settings,
        activation_timeout=args.supervisor_activation_timeout,
        every_iterations=args.supervisor_every_iterations,
        max_activations=args.supervisor_max_activations,
        max_restarts_per_session=args.supervisor_max_restarts,
        required=args.supervisor_required,
    )
    runtime = SupervisorRuntime(config)

    if args.supervisor_activate_now:
        if optimize_args:
            raise SystemExit(
                "--supervisor-activate-now is manual-only and cannot be combined with base "
                "orchestrator arguments"
            )
        bridge = runtime.bridge_for(Path(args.supervisor_activate_now))
        try:
            success = bridge.manual_review(args.supervisor_manual_reason)
            result = bridge.service.last_result
            if result is not None:
                print(
                    "[supervisor] manual activation "
                    f"exit={result.exit_status} tokens={result.tokens} "
                    f"artifact={result.activation_dir}",
                    flush=True,
                )
            return 0 if success else 1
        finally:
            runtime.close()

    # Import lazily so Supervisor-only help/manual review and storage/tool tests do not
    # initialize the comparatively heavy base orchestrator or optional runtime dependencies.
    from orchestrator import optimize as base_optimize

    print(
        "[supervisor] enabled "
        f"model={config.model} data_root={config.data_root} "
        "activation_policy=baseline+every-"
        f"{config.every_iterations}-iterations+before-stop+manual event_activation=off",
        flush=True,
    )
    with install_supervised_runtime(base_optimize, runtime):
        result = base_optimize.main(optimize_args)
        runtime.before_stop(f"orchestrator returned exit={result}")
        return result


if __name__ == "__main__":
    raise SystemExit(main())
