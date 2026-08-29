#!/usr/bin/env python3
"""Run staged Atrex-Bench ABBA arms with one persistent shape worker.

Atrex-Bench normally starts a fresh Python process for every shape. That is
useful fault isolation for a general submission service, but a full 34-shape
A-B-B-A exceeds Agate's hard 600 second allocation limit even though the GPU
work itself is short. Repository Horizon already isolates every ABBA arm in a
fresh process, so all shapes in one arm can share that process and CUDA context.

This launcher does not reimplement correctness or timing. It loads the exact
staged Atrex-Bench runner and replaces only its per-shape process boundary;
``check_correctness`` and ``benchmark_performance`` remain authoritative.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from functools import partial
from pathlib import Path
from types import ModuleType


OFFICIAL_RUNNER = "_run_eval_official.py"


def _load_official() -> ModuleType:
    path = Path(__file__).with_name(OFFICIAL_RUNNER)
    spec = importlib.util.spec_from_file_location("_atrex_bench_official_run_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load staged Atrex-Bench runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "main",
        "_run_single_shape_main",
        "_load_shape_result_payload",
        "_correctness_from_payload",
        "_performance_from_payload",
        "_subworker_failure_results",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            "staged Atrex-Bench runner is incompatible with persistent ABBA: "
            + ", ".join(missing)
        )
    return module


def _run_single_shape_in_process(official: ModuleType, **kwargs):
    """Execute one official shape body without starting another interpreter."""
    shape_results_dir = kwargs.pop("shape_results_dir")
    shape_id = kwargs["shape_id"]
    shape_result_path = shape_results_dir / f"{shape_id}.json"
    shape_result_path.unlink(missing_ok=True)
    validation_mode = kwargs.get("validation_mode", "full")
    try:
        official._run_single_shape_main(
            **kwargs,
            shape_result_path=shape_result_path,
        )
        payload = official._load_shape_result_payload(shape_result_path)
        if payload is None:
            raise RuntimeError("persistent shape worker wrote no valid result")
        correctness = official._correctness_from_payload(payload["correctness"])
        performance = official._performance_from_payload(payload["performance"])
        return correctness, performance, payload.get("compile_succeeded")
    except BaseException as exc:  # match subprocess failure -> failed shape
        reason = (
            "Persistent per-arm shape worker failed: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        )
        return official._subworker_failure_results(
            validation_mode=validation_mode,
            reason=reason,
        )


def main() -> None:
    official = _load_official()
    # The official parent and worker construct child argv from __file__. Point
    # both back through this launcher so the worker process gets this override.
    official.__file__ = __file__
    official._run_single_shape_subprocess = partial(
        _run_single_shape_in_process, official
    )
    # Kernel attribution is an official optional pass and is not part of the
    # ABBA correctness/latency score. Omitting it avoids profiler overhead and
    # process-global tracker state crossing shape boundaries.
    if "--skip-kernel-attribution" not in sys.argv:
        sys.argv.append("--skip-kernel-attribution")
    official.main()


if __name__ == "__main__":
    main()
