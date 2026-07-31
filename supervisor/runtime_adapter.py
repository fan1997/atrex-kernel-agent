from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from typing import Any, Iterator

from .bridge import SupervisorRuntime
from .session_runner import supervised_run_session


EXPECTED_RUN_SESSION_PARAMETERS = (
    "workspace",
    "prompt",
    "timeout",
    "agent_cli",
    "sandbox_hardware",
    "sandbox_profile",
    "sandbox_url",
    "sandbox_timeout",
)


def assert_compatible(base: Any) -> None:
    actual = tuple(inspect.signature(base.run_session).parameters)
    if actual != EXPECTED_RUN_SESSION_PARAMETERS:
        raise RuntimeError(
            "Supervisor adapter is incompatible with orchestrator.optimize.run_session. "
            f"Expected {EXPECTED_RUN_SESSION_PARAMETERS}, found {actual}."
        )
    required = ("_session_command", "_session_env", "_tokens_from_stream", "SessionResult")
    missing = [name for name in required if not hasattr(base, name)]
    if missing:
        raise RuntimeError(f"Supervisor adapter is missing base orchestrator APIs: {missing}")


@contextmanager
def install_supervised_runtime(base: Any, runtime: SupervisorRuntime) -> Iterator[None]:
    """Temporarily route every base orchestrator coding session through Supervisor."""
    assert_compatible(base)
    original = base.run_session
    original_dispatch = getattr(base, "dispatch_framework_campaigns", None)
    campaign_class = getattr(base, "Campaign", None)
    original_campaign_finish = getattr(campaign_class, "_finish", None)
    original_campaign_setup = getattr(campaign_class, "setup_baseline", None)
    original_file = getattr(base, "__file__", "")
    supervisor_entry = runtime.config.repository_root / "supervisor" / "optimize.py"

    env_values = {
        "ATREX_SUPERVISOR_DATA_ROOT": str(runtime.config.data_root),
        "ATREX_SUPERVISOR_CLI": runtime.config.cli,
        "ATREX_SUPERVISOR_MODEL": runtime.config.model,
        "ATREX_SUPERVISOR_REASONING_EFFORT": runtime.config.reasoning_effort,
        "ATREX_SUPERVISOR_SESSION_SETTINGS": runtime.config.settings,
        "ATREX_SUPERVISOR_ACTIVATION_TIMEOUT": str(runtime.config.activation_timeout),
        "ATREX_SUPERVISOR_EVERY_ITERATIONS": str(runtime.config.every_iterations),
        "ATREX_SUPERVISOR_MAX_ACTIVATIONS": str(runtime.config.max_activations),
        "ATREX_SUPERVISOR_MAX_RESTARTS": str(runtime.config.max_restarts_per_session),
        "ATREX_SUPERVISOR_REQUIRED": "1" if runtime.config.required else "0",
    }
    previous_env = {key: os.environ.get(key) for key in env_values}
    os.environ.update(env_values)

    def run_session(
        workspace,
        prompt,
        timeout,
        agent_cli="claude",
        sandbox_hardware="",
        sandbox_profile="",
        sandbox_url="",
        sandbox_timeout=600,
    ):
        return supervised_run_session(
            base,
            runtime,
            workspace,
            prompt,
            timeout,
            agent_cli,
            sandbox_hardware,
            sandbox_profile,
            sandbox_url,
            sandbox_timeout,
        )

    base.run_session = run_session
    if campaign_class is not None and original_campaign_finish is not None:
        def campaign_finish(campaign, reason):
            runtime.bridge_for(campaign.workspace).before_stop(
                f"base orchestrator is about to stop: {reason}"
            )
            return original_campaign_finish(campaign, reason)

        campaign_class._finish = campaign_finish
    if campaign_class is not None and original_campaign_setup is not None:
        def campaign_setup(campaign):
            result = original_campaign_setup(campaign)
            runtime.bridge_for(campaign.workspace).confirm_existing_baseline()
            return result

        campaign_class.setup_baseline = campaign_setup
    if original_dispatch is not None:
        def dispatch_framework_campaigns(*args, **kwargs):
            # The base dispatcher resolves its child script from module.__file__. Redirect only
            # for the duration of this call; other base paths such as layer anchor_bench remain
            # untouched. Child Supervisor config is inherited through the environment above.
            base.__file__ = str(supervisor_entry)
            try:
                return original_dispatch(*args, **kwargs)
            finally:
                base.__file__ = original_file

        base.dispatch_framework_campaigns = dispatch_framework_campaigns
    try:
        yield
    finally:
        base.run_session = original
        if campaign_class is not None and original_campaign_finish is not None:
            campaign_class._finish = original_campaign_finish
        if campaign_class is not None and original_campaign_setup is not None:
            campaign_class.setup_baseline = original_campaign_setup
        if original_dispatch is not None:
            base.dispatch_framework_campaigns = original_dispatch
        base.__file__ = original_file
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        runtime.close()
