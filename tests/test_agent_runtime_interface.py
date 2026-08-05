from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from orchestrator.agent_runtime import (
    AgentBackendAdapter,
    AgentRunRequest,
    BackendAdapterRegistry,
    TokenUsage,
    build_agent_runtime,
)


class AgentRuntimeInterfaceTest(unittest.TestCase):
    def test_factory_builds_each_supported_runtime(self) -> None:
        for runtime_id in ("claude", "qodercli", "codex", "pi"):
            with self.subTest(runtime_id=runtime_id):
                self.assertEqual(build_agent_runtime(runtime_id).id, runtime_id)

    def test_unknown_runtime_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported agent CLI"):
            build_agent_runtime("unknown")

    def test_runtime_executes_one_request_through_the_injected_process_runner(self) -> None:
        captured: dict[str, object] = {}

        def process_runner(
            command: list[str], cwd: Path, timeout: int, env: dict | None = None
        ) -> tuple[str, str, int, bool]:
            captured.update(command=command, cwd=cwd, timeout=timeout, env=env)
            return (
                '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":2}}',
                "stderr",
                0,
                False,
            )

        with tempfile.TemporaryDirectory(prefix="agent-runtime-interface-") as temp_dir:
            workspace = Path(temp_dir)
            runtime = build_agent_runtime("codex", process_runner=process_runner)
            result = runtime.run(
                AgentRunRequest(
                    workspace=workspace,
                    prompt="one bounded iteration",
                    timeout_s=123,
                    reasoning_effort="high",
                    sandbox_hardware="REMOTE_GPU",
                    sandbox_profile="",
                    sandbox_url="https://gateway.example.test",
                    sandbox_timeout_s=456,
                    extra_environment={"ATREX_TELEMETRY_TRACE": "trace.jsonl"},
                )
            )

        self.assertEqual(captured["cwd"], workspace)
        self.assertEqual(captured["timeout"], 123)
        command = captured["command"]
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertEqual(command[-1], "one bounded iteration")
        environment = captured["env"]
        self.assertEqual(environment["IS_SANDBOX"], "1")
        self.assertEqual(environment["ATREX_SANDBOX_GPU"], "REMOTE_GPU")
        self.assertEqual(
            environment["ATREX_SANDBOX_URL"], "https://gateway.example.test"
        )
        self.assertEqual(environment["ATREX_SANDBOX_TIMEOUT"], "456")
        self.assertEqual(environment["ATREX_TELEMETRY_TRACE"], "trace.jsonl")
        self.assertEqual(result.runtime_id, "codex")
        self.assertEqual(result.tokens, 9)
        self.assertEqual(
            result.terminal_usage,
            TokenUsage(
                input_tokens=7,
                output_tokens=2,
                cache_read_tokens=None,
                cache_write_tokens=None,
                total_tokens=9,
                measurement="exact",
            ),
        )
        self.assertEqual([event.kind for event in result.events], ["terminal_usage"])
        self.assertFalse(result.capabilities.usage_delta_observed)
        self.assertEqual(result.stderr_tail, "stderr")

    def test_adapter_normalizes_message_deltas_separately_from_terminal_usage(self) -> None:
        stdout = "\n".join(
            [
                '{"type":"assistant","message":{"usage":{"input_tokens":3,"output_tokens":2}}}',
                '{"type":"result","usage":{"input_tokens":3,"output_tokens":2}}',
            ]
        )

        def process_runner(*args, **kwargs):
            return stdout, "", 0, False

        with tempfile.TemporaryDirectory(prefix="agent-runtime-events-") as temp_dir:
            result = build_agent_runtime(
                "claude", process_runner=process_runner
            ).run(
                AgentRunRequest(
                    workspace=Path(temp_dir),
                    prompt="observe events",
                    timeout_s=10,
                )
            )

        self.assertEqual(
            [event.kind for event in result.events],
            ["usage_delta", "terminal_usage"],
        )
        self.assertEqual(result.events[0].usage.total_tokens, 5)
        self.assertEqual(result.terminal_usage.total_tokens, 5)
        self.assertTrue(result.capabilities.usage_delta_observed)

    def test_custom_adapter_can_be_registered_without_changing_the_runtime(self) -> None:
        class FakeAdapter(AgentBackendAdapter):
            id = "fake"
            settings_variable = "ATREX_FAKE_SESSION_SETTINGS"

            def build_command(self, prompt, session_id, reasoning_effort, settings):
                return ["fake-agent", prompt]

            def normalize_stream(self, stdout):
                return (), TokenUsage.unavailable()

            def auth_hint(self):
                return "configure fake-agent"

        registry = BackendAdapterRegistry()
        registry.register("fake", lambda humanize_dir: FakeAdapter())
        captured: dict[str, object] = {}

        def process_runner(command, cwd, timeout, env=None):
            captured["command"] = command
            return "", "", 0, False

        with tempfile.TemporaryDirectory(prefix="agent-runtime-registry-") as temp_dir:
            result = build_agent_runtime(
                "fake", process_runner=process_runner, registry=registry
            ).run(
                AgentRunRequest(
                    workspace=Path(temp_dir), prompt="hello", timeout_s=10
                )
            )

        self.assertEqual(captured["command"], ["fake-agent", "hello"])
        self.assertEqual(result.runtime_id, "fake")
        self.assertEqual(result.tokens, 0)
        self.assertEqual(result.terminal_usage.measurement, "unavailable")

    def test_observation_parser_failure_preserves_terminal_budget_tokens(self) -> None:
        class BrokenAdapter(AgentBackendAdapter):
            id = "broken"
            settings_variable = "ATREX_BROKEN_SESSION_SETTINGS"

            def build_command(self, prompt, session_id, reasoning_effort, settings):
                return ["broken-agent", prompt]

            def normalize_stream(self, stdout):
                raise RuntimeError("raw parser detail must not escape")

            def auth_hint(self):
                return "configure broken-agent"

        registry = BackendAdapterRegistry()
        registry.register("broken", lambda humanize_dir: BrokenAdapter())

        def process_runner(*args, **kwargs):
            return (
                '{"type":"result","usage":{"input_tokens":7,"output_tokens":2}}',
                "",
                0,
                False,
            )

        with tempfile.TemporaryDirectory(prefix="agent-runtime-failure-") as temp_dir:
            result = build_agent_runtime(
                "broken", process_runner=process_runner, registry=registry
            ).run(
                AgentRunRequest(
                    workspace=Path(temp_dir), prompt="hello", timeout_s=10
                )
            )

        self.assertEqual(result.tokens, 9)
        self.assertEqual(result.events, ())
        self.assertEqual(
            result.observation_errors,
            ("stream_normalization_failed:RuntimeError",),
        )


if __name__ == "__main__":
    unittest.main()
