"""Characterize the current Claude/Qoder/Codex/Pi runtime contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrator import optimize


class AgentCommandCharacterizationTest(unittest.TestCase):
    def test_each_backend_has_a_fresh_noninteractive_command(self) -> None:
        expected = {
            "claude": [
                "claude",
                "--print",
                "--verbose",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--session-id",
                "session-123",
                "--effort",
                "high",
                "prompt",
            ],
            "qodercli": [
                "qodercli",
                "--print",
                "--dangerously-skip-permissions",
                "--output-format",
                "stream-json",
                "--session-id",
                "session-123",
                "--no-session-persistence",
                "--reasoning-effort",
                "high",
                "prompt",
            ],
            "codex": [
                "codex",
                "exec",
                "--json",
                "--ephemeral",
                "--color",
                "never",
                "--dangerously-bypass-approvals-and-sandbox",
                "-c",
                'model_reasoning_effort="high"',
                "prompt",
            ],
            "pi": [
                "pi",
                "--mode",
                "json",
                "--session-id",
                "session-123",
                "--approve",
                "--thinking",
                "high",
                "prompt",
            ],
        }
        with tempfile.TemporaryDirectory(prefix="runtime-command-") as temp_dir:
            with (
                mock.patch.object(optimize, "HUMANIZE_DIR", Path(temp_dir)),
                mock.patch.dict(optimize.os.environ, {}, clear=True),
            ):
                for backend, command in expected.items():
                    with self.subTest(backend=backend):
                        self.assertEqual(
                            optimize._session_command(
                                backend,
                                "prompt",
                                "session-123",
                                reasoning_effort="high",
                            ),
                            command,
                        )

    def test_native_settings_override_the_generic_fallback(self) -> None:
        cases = (
            ("claude", "ATREX_CLAUDE_SESSION_SETTINGS", "claude-native"),
            ("qodercli", "ATREX_QODER_SESSION_SETTINGS", "qoder-native"),
        )
        with tempfile.TemporaryDirectory(prefix="runtime-settings-") as temp_dir:
            for backend, variable, native_value in cases:
                with self.subTest(backend=backend):
                    with (
                        mock.patch.object(optimize, "HUMANIZE_DIR", Path(temp_dir)),
                        mock.patch.dict(
                            optimize.os.environ,
                            {
                                "ATREX_SESSION_SETTINGS": "generic-fallback",
                                variable: native_value,
                            },
                            clear=True,
                        ),
                    ):
                        command = optimize._session_command(
                            backend, "prompt", "session-123"
                        )
                    settings_index = command.index("--settings")
                    self.assertEqual(command[settings_index + 1], native_value)
                    self.assertNotIn("generic-fallback", command)

    def test_pi_settings_select_provider_and_model(self) -> None:
        with (
            mock.patch.object(optimize, "HUMANIZE_DIR", Path("missing")),
            mock.patch.dict(
                optimize.os.environ,
                {
                    "ATREX_PI_SESSION_SETTINGS": json.dumps(
                        {"provider": "anthropic", "model": "claude-opus"}
                    )
                },
                clear=True,
            ),
        ):
            command = optimize._session_command("pi", "prompt", "session-123")

        self.assertIn("--provider", command)
        self.assertEqual(command[command.index("--provider") + 1], "anthropic")
        self.assertIn("--model", command)
        self.assertEqual(command[command.index("--model") + 1], "claude-opus")

    def test_claude_loads_humanize_only_when_the_plugin_is_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="claude-plugin-") as temp_dir:
            humanize = Path(temp_dir)
            marker = humanize / "skills" / "humanize-gen-plan" / "SKILL.md"
            marker.parent.mkdir(parents=True)
            marker.write_text("---\nname: humanize-gen-plan\n---\n", encoding="utf-8")
            with (
                mock.patch.object(optimize, "HUMANIZE_DIR", humanize),
                mock.patch.dict(optimize.os.environ, {}, clear=True),
            ):
                command = optimize._session_command(
                    "claude", "prompt", "session-123"
                )
        plugin_index = command.index("--plugin-dir")
        self.assertEqual(command[plugin_index + 1], str(humanize))

    def test_reasoning_effort_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported reasoning effort"):
            optimize._session_command(
                "codex", "prompt", "session-123", reasoning_effort="xhigh"
            )

    def test_auth_diagnostics_remain_backend_specific(self) -> None:
        self.assertEqual(
            optimize._agent_auth_hint("claude"),
            'run `claude auth status` and `claude --print "test"` to diagnose',
        )
        self.assertEqual(
            optimize._agent_auth_hint("qodercli"),
            'run `qodercli status` and `qodercli --print "test"` to diagnose',
        )
        self.assertEqual(
            optimize._agent_auth_hint("codex"),
            'run `codex login status` and `codex exec --ephemeral "reply ok"` to diagnose',
        )
        self.assertEqual(
            optimize._agent_auth_hint("pi"),
            'run `pi --list-models` and `pi -p "reply ok"` to diagnose',
        )


class AgentEnvironmentCharacterizationTest(unittest.TestCase):
    def test_session_environment_preserves_backend_auth_and_process_guards(self) -> None:
        base = {
            "PATH": "/usr/bin",
            "ANTHROPIC_AUTH_TOKEN": "bearer-token",
            "ANTHROPIC_API_KEY": "api-key",
            "XDG_CACHE_HOME": "/tmp/atrex-characterization-cache",
        }
        python_bin = str(Path(sys.executable).resolve().parent)
        expected_state = "/tmp/atrex-characterization-cache/atrex-local-gateway"
        for backend in optimize.AGENT_CLI_CHOICES:
            with self.subTest(backend=backend):
                with mock.patch.dict(optimize.os.environ, base, clear=True):
                    environment = optimize._session_env(backend)
                self.assertEqual(environment["PATH"].split(optimize.os.pathsep)[0], python_bin)
                self.assertEqual(environment["PIP_ONLY_BINARY"], ":all:")
                self.assertEqual(environment["PIP_INDEX_URL"], optimize.PYPI_MIRROR)
                self.assertEqual(environment["PIP_DISABLE_PIP_VERSION_CHECK"], "1")
                self.assertEqual(environment["BASH_ENV"], str(optimize.SESSION_SHELL_GUARD))
                self.assertEqual(
                    environment["ATREX_PROTECTED_GATEWAY_SCREEN"],
                    optimize.DEFAULT_PROTECTED_GATEWAY_SCREEN,
                )
                self.assertEqual(
                    environment["ATREX_PROTECTED_GATEWAY_STATE_DIR"], expected_state
                )
                if backend == "claude":
                    self.assertNotIn("ANTHROPIC_API_KEY", environment)
                else:
                    self.assertEqual(environment["ANTHROPIC_API_KEY"], "api-key")
                if backend == "pi":
                    self.assertEqual(environment["PI_SKIP_VERSION_CHECK"], "1")
                    self.assertEqual(environment["PI_TELEMETRY"], "0")

    def test_session_environment_deduplicates_the_active_python_bin(self) -> None:
        python_bin = str(Path(sys.executable).resolve().parent)
        with mock.patch.dict(
            optimize.os.environ,
            {"PATH": optimize.os.pathsep.join(["/usr/bin", python_bin, "/bin"])},
            clear=True,
        ):
            environment = optimize._session_env("codex")
        path_parts = environment["PATH"].split(optimize.os.pathsep)
        self.assertEqual(path_parts[0], python_bin)
        self.assertEqual(path_parts.count(python_bin), 1)


class TokenParsingCharacterizationTest(unittest.TestCase):
    def test_terminal_result_usage_wins_for_claude_and_qoder_streams(self) -> None:
        stdout = "\n".join(
            [
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {"usage": {"input_tokens": 999, "output_tokens": 1}},
                    }
                ),
                json.dumps(
                    {
                        "type": "result",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "cache_creation_input_tokens": 3,
                            "cache_read_input_tokens": 4,
                        },
                    }
                ),
            ]
        )
        self.assertEqual(optimize._tokens_from_stream(stdout), 19)

    def test_terminal_model_usage_supports_camel_case_counters(self) -> None:
        stdout = json.dumps(
            {
                "type": "result",
                "modelUsage": {
                    "model-a": {
                        "inputTokens": 11,
                        "outputTokens": 2,
                        "cacheCreationInputTokens": 3,
                        "cacheReadInputTokens": 5,
                    },
                    "model-b": {"inputTokens": 7, "outputTokens": 1},
                },
            }
        )
        self.assertEqual(optimize._tokens_from_stream(stdout), 29)

    def test_message_usage_is_the_fallback_when_no_terminal_result_exists(self) -> None:
        stdout = "\n".join(
            [
                "not-json",
                '{"type":"message","usage":{"input_tokens":5,"output_tokens":2}}',
                '{"message":{"usage":{"inputTokens":3,"outputTokens":1}}}',
                "{malformed-json",
            ]
        )
        self.assertEqual(optimize._tokens_from_stream(stdout), 11)


class RunSessionCharacterizationTest(unittest.TestCase):
    def test_each_backend_receives_the_same_sandbox_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="run-session-") as temp_dir:
            workspace = Path(temp_dir)
            humanize = workspace / "missing-humanize"
            for backend in optimize.AGENT_CLI_CHOICES:
                captured: dict[str, object] = {}

                def fake_run(
                    command: list[str],
                    cwd: Path,
                    timeout: int,
                    env: dict | None = None,
                ) -> tuple[str, str, int, bool]:
                    captured.update(command=command, cwd=cwd, timeout=timeout, env=env)
                    stdout = (
                        "\n".join(
                            [
                                '{"type":"message_end","message":{"role":"assistant",'
                                '"usage":{"input":7,"output":2,"cacheRead":0,'
                                '"cacheWrite":0,"totalTokens":9}}}',
                                '{"type":"agent_settled"}',
                            ]
                        )
                        if backend == "pi"
                        else '{"type":"result","usage":{"input_tokens":7,"output_tokens":2}}'
                    )
                    return stdout, "stderr", 0, False

                with self.subTest(backend=backend):
                    with (
                        mock.patch.object(optimize, "HUMANIZE_DIR", humanize),
                        mock.patch.object(optimize, "_run_bounded", side_effect=fake_run),
                        mock.patch.object(optimize.uuid, "uuid4", return_value="session-fixed"),
                        mock.patch.dict(
                            optimize.os.environ,
                            {"ATREX_SANDBOX_PROFILE": "inherited-profile"},
                            clear=True,
                        ),
                    ):
                        result = optimize.run_session(
                            workspace,
                            "prompt",
                            timeout=123,
                            agent_cli=backend,
                            sandbox_hardware="REMOTE_GPU",
                            sandbox_profile="requested-profile",
                            sandbox_url="https://gateway.example.test",
                            sandbox_timeout=456,
                            reasoning_effort="high",
                        )
                    environment = captured["env"]
                    self.assertIsInstance(environment, dict)
                    self.assertEqual(captured["cwd"], workspace)
                    self.assertEqual(captured["timeout"], 123)
                    self.assertEqual(environment["IS_SANDBOX"], "1")
                    self.assertEqual(environment["ATREX_SANDBOX_GPU"], "REMOTE_GPU")
                    self.assertEqual(
                        environment["ATREX_SANDBOX_URL"],
                        "https://gateway.example.test",
                    )
                    self.assertNotIn("ATREX_SANDBOX_PROFILE", environment)
                    self.assertEqual(environment["ATREX_SANDBOX_TIMEOUT"], "456")
                    command = captured["command"]
                    self.assertIsInstance(command, list)
                    if backend == "codex":
                        self.assertNotIn("session-fixed", command)
                    else:
                        self.assertIn("session-fixed", command)
                    self.assertEqual(result.tokens, 9)
                    self.assertEqual(result.stderr_tail, "stderr")
                    self.assertEqual(result.session_id, "session-fixed")


class SessionProcessPolicyCharacterizationTest(unittest.TestCase):
    def test_dependency_and_host_gpu_actions_are_classified(self) -> None:
        cases = (
            (["pip", "install", "package"], "third-party package installation/build command"),
            (["python", "-m", "pip", "install", "package"], "third-party package installation/build command"),
            (["uv", "pip", "install", "package"], "third-party package installation/build command"),
            (
                ["python", "tools/sandbox.py", "--", "pip", "install", "package"],
                "third-party package installation/build command",
            ),
            (["python", "test_kernel.py"], "kernel/evaluator executed directly on the host"),
            (["python", "-c", "import kernel"], "kernel imported directly on the host"),
            (["ncu", "--set", "full"], "GPU profiler executed directly on the host"),
            (
                ["bash", "tools/profile_nvidia.sh", "kernel.py"],
                "GPU profiler wrapper executed directly on the host",
            ),
        )
        for command, expected in cases:
            with self.subTest(command=command):
                self.assertEqual(optimize._dependency_process_violation(command), expected)

    def test_sandboxed_and_static_research_commands_remain_allowed(self) -> None:
        allowed = (
            [
                "python",
                "tools/sandbox.py",
                "--kind",
                "run",
                "--",
                "python",
                "test_kernel.py",
            ],
            ["python", "-c", "print('import kernel')"],
            ["cat", "kernel.py"],
            ["nvcc", "--version"],
        )
        for command in allowed:
            with self.subTest(command=command):
                self.assertIsNone(optimize._dependency_process_violation(command))

    def test_shared_gateway_lifecycle_and_state_mutations_are_blocked(self) -> None:
        environment = {
            "ATREX_PROTECTED_GATEWAY_SCREEN": "shared-gateway",
            "ATREX_PROTECTED_GATEWAY_STATE_DIR": "/tmp/shared-gateway-state",
        }
        commands = (
            ["screen", "-S", "shared-gateway", "-X", "quit"],
            ["rm", "-rf", "/tmp/shared-gateway-state"],
            ["curl", "http://localhost/v1/jobs/job-1/cancel"],
        )
        with mock.patch.dict(optimize.os.environ, environment, clear=True):
            for command in commands:
                with self.subTest(command=command):
                    self.assertEqual(
                        optimize._dependency_process_violation(command),
                        "shared localhost gateway lifecycle/state mutation",
                    )


class BoundedProcessCharacterizationTest(unittest.TestCase):
    def test_bounded_process_preserves_output_and_exit_status(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bounded-process-") as temp_dir:
            stdout, stderr, returncode, timed_out = optimize._run_bounded(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('stdout'); print('stderr', file=sys.stderr); raise SystemExit(7)",
                ],
                cwd=Path(temp_dir),
                timeout=5,
                env=optimize.os.environ.copy(),
            )
        self.assertEqual(stdout.strip(), "stdout")
        self.assertEqual(stderr.strip(), "stderr")
        self.assertEqual(returncode, 7)
        self.assertFalse(timed_out)

    def test_bounded_process_marks_timeout_and_terminates_the_session_group(self) -> None:
        with tempfile.TemporaryDirectory(prefix="bounded-timeout-") as temp_dir:
            _stdout, _stderr, returncode, timed_out = optimize._run_bounded(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cwd=Path(temp_dir),
                timeout=1,
                env=optimize.os.environ.copy(),
            )
        self.assertTrue(timed_out)
        self.assertNotEqual(returncode, 0)


class RuntimeHydrationCharacterizationTest(unittest.TestCase):
    def test_link_runtime_hydrates_all_three_backend_roots(self) -> None:
        with tempfile.TemporaryDirectory(prefix="runtime-hydration-") as temp_dir:
            root = Path(temp_dir) / "repo"
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            for common in ("tools", "reference", "reference-projects", "gpu-wiki"):
                (root / common).mkdir(parents=True, exist_ok=True)
            project_skill = root / "skills" / "gpu-kernel-profile-optimizer"
            project_skill.mkdir(parents=True)
            (project_skill / "SKILL.md").write_text("# optimizer\n", encoding="utf-8")
            (root / "agents").mkdir(parents=True)

            ncu = root / "3rdparty" / "ncu-report-skill"
            ncu.mkdir(parents=True)
            (ncu / "SKILL.md").write_text("# ncu\n", encoding="utf-8")
            kernel_wiki = root / "gpu-wiki" / "3rdparty" / "KernelWiki"
            kernel_wiki.mkdir(parents=True)
            (kernel_wiki / "SKILL.md").write_text("# wiki\n", encoding="utf-8")

            humanize = root / "3rdparty" / "humanize"
            runtime_skill = humanize / "skills" / "humanize"
            runtime_skill.mkdir(parents=True)
            (runtime_skill / "SKILL.md").write_text(
                "---\nname: humanize\n---\n", encoding="utf-8"
            )
            generator = humanize / "skills" / "humanize-gen-plan"
            generator.mkdir(parents=True)
            (generator / "SKILL.md").write_text(
                "---\nname: humanize-gen-plan\n---\n{{HUMANIZE_RUNTIME_ROOT}}\n",
                encoding="utf-8",
            )
            (humanize / "scripts").mkdir(parents=True)

            with (
                mock.patch.object(optimize, "REPO_ROOT", root),
                mock.patch.object(optimize, "HUMANIZE_DIR", humanize),
            ):
                optimize.link_runtime(workspace)

            for common in ("tools", "reference", "skills", "reference-projects", "gpu-wiki"):
                self.assertTrue((workspace / common).is_symlink(), common)
            for backend_root in (".claude", ".qoder"):
                self.assertTrue(
                    (workspace / backend_root / "skills" / "ncu-report-skill").is_symlink()
                )
                self.assertTrue(
                    (workspace / backend_root / "skills" / "KernelWiki").is_symlink()
                )
                self.assertTrue((workspace / backend_root / "agents").is_symlink())
            codex_skills = workspace / ".agents" / "skills"
            self.assertTrue(
                (codex_skills / "gpu-kernel-profile-optimizer").is_symlink()
            )
            hydrated = codex_skills / "humanize-gen-plan" / "SKILL.md"
            self.assertTrue(hydrated.is_file())
            self.assertIn(str(codex_skills / "humanize"), hydrated.read_text(encoding="utf-8"))
            gitignore = (workspace / ".gitignore").read_text(encoding="utf-8")
            for entry in ("/.claude", "/.qoder", "/.agents"):
                self.assertIn(entry, gitignore)


if __name__ == "__main__":
    unittest.main()
