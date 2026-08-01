from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import subprocess
import sys
import tempfile
import tarfile
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from supervisor.bridge import SupervisorConfig, SupervisorRuntime
from supervisor.cutlass_compat import MARKER, WorkspaceCompatibility
from supervisor.facts import read_workspace_file
from supervisor.mcp_server import ToolDispatcher, tool_definitions
from supervisor.model_session import ActivationResult, ModelSessionConfig, SupervisorModelSession
from supervisor.models import AgentEvent, ControlRequest
from supervisor.optimize import _parse_args, main as supervisor_main
from supervisor.runtime_adapter import install_supervised_runtime
from supervisor.service import SupervisorService
from supervisor.session_runner import _append_guidance, _stream_attempt, supervised_run_session
from supervisor.store import CampaignStore


def init_git(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(path), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(path), check=True)
    (path / "kernel.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "kernel.py"], cwd=str(path), check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=str(path), check=True)


class PromptContractTest(unittest.TestCase):
    def test_guidance_preserves_single_hypothesis_complete_bundle_contract(self) -> None:
        effective = _append_guidance("base prompt", "rank these campaign directions")

        self.assertIn("one locally evidence-supported measurable optimization hypothesis", effective)
        self.assertIn("complete, tightly coupled change bundle", effective)
        self.assertNotIn("choose only one locally evidence-supported optimization action", effective)


class StoreTest(unittest.TestCase):
    def test_events_guidance_and_guarded_control_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-store-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            event = store.append_event(
                AgentEvent(store.campaign_id, "run-1", "agent_output", raw="hello")
            )
            self.assertEqual(event.cursor, 1)
            self.assertEqual(store.read_events()[0]["raw"], "hello")

            store.queue_guidance("reprofile first")
            self.assertEqual(store.consume_guidance(1)["message"], "reprofile first")
            self.assertIsNone(store.consume_guidance(2))

            store.set_current_run({"run_id": "run-1", "status": "running"})
            request = ControlRequest(
                request_id=uuid.uuid4().hex,
                action="interrupt_and_restart",
                expected_run_id="run-1",
                reason="repeated invalid edit",
            )
            store.queue_control(request)
            self.assertIsNone(store.consume_control("run-2"))
            self.assertEqual(store.consume_control("run-1"), request)

    def test_tool_dispatcher_validates_stale_interrupt(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-tools-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            init_git(workspace)
            store = CampaignStore(root / "data", workspace)
            dispatcher = ToolDispatcher(store)
            store.set_current_run({"run_id": "active", "status": "running"})
            result = dispatcher.call(
                "interrupt_and_restart",
                {"expected_run_id": "active", "reason": "evidence-backed interrupt"},
            )
            self.assertTrue(result["queued"])
            with self.assertRaisesRegex(ValueError, "stale"):
                dispatcher.call(
                    "interrupt_and_restart",
                    {"expected_run_id": "old", "reason": "stale request"},
                )

    def test_guidance_is_newest_wins_and_iteration_scoped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-guidance-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            first = store.queue_guidance("old plan", valid_for_iterations=3)
            second = store.queue_guidance("new plan", valid_for_iterations=2)

            self.assertFalse(first.exists())
            self.assertTrue((store.guidance_superseded / first.name).exists())
            self.assertEqual(store.consume_guidance(1)["message"], "new plan")
            self.assertTrue(second.exists())
            self.assertEqual(store.consume_guidance(2)["message"], "new plan")
            self.assertFalse(second.exists())
            self.assertIsNone(store.consume_guidance(3))

    def test_legacy_unscoped_guidance_is_not_replayed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-guidance-legacy-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            legacy = store.guidance_pending / "legacy.json"
            legacy.write_text(json.dumps({"message": "stale plan"}), encoding="utf-8")

            self.assertIsNone(store.consume_guidance(1))
            self.assertTrue((store.guidance_superseded / legacy.name).exists())

    def test_guidance_created_during_active_iteration_targets_following_iteration(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-guidance-active-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            store.write_schedule_state(
                {
                    "baseline_review_attempted": True,
                    "completed_iterations": 4,
                    "last_periodic_review_iteration": 0,
                    "manual_reviews": 0,
                    "stop_reviews": 0,
                }
            )
            store.set_current_run(
                {
                    "run_id": "active",
                    "status": "running",
                    "prompt_kind": "iteration",
                    "controller_pid": os.getpid(),
                }
            )
            store.queue_guidance("after the active iteration")

            self.assertIsNone(store.consume_guidance(5))
            self.assertEqual(
                store.consume_guidance(6)["message"],
                "after the active iteration",
            )

    def test_workspace_read_cannot_escape_campaign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-read-") as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            (workspace / "inside.txt").write_text("one\ntwo\n", encoding="utf-8")
            self.assertEqual(
                read_workspace_file(workspace, "inside.txt", start_line=2)["text"],
                "two",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                read_workspace_file(workspace, "../outside.txt")

    def test_mcp_tool_set_is_the_bounded_v1_surface(self) -> None:
        names = {tool["name"] for tool in tool_definitions()}
        self.assertEqual(
            names,
            {
                "tail_agent_events",
                "wait_agent_events",
                "get_campaign_facts",
                "inspect_git",
                "read_workspace_file",
                "inspect_gateway_job",
                "set_next_iteration_guidance",
                "interrupt_and_restart",
            },
        )

    def test_mcp_stdio_server_negotiates_and_lists_tools(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-mcp-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            requests = "\n".join(
                [
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "initialize",
                            "params": {"protocolVersion": "2025-06-18"},
                        }
                    ),
                    json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
                ]
            ) + "\n"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent.parent / "supervisor" / "mcp_server.py"),
                    "--data-root", str(root / "data"),
                    "--workspace", str(workspace),
                ],
                input=requests,
                capture_output=True,
                text=True,
                timeout=10,
            )
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(result.returncode, 0)
        self.assertEqual(responses[0]["result"]["serverInfo"]["name"], "atrex-campaign-supervisor")
        self.assertEqual(len(responses[1]["result"]["tools"]), 8)


class FakeBridge:
    def __init__(self, request: ControlRequest | None = None):
        self.request = request
        self.events = []

    def poll_control(self, run_id: str):
        request, self.request = self.request, None
        return request

    def publish(self, run_id, kind, raw="", stream="control", metadata=None):
        self.events.append((run_id, kind, raw, stream, metadata or {}))


class StreamingTest(unittest.TestCase):
    def test_stream_attempt_forwards_raw_output(self) -> None:
        bridge = FakeBridge()
        with tempfile.TemporaryDirectory(prefix="supervisor-stream-") as temp_dir:
            result = _stream_attempt(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('stdout-line', flush=True); "
                    "print('stderr-line', file=sys.stderr, flush=True)",
                ],
                Path(temp_dir),
                10,
                {},
                bridge,
                "run-1",
            )
        self.assertEqual(result.exit_status, 0)
        self.assertIn("stdout-line", result.stdout)
        self.assertIn("stderr-line", result.stderr)
        self.assertTrue(any(event[2] == "stdout-line" for event in bridge.events))

    def test_stream_attempt_accepts_only_exact_run_control(self) -> None:
        request = ControlRequest(
            request_id="request-1",
            action="interrupt_and_restart",
            expected_run_id="run-1",
            reason="stop looping",
        )
        bridge = FakeBridge(request)
        with tempfile.TemporaryDirectory(prefix="supervisor-interrupt-") as temp_dir:
            result = _stream_attempt(
                [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(10)"],
                Path(temp_dir),
                20,
                {},
                bridge,
                "run-1",
            )
        self.assertTrue(result.interrupted)
        self.assertEqual(result.control_request, request)
        self.assertNotEqual(result.exit_status, 0)


class FakeSupervisorModelSession:
    def __init__(self):
        self.config = SimpleNamespace(timeout=2)
        self.reasons: list[str] = []

    def activate(self, store, reason, after_cursor):
        self.reasons.append(reason)
        activation_dir = store.create_activation_dir()
        return ActivationResult(
            activation_dir=activation_dir,
            exit_status=0,
            timed_out=False,
            stdout="",
            stderr="",
            observed_cursor=store.cursor,
            tokens=0,
        )


class FixedScheduleTest(unittest.TestCase):
    def test_cli_defaults_to_five_and_rejects_event_trigger_option(self) -> None:
        args, optimize_args = _parse_args([])
        self.assertEqual(args.supervisor_every_iterations, 5)
        self.assertEqual(optimize_args, [])
        with self.assertRaisesRegex(SystemExit, "never event-triggered"):
            supervisor_main(["--supervisor-every-events", "50"])

    def test_only_baseline_five_iterations_stop_and_manual_activate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-schedule-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            model = FakeSupervisorModelSession()
            service = SupervisorService(store, model, every_iterations=5)

            # A failed baseline and arbitrary risky-looking observation data are facts only.
            store.append_event(
                AgentEvent(
                    store.campaign_id,
                    "run-1",
                    "agent_output",
                    raw='timeout correctness failure traceback all_pass": false',
                )
            )
            service.session_checkpoint("baseline", exit_status=1, timed_out=False)
            self.assertEqual(model.reasons, [])

            service.session_checkpoint("baseline", exit_status=0, timed_out=False)
            service.session_checkpoint("baseline", exit_status=0, timed_out=False)
            self.assertEqual(len(model.reasons), 1)
            self.assertIn("baseline completed", model.reasons[0])

            # Outcomes do not cause early escalation. Each returned non-baseline session is one
            # logical iteration; only the fifth boundary activates the strong Supervisor.
            outcomes = [(0, False), (1, False), (124, True), (0, False), (1, False)]
            for index, (exit_status, timed_out) in enumerate(outcomes, start=1):
                service.session_checkpoint("iteration", exit_status, timed_out)
                self.assertEqual(len(model.reasons), 1 if index < 5 else 2)
            self.assertIn("5 completed logical iterations", model.reasons[1])
            self.assertEqual(store.schedule_state()["completed_iterations"], 5)

            self.assertTrue(service.before_stop("test stop"))
            self.assertTrue(service.before_stop("duplicate stop"))
            self.assertEqual(len(model.reasons), 3)
            self.assertIn("before campaign stop", model.reasons[2])

            self.assertTrue(service.manual_review("human requested review"))
            self.assertEqual(len(model.reasons), 4)
            self.assertIn("manual checkpoint", model.reasons[3])

    def test_ten_iterations_produce_exactly_two_periodic_reviews(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-cadence-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            model = FakeSupervisorModelSession()
            service = SupervisorService(store, model, every_iterations=5)

            for _ in range(10):
                service.session_checkpoint("iteration", exit_status=0, timed_out=False)

            self.assertEqual(len(model.reasons), 2)
            self.assertIn("5 completed", model.reasons[0])
            self.assertIn("10 completed", model.reasons[1])

    def test_finished_activation_persists_inactive_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-active-state-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            service = SupervisorService(store, FakeSupervisorModelSession(), every_iterations=5)

            self.assertTrue(service.manual_review("state check"))
            state = json.loads(
                (store.state_dir / "supervisor-state.json").read_text(encoding="utf-8")
            )
            self.assertFalse(state["activation_active"])

    def test_iteration_cadence_persists_across_controller_restart(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-cadence-resume-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            store = CampaignStore(root / "data", workspace)
            first_model = FakeSupervisorModelSession()
            first_service = SupervisorService(store, first_model, every_iterations=5)
            for _ in range(4):
                first_service.session_checkpoint("iteration", 0, False)
            self.assertEqual(first_model.reasons, [])

            resumed_store = CampaignStore(root / "data", workspace)
            resumed_model = FakeSupervisorModelSession()
            resumed_service = SupervisorService(resumed_store, resumed_model, every_iterations=5)
            resumed_service.session_checkpoint("iteration", 0, False)

            self.assertEqual(len(resumed_model.reasons), 1)
            self.assertIn("5 completed", resumed_model.reasons[0])

    def test_resumed_passed_v0_bootstraps_baseline_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-baseline-resume-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            (workspace / "memory").mkdir(parents=True)
            (workspace / "memory" / "v0.json").write_text(
                json.dumps(
                    {
                        "correctness": {"status": "PASS"},
                        "quality_gate": {"result": "PASS"},
                    }
                ),
                encoding="utf-8",
            )
            runtime = SupervisorRuntime(
                SupervisorConfig(
                    data_root=root / "data",
                    repository_root=Path(__file__).resolve().parent.parent,
                    cli="",
                )
            )
            bridge = runtime.bridge_for(workspace)

            self.assertTrue(bridge.confirm_existing_baseline())
            self.assertTrue(bridge.confirm_existing_baseline())
            self.assertEqual(bridge.store.schedule_state()["baseline_review_attempted"], True)
            self.assertEqual(len(list(bridge.store.activations_dir.glob("activation-*"))), 1)
            runtime.close()


class CutlassCompatibilityTest(unittest.TestCase):
    def test_workspace_shim_is_hidden_and_removed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="supervisor-compat-") as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            init_git(workspace)
            compatibility = WorkspaceCompatibility(workspace)

            compatibility.install()
            self.assertIn(MARKER, (workspace / "sitecustomize.py").read_text(encoding="utf-8"))
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(status.stdout, "")

            compatibility.remove()
            self.assertFalse((workspace / "sitecustomize.py").exists())
            self.assertNotIn(
                "/sitecustomize.py",
                (workspace / ".git" / "info" / "exclude").read_text(encoding="utf-8"),
            )

    def test_workspace_shim_is_included_in_selective_evaluator_bundle(self) -> None:
        from tools import sandbox as sandbox_tool

        with tempfile.TemporaryDirectory(prefix="supervisor-compat-bundle-") as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            init_git(workspace)
            compatibility = WorkspaceCompatibility(workspace)
            compatibility.install()

            selected = sandbox_tool._evaluation_input_paths(workspace)
            bundle, _count, _skipped = sandbox_tool._make_input_bundle(
                workspace,
                max_file_bytes=1024 * 1024,
                input_paths=selected,
            )
            with tarfile.open(
                fileobj=io.BytesIO(base64.b64decode(bundle)), mode="r:gz"
            ) as archive:
                names = set(archive.getnames())

            self.assertIn("sitecustomize.py", selected)
            self.assertIn("sitecustomize.py", names)
            compatibility.remove()


class RuntimeAdapterTest(unittest.TestCase):
    def test_workload_bucket_campaigns_get_distinct_supervisor_bridges(self) -> None:
        from orchestrator import optimize as base_optimize

        with tempfile.TemporaryDirectory(prefix="supervisor-workload-buckets-") as temp_dir:
            root = Path(temp_dir)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "reference.py").write_text("def run(*args): return args\n", encoding="utf-8")
            workload_lines = ('{"shape": [1]}', '{"shape": [2]}')
            (op_dir / "workload.jsonl").write_text(
                "\n".join(workload_lines) + "\n", encoding="utf-8"
            )
            aggregate = base_optimize.Campaign(
                name="aggregate",
                kernel_demo=str(op_dir / "reference.py"),
                platform="H20",
                framework="Triton",
                work_dir=str(root / "runs"),
                max_iters=1,
            )
            coordinator = base_optimize.WorkloadBucketCoordinator(aggregate, op_dir)
            source = base_optimize.WorkloadSource(
                kind="sol",
                filename="workload.jsonl",
                ids=("0", "1"),
                entries=({"shape": [1]}, {"shape": [2]}),
                raw_lines=workload_lines,
            )
            buckets = (
                base_optimize.WorkloadBucket("small", (0,)),
                base_optimize.WorkloadBucket("large", (1,)),
            )
            runtime = SupervisorRuntime(
                SupervisorConfig(
                    data_root=root / "supervisor-data",
                    repository_root=Path(__file__).resolve().parent.parent,
                    cli="",
                )
            )
            original_run_session = base_optimize.run_session
            original_finish = base_optimize.Campaign._finish

            with install_supervised_runtime(base_optimize, runtime):
                campaigns = [
                    coordinator._make_bucket_campaign(bucket, source) for bucket in buckets
                ]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    bridges = list(executor.map(runtime.bridge_for, [c.workspace for c in campaigns]))

                self.assertIsNot(base_optimize.run_session, original_run_session)
                self.assertIsNot(base_optimize.Campaign._finish, original_finish)
                self.assertNotEqual(campaigns[0].workspace, campaigns[1].workspace)
                self.assertNotEqual(bridges[0].campaign_id, bridges[1].campaign_id)
                self.assertNotEqual(bridges[0].store.campaign_dir, bridges[1].store.campaign_dir)

            self.assertIs(base_optimize.run_session, original_run_session)
            self.assertIs(base_optimize.Campaign._finish, original_finish)

    def test_campaign_finish_runs_supervisor_before_original_stop_and_is_restored(self) -> None:
        order: list[str] = []

        class Base:
            __file__ = "/repo/orchestrator/optimize.py"

            class Campaign:
                def __init__(self):
                    self.workspace = Path("/tmp/campaign")

                def _finish(self, reason):
                    order.append(f"base:{reason}")
                    return reason

            @staticmethod
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
                return "original"

            _session_command = object()
            _session_env = object()
            _tokens_from_stream = object()
            SessionResult = object()

        bridge = SimpleNamespace(
            before_stop=lambda reason: order.append(f"supervisor:{reason}") or True
        )
        runtime = SimpleNamespace(
            config=SimpleNamespace(
                data_root=Path("/tmp/supervisor-data"),
                repository_root=Path(__file__).resolve().parent.parent,
                cli="codex",
                model="test-model",
                reasoning_effort="low",
                settings="",
                activation_timeout=30,
                every_iterations=5,
                max_activations=10,
                max_restarts_per_session=1,
                required=False,
            ),
            bridge_for=lambda workspace: bridge,
            close=mock.Mock(),
        )
        original_finish = Base.Campaign._finish

        with install_supervised_runtime(Base, runtime):
            self.assertEqual(Base.Campaign()._finish("budget: max-iters"), "budget: max-iters")

        self.assertEqual(
            order,
            [
                "supervisor:base orchestrator is about to stop: budget: max-iters",
                "base:budget: max-iters",
            ],
        )
        self.assertIs(Base.Campaign._finish, original_finish)

    def test_installation_is_process_local_and_restored(self) -> None:
        class Base:
            @staticmethod
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
                return "original"

            _session_command = object()
            _session_env = object()
            _tokens_from_stream = object()
            SessionResult = object()

        original = Base.run_session
        runtime = SimpleNamespace(
            config=SimpleNamespace(
                data_root=Path("/tmp/supervisor-data"),
                repository_root=Path(__file__).resolve().parent.parent,
                cli="codex",
                model="test-model",
                reasoning_effort="low",
                settings="",
                activation_timeout=30,
                every_iterations=5,
                max_activations=10,
                max_restarts_per_session=1,
                required=False,
            ),
            close=mock.Mock(),
        )
        with install_supervised_runtime(Base, runtime):
            self.assertIsNot(Base.run_session, original)
        self.assertIs(Base.run_session, original)
        runtime.close.assert_called_once_with()

    def test_auto_dispatch_children_use_supervisor_entry_and_inherit_config(self) -> None:
        class Base:
            __file__ = "/repo/orchestrator/optimize.py"
            seen_file = ""
            seen_model = ""

            @staticmethod
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
                return "original"

            @classmethod
            def dispatch_framework_campaigns(cls, *args, **kwargs):
                cls.seen_file = cls.__file__
                cls.seen_model = os.environ.get("ATREX_SUPERVISOR_MODEL", "")
                return 0

            _session_command = object()
            _session_env = object()
            _tokens_from_stream = object()
            SessionResult = object()

        runtime = SimpleNamespace(
            config=SimpleNamespace(
                data_root=Path("/tmp/supervisor-data"),
                repository_root=Path(__file__).resolve().parent.parent,
                cli="codex",
                model="supervisor-model",
                reasoning_effort="high",
                settings="",
                activation_timeout=30,
                every_iterations=5,
                max_activations=10,
                max_restarts_per_session=1,
                required=False,
            ),
            close=mock.Mock(),
        )
        original_file = Base.__file__
        with install_supervised_runtime(Base, runtime):
            self.assertEqual(Base.dispatch_framework_campaigns(), 0)
        self.assertTrue(Base.seen_file.endswith("supervisor/optimize.py"))
        self.assertEqual(Base.seen_model, "supervisor-model")
        self.assertEqual(Base.__file__, original_file)

    def test_supervised_session_preserves_base_result_contract_and_tokens(self) -> None:
        @dataclass
        class Result:
            exit_status: int
            timed_out: bool
            tokens: int
            stdout_tail: str
            stderr_tail: str

        class Base:
            SessionResult = Result

            @staticmethod
            def _session_env(agent_cli):
                return {}

            @staticmethod
            def _session_command(agent_cli, prompt, session_id):
                payload = json.dumps(
                    {"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 2}}
                )
                return [sys.executable, "-c", f"print({payload!r})"]

            @staticmethod
            def _tokens_from_stream(stdout):
                event = json.loads(stdout.strip())
                return event["usage"]["input_tokens"] + event["usage"]["output_tokens"]

        with tempfile.TemporaryDirectory(prefix="supervisor-session-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            init_git(workspace)
            runtime = SupervisorRuntime(
                SupervisorConfig(
                    data_root=root / "data",
                    repository_root=Path(__file__).resolve().parent.parent,
                    cli="",
                    every_iterations=10_000,
                )
            )
            result = supervised_run_session(
                Base,
                runtime,
                workspace,
                "ordinary iteration",
                timeout=10,
                agent_cli="codex",
            )
            store = runtime.bridge_for(workspace).store
            events = store.read_events(limit=100)
            runtime.close()

        self.assertEqual(result.exit_status, 0)
        self.assertEqual(result.tokens, 9)
        self.assertTrue(any(event["kind"] == "agent_output" for event in events))

    def test_supervisor_interrupt_restarts_inside_one_base_iteration(self) -> None:
        @dataclass
        class Result:
            exit_status: int
            timed_out: bool
            tokens: int
            stdout_tail: str
            stderr_tail: str

        class Base:
            SessionResult = Result
            command_count = 0

            @staticmethod
            def _session_env(agent_cli):
                return {}

            @classmethod
            def _session_command(cls, agent_cli, prompt, session_id):
                cls.command_count += 1
                if cls.command_count == 1:
                    return [sys.executable, "-c", "import time; time.sleep(10)"]
                payload = json.dumps(
                    {"type": "turn.completed", "usage": {"input_tokens": 4, "output_tokens": 3}}
                )
                return [sys.executable, "-c", f"print({payload!r})"]

            @staticmethod
            def _tokens_from_stream(stdout):
                total = 0
                for line in stdout.splitlines():
                    event = json.loads(line)
                    usage = event.get("usage") or {}
                    total += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                return total

        class RestartBridge:
            campaign_id = "campaign"

            def __init__(self):
                self.current_run = ""
                self.sent = False
                self.checkpoint_prompt_kind = None

            def begin_run(self, run_id, agent_cli, attempt, prompt_kind):
                self.current_run = run_id

            def end_run(self, *args, **kwargs):
                pass

            def publish(self, *args, **kwargs):
                pass

            def take_guidance(self):
                return ""

            def poll_control(self, run_id):
                if not self.sent:
                    self.sent = True
                    return ControlRequest(
                        request_id="request-1",
                        action="interrupt_and_restart",
                        expected_run_id=run_id,
                        reason="restart with fresh context",
                    )
                return None

            def checkpoint(self, prompt_kind, exit_status, timed_out):
                self.checkpoint_prompt_kind = prompt_kind

            def take_supervisor_tokens(self):
                return 0

        with tempfile.TemporaryDirectory(prefix="supervisor-restart-") as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            workspace.mkdir()
            init_git(workspace)
            bridge = RestartBridge()
            runtime = SimpleNamespace(
                config=SimpleNamespace(max_restarts_per_session=1),
                bridge_for=lambda ignored: bridge,
            )
            result = supervised_run_session(
                Base,
                runtime,
                workspace,
                "ordinary iteration",
                timeout=20,
                agent_cli="codex",
            )

        self.assertEqual(Base.command_count, 2)
        self.assertEqual(result.exit_status, 0)
        self.assertEqual(result.tokens, 7)
        self.assertEqual(bridge.checkpoint_prompt_kind, "iteration")


class ModelSessionTest(unittest.TestCase):
    def test_codex_supervisor_is_read_only_ephemeral_and_uses_bounded_mcp(self) -> None:
        completed = subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                '{"type":"turn.completed","usage":'
                '{"input_tokens":11,"cached_input_tokens":5,"output_tokens":3}}\n'
            ),
            stderr="",
        )
        with tempfile.TemporaryDirectory(prefix="supervisor-model-") as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            init_git(workspace)
            store = CampaignStore(root / "data", workspace)
            store.append_event(
                AgentEvent(store.campaign_id, "run-1", "agent_output", raw="inspect me")
            )
            session = SupervisorModelSession(
                ModelSessionConfig(cli="codex", model="gpt-5.6-sol", timeout=30),
                repository_root=Path(__file__).resolve().parent.parent,
            )
            with (
                mock.patch("supervisor.model_session.shutil.which", return_value="/bin/codex"),
                mock.patch("supervisor.model_session.subprocess.run", return_value=completed) as run,
            ):
                result = session.activate(store, "periodic review", after_cursor=0)

        cmd = run.call_args.args[0]
        self.assertEqual(result.exit_status, 0)
        self.assertEqual(result.tokens, 14)
        self.assertIn("--ephemeral", cmd)
        self.assertIn("--ignore-user-config", cmd)
        self.assertEqual(cmd[cmd.index("--sandbox") + 1], "read-only")
        disabled = [cmd[index + 1] for index, item in enumerate(cmd[:-1]) if item == "--disable"]
        self.assertTrue(
            {"shell_tool", "unified_exec", "hooks", "plugins", "apps", "multi_agent"}.issubset(
                set(disabled)
            )
        )
        configs = [cmd[index + 1] for index, item in enumerate(cmd[:-1]) if item == "-c"]
        self.assertTrue(any(item.startswith("mcp_servers.atrex_supervisor.command=") for item in configs))
        self.assertTrue(any(item.startswith("mcp_servers.atrex_supervisor.args=") for item in configs))
        self.assertIn(
            'mcp_servers.atrex_supervisor.default_tools_approval_mode="approve"',
            configs,
        )
        self.assertNotIn("--dangerously-bypass-approvals-and-sandbox", cmd)


if __name__ == "__main__":
    unittest.main()
