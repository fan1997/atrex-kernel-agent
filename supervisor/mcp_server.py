#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from supervisor.facts import (  # noqa: E402
    campaign_facts,
    inspect_environment,
    inspect_gateway_job,
    inspect_git,
    read_supervisor_resource,
    read_workspace_file,
)
from supervisor.models import ControlRequest  # noqa: E402
from supervisor.store import CampaignStore  # noqa: E402


PROTOCOL_VERSION = "2025-06-18"


def tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "tail_agent_events",
            "description": "Read raw executor conversation and tool events after a cursor.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "after_cursor": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "event_types": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        {
            "name": "wait_agent_events",
            "description": "Wait up to 30 seconds for raw executor events after a cursor.",
            "inputSchema": {
                "type": "object",
                "required": ["after_cursor"],
                "properties": {
                    "after_cursor": {"type": "integer", "minimum": 0},
                    "timeout_seconds": {"type": "integer", "minimum": 0, "maximum": 30},
                },
            },
        },
        {
            "name": "get_campaign_facts",
            "description": (
                "Get the Supervisor's campaign-wide view: controller facts, the AKA execution "
                "contract and prompt catalog, version trajectory, captured prompts, and prior "
                "guidance history. Use this first at every activation."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "inspect_git",
            "description": "Run a bounded read-only Git status, log, diff, or show query.",
            "inputSchema": {
                "type": "object",
                "required": ["operation"],
                "properties": {
                    "operation": {"type": "string", "enum": ["status", "log", "diff", "show"]},
                    "revision": {"type": "string"},
                    "path": {"type": "string"},
                    "max_count": {"type": "integer", "minimum": 1, "maximum": 100},
                },
            },
        },
        {
            "name": "read_workspace_file",
            "description": (
                "Read a bounded workspace file or a virtual execution-contract resource. Virtual "
                "paths include @executor/latest/base, @executor/latest/effective, captured session "
                "ids, and the @runtime/... entries returned by get_campaign_facts."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
            },
        },
        {
            "name": "inspect_gateway_job",
            "description": (
                "Inspect raw captured evidence for a gateway job, or run a fixed non-mutating "
                "environment diagnostic. Diagnostics are toolchain, workspace_layout, and syntax; "
                "syntax checks copy inputs into an isolated temporary directory first."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "diagnostic": {
                        "type": "string",
                        "enum": ["toolchain", "workspace_layout", "syntax"],
                    },
                    "paths": {
                        "type": "array",
                        "maxItems": 50,
                        "items": {"type": "string"},
                    },
                },
            },
        },
        {
            "name": "set_next_iteration_guidance",
            "description": (
                "Publish newest-wins standing campaign strategy for a bounded future horizon. "
                "Despite the compatibility name, this is not an iteration scheduler: do not assign "
                "directions to numbered iterations or replace the normal AKA cycle. Any older "
                "pending strategy is superseded."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["message"],
                "properties": {
                    "message": {"type": "string", "minLength": 1, "maxLength": 12000},
                    "valid_for_iterations": {"type": "integer", "minimum": 1, "maximum": 20},
                },
            },
        },
        {
            "name": "interrupt_and_restart",
            "description": (
                "Request a guarded executor interruption and fresh-session restart. The parent "
                "orchestrator validates the exact run id and remains the only process controller."
            ),
            "inputSchema": {
                "type": "object",
                "required": ["expected_run_id", "reason"],
                "properties": {
                    "expected_run_id": {"type": "string", "minLength": 1},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 8000},
                    "evidence_event_ids": {"type": "array", "items": {"type": "integer"}},
                    "guidance": {"type": "string", "maxLength": 12000},
                    "strategy": {
                        "type": "string",
                        "enum": ["fresh_context", "reprofile_first", "discard_uncommitted_attempt"],
                    },
                },
            },
        },
    ]


class ToolDispatcher:
    def __init__(self, store: CampaignStore, repository_root: Path | None = None):
        self.store = store
        self.repository_root = (
            repository_root.resolve()
            if repository_root is not None
            else Path(__file__).resolve().parent.parent
        )

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name == "tail_agent_events":
            return self.store.read_events(
                after_cursor=int(args.get("after_cursor") or 0),
                limit=int(args.get("limit") or 100),
                event_types=args.get("event_types") or (),
            )
        if name == "wait_agent_events":
            return self.store.wait_events(
                after_cursor=int(args.get("after_cursor") or 0),
                timeout_seconds=int(args.get("timeout_seconds") or 20),
            )
        if name == "get_campaign_facts":
            return campaign_facts(self.store, self.repository_root)
        if name == "inspect_git":
            return inspect_git(
                self.store.workspace,
                operation=str(args.get("operation") or ""),
                revision=str(args.get("revision") or "HEAD"),
                path=str(args.get("path") or ""),
                max_count=int(args.get("max_count") or 20),
            )
        if name == "read_workspace_file":
            path = str(args.get("path") or "")
            if path.startswith("@"):
                return read_supervisor_resource(
                    self.store,
                    self.repository_root,
                    resource=path,
                    start_line=int(args.get("start_line") or 1),
                    max_lines=int(args.get("max_lines") or 300),
                )
            return read_workspace_file(
                self.store.workspace,
                path=path,
                start_line=int(args.get("start_line") or 1),
                max_lines=int(args.get("max_lines") or 300),
            )
        if name == "inspect_gateway_job":
            diagnostic = str(args.get("diagnostic") or "").strip()
            if diagnostic:
                return inspect_environment(
                    self.store,
                    diagnostic=diagnostic,
                    paths=[str(path) for path in args.get("paths") or []],
                )
            return inspect_gateway_job(self.store, str(args.get("job_id") or ""))
        if name == "set_next_iteration_guidance":
            message = str(args.get("message") or "").strip()
            if not message:
                raise ValueError("message is required")
            path = self.store.queue_guidance(
                message=message,
                valid_for_iterations=int(args.get("valid_for_iterations") or 1),
            )
            return {
                "queued": True,
                "policy": (
                    "newest-wins standing campaign strategy with logical-iteration expiry; "
                    "not an iteration-by-iteration schedule"
                ),
                "path": str(path),
            }
        if name == "interrupt_and_restart":
            expected_run_id = str(args.get("expected_run_id") or "").strip()
            reason = str(args.get("reason") or "").strip()
            if not expected_run_id or not reason:
                raise ValueError("expected_run_id and reason are required")
            current = self.store.current_run()
            if current.get("run_id") != expected_run_id or current.get("status") != "running":
                raise ValueError("expected_run_id is stale or no longer running")
            request = ControlRequest(
                request_id=uuid.uuid4().hex,
                action="interrupt_and_restart",
                expected_run_id=expected_run_id,
                reason=reason,
                evidence_event_ids=tuple(int(x) for x in args.get("evidence_event_ids") or []),
                guidance=str(args.get("guidance") or ""),
                strategy=str(args.get("strategy") or "fresh_context"),
            )
            path = self.store.queue_control(request)
            return {"queued": True, "request_id": request.request_id, "path": str(path)}
        raise ValueError(f"unknown supervisor tool: {name}")


def _response(request_id: Any, result: Any = None, error: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve(store: CampaignStore, repository_root: Path | None = None) -> int:
    dispatcher = ToolDispatcher(store, repository_root)
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict):
            continue
        method = request.get("method")
        request_id = request.get("id")
        if request_id is None:
            continue
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": request.get("params", {}).get("protocolVersion") or PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "atrex-campaign-supervisor", "version": "0.1.0"},
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": tool_definitions()}
            elif method == "tools/call":
                params = request.get("params") or {}
                output = dispatcher.call(str(params.get("name") or ""), params.get("arguments") or {})
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(output, indent=2, ensure_ascii=False),
                        }
                    ],
                    "isError": False,
                }
            else:
                _response(request_id, error={"code": -32601, "message": f"method not found: {method}"})
                continue
            _response(request_id, result=result)
        except Exception as exc:  # tool errors must be visible to the supervisor, not crash stdio
            if method == "tools/call":
                _response(
                    request_id,
                    result={
                        "content": [{"type": "text", "text": f"Supervisor tool error: {exc}"}],
                        "isError": True,
                    },
                )
            else:
                _response(request_id, error={"code": -32000, "message": str(exc)})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atrex supervisor MCP stdio server")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parent.parent))
    args = parser.parse_args(argv)
    store = CampaignStore(Path(args.data_root), Path(args.workspace))
    return serve(store, Path(args.repository_root))


if __name__ == "__main__":
    raise SystemExit(main())
