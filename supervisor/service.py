from __future__ import annotations

import json
import threading

from .model_session import ActivationResult, SupervisorModelSession
from .store import CampaignStore


class SupervisorService:
    """Fixed-checkpoint scheduler around fresh strong-model Supervisor sessions.

    Raw executor events and trusted facts are observation data only. They never activate the
    strong Supervisor. Automatic activations happen only after a successful baseline, every
    configured number of completed logical iterations, and immediately before normal campaign
    exit. Manual review remains available as an explicit operator action.
    """

    def __init__(
        self,
        store: CampaignStore,
        model_session: SupervisorModelSession,
        every_iterations: int = 5,
        max_activations: int = 100,
        required: bool = False,
    ):
        self.store = store
        self.model_session = model_session
        self.every_iterations = max(1, every_iterations)
        self.max_activations = max(1, max_activations)
        self.required = required
        self._lock = threading.Lock()
        self._schedule_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_observed_cursor = self._read_last_observed_cursor()
        self._last_result: ActivationResult | None = None
        self._last_error = ""
        self._stop_review_requested = False
        self._token_lock = threading.Lock()
        self._unaccounted_tokens = 0

    @property
    def last_result(self) -> ActivationResult | None:
        return self._last_result

    def _read_last_observed_cursor(self) -> int:
        path = self.store.state_dir / "supervisor-state.json"
        try:
            return int(json.loads(path.read_text(encoding="utf-8")).get("last_observed_cursor") or 0)
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def _write_state(self, activation_active: bool | None = None) -> None:
        if activation_active is None:
            activation_active = bool(self._thread and self._thread.is_alive())
        path = self.store.state_dir / "supervisor-state.json"
        path.write_text(
            json.dumps(
                {
                    "last_observed_cursor": self._last_observed_cursor,
                    "last_error": self._last_error,
                    "activation_active": activation_active,
                    "activation_policy": {
                        "event_activation": False,
                        "after_baseline": True,
                        "every_completed_iterations": self.every_iterations,
                        "before_stop": True,
                        "manual": True,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _activation_count(self) -> int:
        return len(list(self.store.activations_dir.glob("activation-*")))

    def session_checkpoint(
        self,
        prompt_kind: str,
        exit_status: int,
        timed_out: bool,
    ) -> None:
        """Record one logical session and run only fixed automatic checkpoints.

        Non-zero exits, timeouts, restarts, stalls, and output contents intentionally do not
        activate the strong Supervisor. A failed baseline is not considered completed; all
        returned non-baseline sessions count as one logical iteration, regardless of outcome.
        """
        if prompt_kind == "baseline":
            if exit_status != 0 or timed_out:
                return
            with self._schedule_lock:
                state = self.store.schedule_state()
                if state["baseline_review_attempted"]:
                    return
                state["baseline_review_attempted"] = True
                self.store.write_schedule_state(state)
            self.request_activation("scheduled checkpoint: baseline completed", blocking=True)
            return

        with self._schedule_lock:
            state = self.store.schedule_state()
            completed = int(state["completed_iterations"]) + 1
            state["completed_iterations"] = completed
            due = completed % self.every_iterations == 0
            if due:
                state["last_periodic_review_iteration"] = completed
            self.store.write_schedule_state(state)
        if due:
            self.request_activation(
                "scheduled checkpoint: "
                f"{completed} completed logical iterations "
                f"({self.every_iterations} since the fixed cadence boundary)",
                blocking=True,
            )

    def before_stop(self, reason: str = "orchestrator returned") -> bool:
        with self._schedule_lock:
            if self._stop_review_requested:
                return True
            self._stop_review_requested = True
            state = self.store.schedule_state()
            state["stop_reviews"] = int(state["stop_reviews"]) + 1
            self.store.write_schedule_state(state)
        return self.request_activation(
            f"scheduled checkpoint: before campaign stop ({reason})",
            blocking=True,
        )

    def manual_review(self, reason: str) -> bool:
        with self._schedule_lock:
            state = self.store.schedule_state()
            state["manual_reviews"] = int(state["manual_reviews"]) + 1
            self.store.write_schedule_state(state)
        return self.request_activation(f"manual checkpoint: {reason}", blocking=True)

    def request_activation(self, reason: str, blocking: bool) -> bool:
        with self._lock:
            if self._activation_count() >= self.max_activations:
                self._last_error = "Supervisor activation cap reached"
                self._write_state()
                if blocking and self.required:
                    raise RuntimeError(self._last_error)
                return False
            if self._thread and self._thread.is_alive():
                thread = self._thread
            else:
                after_cursor = self._last_observed_cursor
                thread = threading.Thread(
                    target=self._activate,
                    args=(reason, after_cursor),
                    name=f"atrex-supervisor-{self.store.campaign_id}",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
        if not blocking:
            return True
        thread.join(timeout=self.model_session.config.timeout + 10)
        if thread.is_alive():
            self._last_error = "Supervisor activation did not finish before checkpoint timeout"
            self._write_state()
            if self.required:
                raise RuntimeError(self._last_error)
            return False
        success = self._last_result is not None and self._last_result.exit_status == 0
        if self.required and not success:
            raise RuntimeError(self._last_error or "Supervisor activation failed")
        return success

    def _activate(self, reason: str, after_cursor: int) -> None:
        try:
            result = self.model_session.activate(self.store, reason, after_cursor)
            self._last_result = result
            with self._token_lock:
                self._unaccounted_tokens += result.tokens
            # Advance only to the boundary handed to this activation. Later events stay visible
            # at the next fixed checkpoint, but cannot themselves trigger an activation.
            self._last_observed_cursor = max(self._last_observed_cursor, result.observed_cursor)
            if result.exit_status == 0:
                self._last_error = ""
            else:
                self._last_error = (
                    f"Supervisor activation failed exit={result.exit_status}: "
                    f"{result.stderr[-1000:]}"
                )
        except Exception as exc:
            self._last_error = f"Supervisor activation raised: {exc}"
            self._last_result = None
        finally:
            # This code still executes on the activation thread, so is_alive() remains true
            # until the function returns.  Persist the semantic post-activation state explicitly.
            self._write_state(activation_active=False)

    def take_unaccounted_tokens(self) -> int:
        with self._token_lock:
            tokens = self._unaccounted_tokens
            self._unaccounted_tokens = 0
            return tokens

    def close(self) -> None:
        with self._lock:
            thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=5)
