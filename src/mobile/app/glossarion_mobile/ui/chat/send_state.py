"""Send/Stop button state machine (UI_SPEC §2.4). Pure Python, no Flet import.

States: ``idle_empty`` · ``idle_ready`` · ``queue`` · ``blocked`` · ``running`` ·
``finishing`` · ``stopping``. ``derive_state(SendInputs)`` maps what the
composer and the job service report to a state; ``SendStopMachine`` adds the
tap and long-press behaviour, including the optimistic running -> finishing /
stopping step that happens before JobService confirms the stop.

Precedence (first match wins):
1. this chat's job: RUNNING/STARTING -> running, STOPPING -> finishing,
   FORCE_STOPPING -> stopping (terminal job states fall through);
2. a blocking reason (engine not ready, glossary decision pending, manual
   glossary missing, attachment missing, excluded route, no key, no ChatGPT
   sign-in for an ``authgpt/`` model such as the default ``authgpt/gpt-6-luna``)
   -> blocked, even with an empty composer, so the reason and its fix action are
   visible up front (§4.17: Send "stays in the blocked state" until sign-in);
3. nothing to send -> idle_empty;
4. another chat's or a book's job running -> queue;
5. otherwise idle_ready.

The button never reads ``os.environ``; it only observes app state (§2.4).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, NamedTuple, Optional

__all__ = [
    "BLOCK_ENGINE_FAILED",
    "BLOCK_ENGINE_NOT_READY",
    "BLOCK_GLOSSARY_PENDING",
    "BLOCK_MANUAL_GLOSSARY",
    "BLOCK_ATTACHMENT_MISSING",
    "BlockReason",
    "DEFAULT_MODEL",
    "SendAction",
    "SendInputs",
    "SendState",
    "SendStopMachine",
    "SendVisual",
    "chatgpt_sign_in_reason",
    "derive_state",
    "excluded_route_reason",
    "no_key_reason",
    "requires_chatgpt_sign_in",
    "status_caption",
    "visual_for",
]

# Desktop default model (translator_gui: self.config.get('model', 'authgpt/gpt-6-luna')).
DEFAULT_MODEL = "authgpt/gpt-6-luna"
DEFAULT_MODEL_NAME = "GPT-6 Luna"


class SendState(str, Enum):
    IDLE_EMPTY = "idle_empty"
    IDLE_READY = "idle_ready"
    QUEUE = "queue"
    BLOCKED = "blocked"
    RUNNING = "running"
    FINISHING = "finishing"
    STOPPING = "stopping"


class SendAction(str, Enum):
    NONE = "none"
    SEND = "send"
    QUEUE = "queue"
    EXPLAIN_BLOCK = "explain_block"
    STOP = "stop"  # request_stop(): graceful when configured
    FORCE_STOP = "force_stop"
    # long-press menu items
    SEND_ONCE_WITH_MODEL = "send_once_with_model"
    ADD_WITHOUT_TRANSLATING = "add_without_translating"
    SEND_AS_SCRATCH = "send_as_scratch"
    STOP_CURRENT_AND_SEND = "stop_current_and_send"


@dataclass(frozen=True)
class BlockReason:
    """Why Send cannot run, with the fix action(s) the caption/snackbar offers."""

    code: str
    message: str
    fix_label: Optional[str] = None
    fix_action: Optional[str] = None
    secondary_label: Optional[str] = None
    secondary_action: Optional[str] = None


BLOCK_ENGINE_NOT_READY = BlockReason("engine_not_ready", "Preparing engine…")
BLOCK_ENGINE_FAILED = BlockReason(
    "engine_failed", "The translation engine failed to load", "View diagnostics", "open_diagnostics"
)
BLOCK_GLOSSARY_PENDING = BlockReason("glossary_pending", "Glossary ready — choose Edit, Yes, or No")
BLOCK_MANUAL_GLOSSARY = BlockReason("manual_glossary", "Manual glossary required", "Provide glossary", "provide_glossary")
BLOCK_ATTACHMENT_MISSING = BlockReason("attachment_missing", "Attachment missing", "Remove attachment", "remove_attachment")


def requires_chatgpt_sign_in(model: Optional[str]) -> bool:
    """``authgpt/…`` and numbered account routes (``authgpt2/…``) need a ChatGPT login."""
    prefix = str(model or "").split("/", 1)[0].lower()
    return prefix.startswith("authgpt") and (prefix == "authgpt" or prefix[7:].isdigit())


def chatgpt_sign_in_reason(model: Optional[str] = DEFAULT_MODEL) -> BlockReason:
    name = DEFAULT_MODEL_NAME if (model or DEFAULT_MODEL) == DEFAULT_MODEL else str(model)
    return BlockReason(
        "chatgpt_sign_in",
        f"Sign in with ChatGPT to use {name}",
        "Sign in with ChatGPT",
        "sign_in_chatgpt",
        "Choose another model",
        "choose_model",
    )


def excluded_route_reason(model: str) -> BlockReason:
    route = str(model).split("/", 1)[0] + "/"
    return BlockReason("excluded_route", f"{route} isn't available on mobile", "Choose model", "choose_model")


def no_key_reason(provider: str) -> BlockReason:
    return BlockReason("no_key", f"No API key for {provider}", "Add key", "add_key", "Choose model", "choose_model")


_OWN_JOB_STATES = {
    "STARTING": SendState.RUNNING,
    "RUNNING": SendState.RUNNING,
    "STOPPING": SendState.FINISHING,
    "FORCE_STOPPING": SendState.STOPPING,
}


@dataclass(frozen=True)
class SendInputs:
    has_content: bool = False  # text, a pasted-text chip or an attachment
    block: Optional[BlockReason] = None
    other_job_running: bool = False
    other_job_title: str = ""
    own_job_state: Optional[str] = None  # JobSnapshot.state of this chat's job (JobState name)
    graceful_stop: bool = True  # request_stop() waits for in-flight chunks (desktop graceful stop)


def derive_state(inputs: SendInputs) -> SendState:
    own = _OWN_JOB_STATES.get(str(inputs.own_job_state or "").upper())
    if own is not None:
        return own
    if inputs.block is not None:
        return SendState.BLOCKED
    if not inputs.has_content:
        return SendState.IDLE_EMPTY
    if inputs.other_job_running:
        return SendState.QUEUE
    return SendState.IDLE_READY


class SendVisual(NamedTuple):
    icon: Optional[str]  # Material icon name; None for the progress ring
    style: str  # muted | filled | tonal | error | warning_ring | progress
    tooltip: str
    enabled: bool  # False only for `stopping`; `blocked` stays tappable (§2.4 notes)


_VISUALS = {
    SendState.IDLE_EMPTY: SendVisual("ARROW_UPWARD", "muted", "Send", True),
    SendState.IDLE_READY: SendVisual("ARROW_UPWARD", "filled", "Send text for translation", True),
    SendState.QUEUE: SendVisual("SCHEDULE_SEND", "tonal", "Queue: runs after the current job", True),
    SendState.BLOCKED: SendVisual("ARROW_UPWARD", "muted", "Send", True),
    SendState.RUNNING: SendVisual("STOP", "error", "Stop translation", True),
    SendState.FINISHING: SendVisual(
        "HOURGLASS_BOTTOM", "warning_ring", "Graceful stop requested. Tap again to force stop.", True
    ),
    SendState.STOPPING: SendVisual(None, "progress", "Force stop requested", False),
}


def visual_for(state: SendState, block: Optional[BlockReason] = None) -> SendVisual:
    visual = _VISUALS[state]
    if state is SendState.BLOCKED and block is not None:
        return visual._replace(tooltip=block.message)
    return visual


def status_caption(state: SendState, inputs: SendInputs) -> Optional[str]:
    """One-line caption above the composer; ``None`` when the state is Ready (§2.3).

    Desktop footer strings, with "Click" -> "Tap".
    """
    if state is SendState.BLOCKED and inputs.block is not None:
        return inputs.block.message
    if state is SendState.RUNNING:
        return "Translating…"
    if state is SendState.FINISHING:
        return "Finishing current request… Tap again to force stop"
    if state is SendState.STOPPING:
        return "Force stopping…"
    if state is SendState.QUEUE:
        title = inputs.other_job_title or "the current job"
        return f"Send queues this message · runs after {title}"
    return None


_LONG_PRESS = {
    SendState.IDLE_READY: (
        (SendAction.SEND_ONCE_WITH_MODEL, "Translate once with another model…"),
        (SendAction.ADD_WITHOUT_TRANSLATING, "Add without translating"),
        (SendAction.SEND_AS_SCRATCH, "Send as scratch"),
    ),
    SendState.QUEUE: (
        (SendAction.QUEUE, "Queue"),
        (SendAction.STOP_CURRENT_AND_SEND, "Stop current & send"),
    ),
    SendState.RUNNING: ((SendAction.FORCE_STOP, "Force stop now"),),
    SendState.FINISHING: ((SendAction.FORCE_STOP, "Force stop now"),),
}


@dataclass
class SendStopMachine:
    """Tap / long-press behaviour on top of ``derive_state``.

    ``apply(inputs)`` re-derives the state from app signals. A stop tap moves
    the button ahead of JobService (running -> finishing or stopping); that
    local step is kept until the job state itself moves on.
    """

    inputs: SendInputs = field(default_factory=SendInputs)
    clock: Callable[[], float] = time.monotonic
    _local: Optional[SendState] = field(default=None, init=False, repr=False)
    _local_for_job_state: Optional[str] = field(default=None, init=False, repr=False)
    finishing_since: Optional[float] = field(default=None, init=False)

    FORCE_WINDOW_S = 2.0  # desktop "click again quickly": both inside and after it a tap forces

    @property
    def state(self) -> SendState:
        derived = derive_state(self.inputs)
        if self._local is not None and derived is SendState.RUNNING:
            return self._local
        return derived

    @property
    def visual(self) -> SendVisual:
        return visual_for(self.state, self.inputs.block)

    @property
    def caption(self) -> Optional[str]:
        return status_caption(self.state, self.inputs)

    def apply(self, inputs: SendInputs) -> bool:
        """Replace the inputs; returns True when the visible state changed."""
        before = self.state
        job_state = str(inputs.own_job_state or "").upper()
        if self._local is not None and job_state != self._local_for_job_state:
            self._local = None  # JobService caught up (or the job ended)
            self._local_for_job_state = None
        self.inputs = inputs
        after = self.state
        if after is SendState.FINISHING and before is not SendState.FINISHING:
            self.finishing_since = self.clock()
        elif after is not SendState.FINISHING:
            self.finishing_since = None
        return after is not before

    def _go_local(self, state: SendState) -> None:
        self._local = state
        self._local_for_job_state = str(self.inputs.own_job_state or "").upper()
        self.finishing_since = self.clock() if state is SendState.FINISHING else None

    def tap(self) -> SendAction:
        state = self.state
        if state is SendState.IDLE_READY:
            return SendAction.SEND
        if state is SendState.QUEUE:
            return SendAction.QUEUE
        if state is SendState.BLOCKED:
            return SendAction.EXPLAIN_BLOCK
        if state is SendState.RUNNING:
            if self.inputs.graceful_stop:
                self._go_local(SendState.FINISHING)
                return SendAction.STOP
            self._go_local(SendState.STOPPING)
            return SendAction.FORCE_STOP
        if state is SendState.FINISHING:
            self._go_local(SendState.STOPPING)
            return SendAction.FORCE_STOP
        return SendAction.NONE  # idle_empty, stopping

    def within_force_window(self) -> bool:
        return self.finishing_since is not None and self.clock() - self.finishing_since <= self.FORCE_WINDOW_S

    def long_press_items(self) -> tuple[tuple[SendAction, str], ...]:
        return _LONG_PRESS.get(self.state, ())

    def select(self, action: SendAction) -> SendAction:
        """A long-press menu choice; force stop moves the button to ``stopping``."""
        if action is SendAction.FORCE_STOP and self.state in (SendState.RUNNING, SendState.FINISHING):
            self._go_local(SendState.STOPPING)
        return action
