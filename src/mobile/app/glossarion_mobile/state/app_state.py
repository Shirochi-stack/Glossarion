"""App-wide UI state (Signals). Pure Python, no Flet import.

Until MobileConfigStore (U2), JobService and the shared cores (U3+) exist, the
values here are placeholders with the desktop defaults: model
``authgpt/gpt-6-luna`` (translator_gui ``config.get('model', ...)``), the first
built-in prompt profile ``Universal`` and target language ``English``
(``lang_var`` fallback). No ChatGPT account is signed in, so Send shows the
``blocked`` state with "Sign in with ChatGPT" once the engine is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from glossarion_mobile.services.oauth import sign_in_satisfied
from glossarion_mobile.state.chat_index import InMemoryChatIndex
from glossarion_mobile.state.store import LoopGuard, Signal
from glossarion_mobile.ui.chat.output_modes import OutputModeState
from glossarion_mobile.ui.chat.send_state import (
    BLOCK_ENGINE_FAILED,
    BLOCK_ENGINE_NOT_READY,
    DEFAULT_MODEL,
    BlockReason,
    chatgpt_sign_in_reason,
    requires_chatgpt_sign_in,
)
from glossarion_mobile.ui.responsive import SizeClass

__all__ = [
    "AppState",
    "ChatContext",
    "DEFAULT_PROFILE",
    "DEFAULT_TARGET_LANGUAGE",
    "JobStripModel",
    "JobsBadge",
]

DEFAULT_PROFILE = "Universal"  # first key of translator_gui default_prompts
DEFAULT_TARGET_LANGUAGE = "English"  # translator_gui lang_var fallback


@dataclass(frozen=True)
class ChatContext:
    """The header subtitle: Model · Profile · → Target (and the "custom" badge)."""

    model: str = DEFAULT_MODEL
    profile: str = DEFAULT_PROFILE
    target_language: str = DEFAULT_TARGET_LANGUAGE
    custom: bool = False


@dataclass(frozen=True)
class JobStripModel:
    """What the global JobStrip shows (UI_SPEC §1.7); placeholder data until JobService."""

    title: str
    subtitle: str = ""
    progress: Optional[float] = None  # 0..1, None = indeterminate
    kind_icon: str = "TRANSLATE"
    queued: int = 0
    state: str = "running"  # running | finishing | stopping | done | failed
    warning: bool = False  # subtitle in the warning colour (e.g. waiting for a glossary decision)
    owner_chat: Optional[str] = None  # cid of the chat that owns the job


@dataclass(frozen=True)
class JobsBadge:
    running: int = 0
    queued: int = 0

    @property
    def count(self) -> int:
        return self.running + self.queued


class AppState:
    def __init__(self, *, guard: Optional[LoopGuard] = None, chats: Optional[InMemoryChatIndex] = None) -> None:
        def sig(value: Any, name: str) -> Signal[Any]:
            return Signal(value, name=name, guard=guard)

        self.width = sig(0.0, "width")
        self.size_class = sig(SizeClass.PHONE, "size_class")
        self.text_scale = sig(1.0, "text_scale")
        self.backend = sig(None, "backend")  # warm-import result dict; None while warming up
        self.chat_context = sig(ChatContext(), "chat_context")
        self.signed_in = sig(frozenset(), "signed_in")  # signed-in slot keys: "authgpt", "authgpt2", "authgem"…
        self.output_mode = sig(OutputModeState(), "output_mode")
        self.job_strip = sig(None, "job_strip")  # Optional[JobStripModel]
        self.jobs_badge = sig(JobsBadge(), "jobs_badge")
        self.current_chat = sig("1", "current_chat")
        self.selftest_result = sig(None, "selftest_result")
        self.selftest_running = sig(False, "selftest_running")
        self.chats = chats if chats is not None else InMemoryChatIndex()

    @property
    def engine_ready(self) -> bool:
        result = self.backend.value
        return bool(result) and bool(result.get("ok"))

    def send_block(self) -> Optional[BlockReason]:
        """First reason Send cannot run, or None (only what U1 can know)."""
        result = self.backend.value
        if result is None:
            return BLOCK_ENGINE_NOT_READY
        if not result.get("ok"):
            return BLOCK_ENGINE_FAILED
        model = self.chat_context.value.model
        if self.needs_chatgpt_sign_in(model):
            return chatgpt_sign_in_reason(model)
        return None

    def needs_chatgpt_sign_in(self, model: Optional[str] = None) -> bool:
        """A ChatGPT route whose account is not signed in (``signed_in`` holds slot keys:
        ``authgptN/`` needs ``authgptN``; the ``authgpt0/`` pool any ChatGPT slot)."""
        model = self.chat_context.value.model if model is None else model
        return requires_chatgpt_sign_in(model) and not sign_in_satisfied(model, self.signed_in.value)
