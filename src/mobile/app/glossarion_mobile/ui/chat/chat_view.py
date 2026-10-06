"""ChatView: the chat home (UI_SPEC §1.2, §2) assembled from its parts.

Body: ``SafeArea(Column[Stack[Transcript, ↓ FAB], JobStrip?, StatusCaption?, Composer])``
with the ChatHeader as the View app bar (phone) or a bar on top of the main area
(tablet). The chat column is full width on phones, max 760 dp on large phones and
max 860 dp, centred, on tablets.

Without services (``bind(env)`` not called: the U1 shell, host tests) the view only
explains what is missing. With a ``ChatEnv`` it is the Direct Text chat:

* the transcript renders the current v2 session (``ChatStoreAdapter``) in the
  desktop rendered-card window, attachment turns grouped into JobCards, plus the
  live tail (streaming request cards, the glossary approval card, a Plan card);
* the composer restores/saves the draft and the attachment, keeps the output mode
  (auto Vision for images/CBZ) and shows option pills;
* Send/Stop follows the chat's run (``ChatRuns``) and JobService: send records the
  turn and submits a ``direct_text`` job; Stop is graceful, a second tap forces;
* sheets: ＋ (attach), output-mode options, chat settings, the minimal model sheet,
  the manual glossary sheet and the ChatGPT LoginSheet.

All Flet mutation happens on the UI loop; worker-thread notifications arrive
through ``ChatEnv.on_ui``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.services.background import IOS_BACKGROUND_NOTICE
from glossarion_mobile.state.app_state import AppState, ChatContext, JobStripModel
from glossarion_mobile.ui.chat.cards import (
    GlossaryApprovalCard,
    GlossaryEditorView,
    JobCard,
    RequestSheet,
    glossary_preview,
)
from glossarion_mobile.ui.chat.composer import Composer, StatusCaption
from glossarion_mobile.ui.chat.direct_text_rules import (
    TOKEN_HINT_MIN_CHARS,
    DirectTextSettings,
    ManualGlossarySource,
    count_tokens,
    effective_glossary_label,
    is_supported_attachment,
    needs_plan,
    token_hint,
)
from glossarion_mobile.ui.chat.header import ChatHeader
from glossarion_mobile.ui.chat.job_binding import (
    CardPhase,
    EtaEstimator,
    chat_id_of,
    ended_card,
    ended_kind,
    progress_counts,
    progress_line,
    running_label,
    state_name,
)
from glossarion_mobile.ui.chat.messages import AssistantMessage, UserBubble, UserFileCard
from glossarion_mobile.ui.chat.mode_options_sheet import ModeOptionsSheet
from glossarion_mobile.ui.chat.output_modes import OutputModeState, is_vision_attachment, normalize_mode
from glossarion_mobile.ui.chat.run_request import attachment_record
from glossarion_mobile.ui.chat.send_state import (
    BLOCK_ATTACHMENT_MISSING,
    BLOCK_GLOSSARY_PENDING,
    SendAction,
    SendInputs,
    SendState,
    excluded_route_reason,
)
from glossarion_mobile.ui.chat.transcript import Transcript
from glossarion_mobile.ui.chat.transcript_model import ACTIONS_LABEL, REPORT_LABEL, build_items, slide_window, tail_window
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.responsive import Layout, layout_for
from glossarion_mobile.ui.sheets.plus_sheet import PlusSheet
from glossarion_mobile.ui.shell.job_strip import JobStrip

__all__ = ["ChatView", "TOOL_ROUTES"]

log = logging.getLogger("glossarion.chat")

# ＋ sheet tool -> route name of its surface (None: the tool runs from the chat later)
TOOL_ROUTES: dict[str, Optional[str]] = {
    "extract_glossary": None,
    "qa": "tools.qa",
    "compile": "tools.convert",
    "headers": "tools.headers",
    "manga": "tools.manga",
    "review": "tools.review",
    "async": "tools.async",
    "progress": "tools.progress",
    "glossary_progress": "tools.progress.glossary",
    "retranslate": None,
}
_NOT_YET = {
    "extract_glossary": "Glossary extraction is not available in this session",
    "retranslate": "Retranslating chapters arrives in U7",
}
_LATER = {
    SendAction.SEND_AS_SCRATCH: "Scratch chats arrive in U7",
}
# ModelSheet field -> chat override key (run_request.OVERRIDE_CONFIG_KEYS maps them to config keys)
_SHEET_OVERRIDE_KEYS = {"model": "model", "profile": "profile", "language": "target_language"}
_ATTACH_EXTENSIONS = [
    "txt", "epub", "pdf", "md", "markdown", "html", "htm", "xhtml", "xml", "json", "csv", "tsv", "srt", "ass",
    "lrc", "vtt", "log", "sdlxliff", "zip", "cbz", "mp4", "png", "jpg", "jpeg", "gif", "bmp", "webp", "tif",
    "tiff", "svg", "ico", "heic", "heif", "avif", "jxl",
]
_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "bmp", "webp", "tif", "tiff", "heic", "heif", "avif", "jxl"]
COPIED_SECONDS = 1.6


class ChatView:
    def __init__(
        self,
        page: Any,
        *,
        state: AppState,
        navigate: Callable[..., Any],  # navigate(route_name, params=None)
        notify: Callable[..., Any],  # notify(message, action_label=None, on_action=None)
        open_drawer: Optional[Callable[[], Any]] = None,
        haptics: Any = None,
        env: Any = None,
    ) -> None:
        self.page = page
        self.state = state
        self.navigate = navigate
        self.notify = notify
        self.open_drawer = open_drawer
        self.haptics = haptics
        self.env: Any = None
        self.layout: Layout = layout_for(getattr(page, "width", None) or 0)
        self._unsubs: list[Callable[[], None]] = []
        self._env_unsubs: list[Callable[[], None]] = []
        self.plus_sheet: Optional[PlusSheet] = None
        self.mode_sheet: Optional[ModeOptionsSheet] = None
        # Set by GlossaryFeature (U6): open a glossary file in the Glossary Manager's table editor /
        # add a term to the chat workspace's glossary (None: the actions stay disabled).
        self.glossary_table_opener: Optional[Callable[[str], Any]] = None
        self.glossary_term_adder: Optional[Callable[..., Any]] = None
        self.model_sheet: Any = None
        self.settings_sheet: Any = None
        self.glossary_sheet: Any = None
        self.login_sheet: Any = None
        self.cid = str(state.current_chat.value)
        self.window: Optional[tuple] = None
        self.live_cards: dict = {}
        self.live_job_card: Optional[JobCard] = None
        self.approval_card: Optional[GlossaryApprovalCard] = None
        self._approval_key: Any = None
        self._stream_task: Any = None
        self._eta = EtaEstimator()
        self._applying_mode = False
        self._token_task: Any = None
        self._token_text = ""
        self.last_manual_glossary: Optional[ManualGlossarySource] = None
        self.sent: list = []  # (cid, text, attachment) of submitted sends (diagnostics/tests)

        chat = state.chats.get(state.current_chat.value)
        self.header = ChatHeader(
            title=chat.title if chat else "New chat",
            context=state.chat_context.value,
            on_menu=self._on_menu,
            on_open_model_sheet=self.open_model_sheet,
            on_new_chat=self._on_new_chat,
            on_new_scratch=self._on_new_scratch,
            on_menu_action=self._on_menu_action,
            on_rename=lambda e: self.open_rename(),
        )
        self.transcript = Transcript(
            on_suggestion=self._on_suggestion,
            on_scroll=self.header.on_transcript_scroll,
            on_load_earlier=lambda: self.slide(-1),
            on_load_later=lambda: self.slide(1),
            on_follow_change=self._on_follow_change,
        )
        self.new_fab = ft.FloatingActionButton(
            icon=ft.Icons.ARROW_DOWNWARD, mini=True, visible=False, tooltip="Jump to the newest message",
            on_click=self._jump_to_end,
        )
        self.transcript_stack = ft.Stack(
            [self.transcript, ft.Container(content=self.new_fab, right=12, bottom=8)], expand=True
        )
        self.job_strip = JobStrip(on_open=lambda: self.navigate("jobs"), on_stop=self._on_strip_stop)
        self.caption = StatusCaption(on_fix=self.run_fix)
        self.composer = Composer(
            mode_signal=state.output_mode,
            row_style=self.layout.output_row,
            on_plus=self.open_plus_sheet,
            on_plus_long_press=lambda: self._spawn(self.pick_and_attach(images=True)),
            on_send_action=self.on_send_action,
            on_content_changed=self._on_content_changed,
            on_expand=self._on_expand,
            on_open_mode_options=self.open_mode_options,
            on_draft_changed=self._on_draft_changed,
            on_attachment_removed=self._on_attachment_removed,
            on_pill=self._on_pill,
            on_pill_reset=self._on_pill_reset,
        )
        self.column = ft.Column(
            [self.transcript_stack, self.job_strip, self.caption, self.composer],
            spacing=0,
            expand=True,
        )
        self.body_control: Any = None
        self.refresh_send(push=False)
        self._apply_job_strip(state.job_strip.value)
        if env is not None:
            self.bind(env)

    # ---- layout ------------------------------------------------------------------

    def build_body(self, layout: Layout, available_width: Optional[float] = None) -> ft.Control:
        """The view body for a layout (the inner column is reused across rebuilds)."""
        self.apply_layout(layout)
        width = available_width if available_width is not None else layout.width
        if layout.chat_max_width is None or not width or width <= layout.chat_max_width:
            inner: ft.Control = self.column
        else:
            inner = ft.Row(
                [ft.Container(width=layout.chat_max_width, content=self.column)],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.STRETCH,
                expand=True,
            )
        self.body_control = ft.SafeArea(content=inner, expand=True)
        return self.body_control

    def apply_layout(self, layout: Layout) -> None:
        self.layout = layout
        self.composer.set_row_style(layout.output_row)
        self.composer.set_compact_text(layout.compact_text)
        self.header.set_compact_text(layout.compact_text)

    # ---- services ----------------------------------------------------------------

    def bind(self, env: Any) -> None:
        """Attach the chat services (``ChatEnv``) and show the current chat."""
        for unsub in self._env_unsubs:
            unsub()
        self._env_unsubs = []
        self.env = env
        if env.chats is not None:
            self._env_unsubs.append(env.chats.subscribe(self._on_store_changed))
        if env.runs is not None:
            self._env_unsubs.append(env.runs.subscribe(lambda cid: env.on_ui(self._on_run_changed, cid)))
        if env.jobs is not None and getattr(env.jobs, "available", False):
            # ChatRuns consumes the snapshots (run state); the view only re-derives Send/Queue
            self._env_unsubs.append(env.jobs.subscribe(lambda snap: env.on_ui(self._on_job_snapshot, snap)))
        if env.oauth is not None:
            self._env_unsubs.append(env.oauth.subscribe(lambda _s: env.on_ui(self._on_signin_changed)))
        self.load_chat(self.state.current_chat.value)

    @property
    def bound(self) -> bool:
        return self.env is not None and getattr(self.env, "chats", None) is not None and self.env.chats.available

    def _spawn(self, coro: Any) -> Any:
        if self.env is not None:
            return self.env.spawn(coro)
        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    # ---- state binding ----------------------------------------------------------

    def attach(self) -> None:
        if self._unsubs:
            return
        state = self.state
        self._unsubs = [
            state.backend.subscribe(lambda _v: self.refresh_send()),
            state.signed_in.subscribe(lambda _v: self.refresh_send()),
            state.chat_context.subscribe(self._on_context),
            state.job_strip.subscribe(self._on_job_strip),
            state.current_chat.subscribe(self._on_current_chat),
            state.output_mode.subscribe(self._on_output_mode),
        ]

    def detach(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    def _on_context(self, context: Any) -> None:
        self.header.set_context(context)
        self.refresh_send()
        self._push(self.header.wrapper)

    def _on_current_chat(self, cid: str) -> None:
        if self.bound and str(cid) != self.cid:
            self.load_chat(cid)
            return
        chat = self.state.chats.get(cid)
        self.header.set_title(chat.title if chat else "Chat")
        self._push(self.header.wrapper)

    def _apply_job_strip(self, model: Optional[JobStripModel]) -> None:
        # On the chat home the strip shows only for another chat's job (§1.7).
        if model is not None and model.owner_chat == self.state.current_chat.value:
            model = None
        self.job_strip.set_model(model)

    def _on_job_strip(self, model: Optional[JobStripModel]) -> None:
        self._apply_job_strip(model)
        self.refresh_send()  # another chat's job running -> Send queues

    # ---- chat loading / settings ---------------------------------------------------

    def settings(self, cid: Optional[str] = None) -> DirectTextSettings:
        cid = str(cid or self.cid)
        base = DirectTextSettings.from_config(self.env.config_get) if self.env is not None else DirectTextSettings()
        if not self.bound:
            return base
        overrides = dict(self.env.chats.overrides(cid))
        meta = self.env.chats.meta(cid)
        if meta.get("skip_plan") is not None:
            overrides["skip_plan"] = meta.get("skip_plan")
        return base.with_overrides(overrides)

    def chat_context_for(self, cid: str) -> ChatContext:
        env = self.env
        overrides = env.chats.overrides(cid) if self.bound else {}
        model = overrides.get("model") or env.config_get("model", None) or self.state.chat_context.value.model
        profile = overrides.get("profile") or env.config_get("active_profile", None) or self.state.chat_context.value.profile
        target = (
            overrides.get("target_language")
            or env.config_get("output_language", None)
            or env.config_get("glossary_target_language", None)
            or self.state.chat_context.value.target_language
        )
        custom = any(overrides.get(k) is not None for k in overrides)
        return ChatContext(model=str(model), profile=str(profile), target_language=str(target), custom=custom)

    def load_chat(self, cid: Any) -> None:
        """Show chat ``cid``: draft, attachment, mode, header, transcript tail (``_load_chat_session``)."""
        cid = str(cid)
        self.cid = cid
        if not self.bound:
            self._on_current_chat(cid)
            return
        chats = self.env.chats
        chats.select(cid)
        session = chats.session(cid)
        title = str((session or {}).get("title") or "New chat")
        self.header.set_title(title)
        self.composer.set_text(chats.draft(cid))
        record = chats.attachment(cid)
        if record and not os.path.isfile(str(record.get("path") or "")):
            record = None  # desktop: the pending attachment is restored only if the file still exists
        settings = self.settings(cid)
        mode_state = OutputModeState(normalize_mode(settings.output_mode))
        if record:
            mode_state = mode_state.attachment_changed(record.get("path"))
        self._set_mode(mode_state)
        self.composer.set_attachment(record)
        self.state.chat_context.set(self.chat_context_for(cid))
        self.composer.set_pills(self.option_pills())
        self.window = None
        self.live_cards = {}
        self.live_job_card = None
        self.approval_card = None
        self._approval_key = None
        self.render_transcript(follow=True)
        self.refresh_send()
        self._ensure_stream_task()

    def _on_store_changed(self) -> None:
        if not self.bound:
            return
        session = self.env.chats.session(self.cid)
        if session is None:  # the chat was deleted
            self.cid = self.env.chats.current_cid()
            self.load_chat(self.cid)
            return
        self.header.set_title(str(session.get("title") or "New chat"))
        self._push(self.header.wrapper)

    # ---- transcript rendering ------------------------------------------------------------

    def _messages(self) -> list:
        return self.env.chats.messages(self.cid) if self.bound else []

    def render_transcript(self, *, follow: bool = False) -> None:
        if not self.bound:
            return
        messages = self._messages()
        settings = self.settings()
        expanded = self.env.chats.expanded(self.cid)
        total = len(messages)
        if self.window is None or self.window[1] >= total - 1 or follow:
            self.window = tail_window(messages, settings.rendered_card_limit, expanded)
        start, end = self.window
        run = self.env.runs.live_run(self.cid) if self.env.runs is not None else None
        controls = [self._item_control(item, messages, expanded, run) for item in build_items(messages, start, end)]
        self.transcript.set_messages([c for c in controls if c is not None], hidden_before=start,
                                     hidden_after=max(0, total - end))
        self.transcript.set_tail(self._tail_controls(run))
        self.header.set_empty(self.transcript.is_empty and not self.composer.has_content)
        self._push(self.transcript)
        if follow and self.transcript.follow_tail and not settings.disable_auto_scroll:
            self._spawn(self.transcript.scroll_to_end())

    def slide(self, direction: int) -> None:
        if not self.bound:
            return
        messages = self._messages()
        bounds = self.window or tail_window(messages, self.settings().rendered_card_limit, self.env.chats.expanded(self.cid))
        self.window = slide_window(bounds, len(messages), self.settings().rendered_card_limit, direction)
        self.render_transcript()

    def _item_control(self, item: Any, messages: list, expanded: set, run: Any) -> Optional[ft.Control]:
        message = messages[item.index] if 0 <= item.index < len(messages) else None
        if item.kind == "user":
            return UserBubble(message[1], index=item.index, available_width=self.layout.width or 400,
                              on_long_press=self._user_actions, key=ft.ScrollKey(self._mid(item.index) or item.key))
        if item.kind == "user_file":
            return UserFileCard(
                message[1], message[2] if len(message) > 2 else "", message[3] if len(message) > 3 else 0,
                message[4] if len(message) > 4 else "", message[5] if len(message) > 5 else "user",
                index=item.index, missing=not os.path.isfile(str(message[2] if len(message) > 2 else "")),
                key=ft.ScrollKey(self._mid(item.index) or item.key),
            )
        if item.kind == "assistant":
            return self._assistant_control(item.index, messages[item.index], expanded)
        if item.kind == "job":
            return self._job_control(item, messages, run)
        return None

    def _mid(self, index: int) -> Optional[str]:
        try:
            return self.env.chats.mid_for_index(self.cid, index)
        except Exception:
            return None

    def _assistant_control(self, index: int, message: tuple, expanded: set) -> AssistantMessage:
        cid = self.cid
        chats = self.env.chats
        storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
        return AssistantMessage(
            index=index,
            request_label=str(message[5] if len(message) > 5 else ""),
            created_at=str(storage.get("created_at") or ""),
            processing_label=str(message[3] if len(message) > 3 else "Processing") or "Processing",
            content=lambda i=index: chats.message_text(cid, i, "content"),
            thinking=lambda i=index: chats.message_text(cid, i, "thinking"),
            expanded=index in expanded,
            on_toggle_thinking=lambda card, value: chats.set_expanded(cid, card.index, value),
            on_copy=self._copy_message,
            on_retranslate=self._retranslate,
            on_more=self._message_more,
            on_show_full=self._show_full,
            key=ft.ScrollKey(self._mid(index) or f"m-{index}"),
        )

    def _job_control(self, item: Any, messages: list, run: Any) -> JobCard:
        file_message = messages[item.index] if item.index >= 0 else None
        record = None
        if file_message is not None:
            name = str(file_message[1])
            record = {"name": name, "path": str(file_message[2] if len(file_message) > 2 else ""),
                      "extension": os.path.splitext(name)[1].lower(), "size": file_message[3] if len(file_message) > 3 else 0}
        live = run is not None and run.user_index == item.index
        plan = self._pending_plan()
        if plan is not None and plan.get("user_index") == item.index and not live:
            card = JobCard(attachment=record, phase=CardPhase("plan"), on_action=self._on_job_action, key=f"job-{item.index}")
            card.set_plan(self._plan_controls(plan))
            return card
        if live:
            card = JobCard(attachment=record, phase=CardPhase("running"), on_action=self._on_job_action,
                           on_open_request=self._open_request, key=f"job-{item.index}")
            self.live_job_card = card
            self._update_live_job_card(run)
            return card
        status = ""
        if item.report is not None:
            status = "Done"
        if not item.requests and item.report is None and item.actions is None:
            phase = CardPhase("interrupted")
            status = "Interrupted · no output yet"
        else:
            phase = CardPhase("done")
        runs = self.env.runs
        last = runs.run_for(self.cid) if runs is not None else None
        ended = None
        if last is not None and not last.live and last.user_index == item.index:
            ended = ended_card(last.state if last.state in ("stopped", "failed") else "done", last.last_snapshot)
        elif runs is not None:
            # After a relaunch (no run in this session) JobService still knows how the turn's
            # job ended: Interrupted (killed) / Stopped / Failed offer Resume, whatever cards
            # the run committed before it ended.
            remembered = runs.persisted_job(self.cid, item.index)
            if remembered is not None:
                ended = ended_card(ended_kind(remembered), remembered)
        if ended is not None:
            phase, status = CardPhase(ended[0]), ended[1]
        card = JobCard(attachment=record, phase=phase, on_action=lambda a, it=item: self._on_job_action(a, it),
                       on_open_request=self._open_request, key=f"job-{item.index}")
        card.set_phase(phase, status=status)
        segments = []
        for index in item.requests:
            msg = messages[index]
            segments.append({
                "label": str(msg[5] if len(msg) > 5 else ""),
                "content": str(msg[1] or ""),
                "thinking": str(msg[2] or "") if len(msg) > 2 else "",
                "phase": "processing",
                "complete": True,
                "index": index,
            })
        card.set_requests(segments)
        if item.report is not None:
            card.set_report(self.env.chats.message_text(self.cid, item.report, "content"))
        return card

    def _tail_controls(self, run: Any) -> list:
        controls: list = []
        if run is None or not run.live:
            self.live_cards = {}
            return controls
        if not run.run.is_attachment:
            for key, segment in self._live_segments(run):
                card = self.live_cards.get(key)
                if card is None:
                    card = AssistantMessage(live=True, actions=False, key=f"live-{key}")
                    self.live_cards[key] = card
                card.set_live(segment)
                controls.append(card)
            if not controls:
                placeholder = self.live_cards.get("_pending")
                if placeholder is None:
                    placeholder = AssistantMessage(live=True, actions=False, key="live-pending",
                                                   request_label=f"Request {self.env.chats.request_count(self.cid) + 1}")
                    self.live_cards["_pending"] = placeholder
                controls.append(placeholder)
        if run.awaiting_glossary:
            controls.append(self._approval_control(run))
        return controls

    def _live_segments(self, run: Any) -> list:
        out = []
        for position, segment in enumerate(run.stream.segments()):
            key = segment.get("request_number") or segment.get("thread") or position
            out.append((str(key), segment))
        return out

    def _approval_control(self, run: Any) -> GlossaryApprovalCard:
        question = run.question or {}
        key = question.get("id") or "q"
        if self.approval_card is not None and self._approval_key == key:
            return self.approval_card
        path = str((question.get("data") or {}).get("path") or (question.get("data") or {}).get("glossary_path") or "")
        info = None
        try:
            info = glossary_preview(path)
        except Exception:
            info = None
        self.approval_card = GlossaryApprovalCard(
            path=path, info=info, on_answer=self._answer_glossary, on_edit=self.open_glossary_editor,
        )
        self._approval_key = key
        return self.approval_card

    # ---- live updates ----------------------------------------------------------------------

    def _on_run_changed(self, cid: str) -> None:
        if str(cid) != self.cid:
            self.refresh_send()
            return
        run = self.env.runs.run_for(cid)
        if run is not None and not run.live:
            # finished: show the committed cards from the store
            self.live_cards = {}
            self.live_job_card = None
            self.approval_card = None
            self.render_transcript(follow=True)
        else:
            self.render_transcript(follow=self.transcript.follow_tail)
        self.refresh_send()
        self._ensure_stream_task()

    def _on_job_snapshot(self, snapshot: Any) -> None:
        if self.env is None:
            return
        self.refresh_send()
        run = self.env.runs.live_run(self.cid) if self.env.runs is not None else None
        if run is not None and self.live_job_card is not None:
            self._update_live_job_card(run)
            self.live_job_card.push()

    def _update_live_job_card(self, run: Any) -> None:
        card = self.live_job_card
        if card is None:
            return
        snapshot = run.last_snapshot
        counts = progress_counts(snapshot)
        phase_name = {"queued": "queued", "stopping": "stopping", "force_stopping": "force_stopping"}.get(run.state, "running")
        card.set_phase(CardPhase(phase_name), status=running_label(snapshot, waiting_for_glossary=run.awaiting_glossary)
                       if phase_name == "running" else "")
        segments = run.stream.segments()
        current = str(segments[-1].get("label") or "") if segments else ""
        card.set_progress(counts.get("fraction"), progress_line(snapshot, eta=self._eta), current)
        card.set_requests(segments)

    def _ensure_stream_task(self) -> None:
        if not self.bound or self.env.runs is None:
            return
        run = self.env.runs.live_run(self.cid)
        if run is None or (self._stream_task is not None and not self._stream_task.done()):
            return
        self._stream_task = self._spawn(self._stream_loop(run))

    async def _stream_loop(self, run: Any) -> None:
        """Coalesced streaming repaints (desktop cadence 280-900 ms)."""
        while run.live and str(run.cid) == self.cid:
            await asyncio.sleep(run.stream.render_interval_ms() / 1000.0)
            if not run.live or str(run.cid) != self.cid:
                break  # finished meanwhile: _on_run_changed renders the committed cards
            if run.stream.drain():
                self.transcript.set_tail(self._tail_controls(run))
                if self.live_job_card is not None:
                    self._update_live_job_card(run)
                self._push(self.transcript)
                if self.transcript.follow_tail and not self.settings().disable_auto_scroll:
                    await self.transcript.scroll_to_end(0)
                elif not self.transcript.follow_tail:
                    self.new_fab.visible = True
                    self._push(self.new_fab)

    def _on_follow_change(self, follow: bool) -> None:
        if follow and self.new_fab.visible:
            self.new_fab.visible = False
            self._push(self.new_fab)

    async def _jump_to_end(self, e: Any = None) -> None:
        self.window = None
        self.render_transcript()
        self.new_fab.visible = False
        self._push(self.new_fab)
        await self.transcript.scroll_to_end()

    # ---- Send/Stop ---------------------------------------------------------------------------

    def _chat_block(self) -> Any:
        block = self.state.send_block()
        if block is not None:
            return block
        run = self.env.runs.live_run(self.cid) if (self.bound and self.env.runs is not None) else None
        if run is not None and run.awaiting_glossary:
            return BLOCK_GLOSSARY_PENDING
        record = self.composer.attachment
        if record and not os.path.isfile(str(record.get("path") or "")):
            return BLOCK_ATTACHMENT_MISSING
        model = self.state.chat_context.value.model
        try:
            from glossarion_mobile.ui.sheets.model_sheet_min import excluded_route

            if excluded_route(model):
                return excluded_route_reason(model)
        except Exception:
            pass
        return None

    def send_inputs(self) -> SendInputs:
        own_state = None
        other_running, other_title = False, ""
        if self.bound and self.env.runs is not None:
            own_state = self.env.runs.own_job_state(self.cid)
        snapshot = self.env.jobs.snapshot() if (self.env is not None and self.env.jobs is not None) else None
        if snapshot is not None and state_name(snapshot) in ("QUEUED", "STARTING", "RUNNING", "STOPPING", "FORCE_STOPPING"):
            if chat_id_of(snapshot) != self.cid:
                other_running = True
                other_title = str(getattr(getattr(snapshot, "spec", None), "title", "") or "the current job")
        else:
            strip = self.state.job_strip.value
            other_running = strip is not None and strip.owner_chat != self.state.current_chat.value and strip.state in (
                "running",
                "finishing",
                "stopping",
            )
            other_title = strip.title if (strip is not None and other_running) else ""
        graceful = True
        if self.env is not None and self.env.jobs is not None:
            graceful = self.env.jobs.graceful_stop_configured()
        return SendInputs(
            has_content=self.composer.has_content,
            block=self._chat_block(),
            other_job_running=other_running,
            other_job_title=other_title,
            own_job_state=own_state,
            graceful_stop=graceful,
        )

    def refresh_send(self, push: bool = True) -> SendState:
        inputs = self.send_inputs()
        self.composer.apply_send_inputs(inputs)
        machine = self.composer.send_button.machine
        state = machine.state
        caption = machine.caption
        if self.bound and self.env.runs is not None:
            run_caption = self.env.runs.caption(self.cid)
            if run_caption and state in (SendState.RUNNING, SendState.IDLE_EMPTY, SendState.IDLE_READY):
                caption = run_caption
        self.caption.show(caption, inputs.block if state is SendState.BLOCKED else None)
        self.header.set_empty(self.transcript.is_empty and not self.composer.has_content)
        if push:
            self._push(self.caption, self.header.wrapper)
        return state

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            if control is None:
                continue
            try:
                control.update()
            except Exception:  # not mounted
                pass

    def _haptic(self, kind: str) -> None:
        if self.haptics is not None:
            self.haptics.fire(kind)

    def on_send_action(self, action: SendAction) -> None:
        if action is SendAction.EXPLAIN_BLOCK:
            block = self.send_inputs().block
            if block is None:
                return
            self.notify(
                block.message,
                action_label=block.fix_label,
                on_action=(lambda a=block.fix_action: self.run_fix(a)) if block.fix_action else None,
            )
            return
        if action in (SendAction.STOP, SendAction.FORCE_STOP):
            self._haptic("heavy_impact" if action is SendAction.FORCE_STOP else "light_impact")
            if self.bound and self.env.runs is not None and self.env.runs.live_run(self.cid) is not None:
                self.env.runs.request_stop(self.cid, force=action is SendAction.FORCE_STOP)
            else:
                self.notify("No job is running")
            self.refresh_send()
            return
        if action in (SendAction.SEND, SendAction.QUEUE):
            self._haptic("light_impact")
            if not self.bound or self.env.runs is None:
                self.notify("Translating needs the job service, which is not running in this build.")
                return
            self.begin_send()
            return
        if action is SendAction.ADD_WITHOUT_TRANSLATING:
            self.add_without_translating()
            return
        if action is SendAction.SEND_ONCE_WITH_MODEL:
            self.open_model_once()
            return
        if action is SendAction.STOP_CURRENT_AND_SEND:
            self.confirm_stop_current_and_send()
            return
        message = _LATER.get(action)
        if message:
            self.notify(message)

    def confirm_stop_current_and_send(self) -> Optional[ConfirmDialog]:
        """UI_SPEC §2.4 "Stop current & send" (confirm): stop the running job (graceful first,
        like the Stop button), then send this message (it queues until the job has stopped)."""
        snapshot = self.env.jobs.snapshot() if (self.env is not None and self.env.jobs is not None) else None
        title = str(getattr(getattr(snapshot, "spec", None), "title", "") or "the current job")

        def go() -> None:
            if self.env is not None and self.env.jobs is not None:
                self.env.jobs.request_stop(force=False)
            self.begin_send()

        if self.page is None:
            go()
            return None
        dialog = ConfirmDialog(title="Stop current job?",
                               body=f"{title} stops after its current request; this message is sent next. "
                                    "The stopped job can be resumed later.",
                               confirm_label="Stop & send", cancel_label="Cancel", destructive=True, on_confirm=go)
        dialog.show(self.page)
        return dialog

    def begin_send(self, once: Optional[dict] = None) -> None:
        """``_on_enter_clicked`` -> ``_start_translation`` (asks the manual glossary first when required).

        ``once``: chat override keys (model / profile / target_language) for this send only, from
        long-press Send -> "Translate once with another model…" (UI_SPEC §2.2 one-shot)."""
        text = self.composer.send_text()
        record = self.composer.attachment
        if record and not os.path.isfile(str(record.get("path") or "")):
            self.notify("Attachment missing", action_label="Remove", on_action=self.composer._remove_attachment)
            return
        if not text and not record:
            self.notify("Enter text or attach a TXT, EPUB, PDF, CBZ, or image file to translate.")
            return
        settings = self.settings()
        if settings.glossary_override_mode == "manual":
            self.open_manual_glossary(lambda source: self._send(text, record, settings, source, once))
            return
        self._send(text, record, settings, None, once)

    def _send(self, text: str, record: Optional[dict], settings: DirectTextSettings,
              manual: Optional[ManualGlossarySource], once: Optional[dict] = None) -> None:
        env = self.env
        cid = self.cid
        output_mode = self.state.output_mode.value.mode
        if record and needs_plan(record.get("extension") or "", len(text), skip_plan=settings.skip_plan):
            from glossarion_mobile.ui.chat.run_request import user_turn

            index = env.chats.record_user_turn(
                cid, user_turn(text, record, settings.attachment_prompt_role), record.get("name") or text
            )
            env.chats.set_meta(cid, "pending_plan", {
                "user_index": index, "text": text, "attachment": dict(record), "output_mode": output_mode,
                "manual_glossary": manual.as_dict() if manual else None, "created": time.time(),
                "once": dict(once) if once else None,
            })
            self._after_send_ui(output_mode)
            return
        self._after_send_ui(output_mode)
        self.sent.append((cid, text, dict(record) if record else None))
        self._spawn(self._submit(cid, text, record, settings, output_mode, manual, None, once))

    def _after_send_ui(self, output_mode: str) -> None:
        """Clear the composer and restore the mode non-automatically (desktop after recording the turn)."""
        self.composer.clear()
        self._set_mode(OutputModeState(normalize_mode(output_mode)))
        self.window = None
        self.render_transcript(follow=True)
        self.refresh_send()

    async def _submit(self, cid: str, text: str, record: Optional[dict], settings: DirectTextSettings,
                      output_mode: str, manual: Optional[ManualGlossarySource], user_index: Optional[int],
                      once: Optional[dict] = None) -> Any:
        overrides = self.env.chats.overrides(cid)
        if once:
            overrides = {**overrides, **{k: v for k, v in once.items() if v}}
        try:
            run = await self.env.runs.send(
                cid, text=text, attachment=record, settings=settings, output_mode=output_mode,
                overrides=overrides, manual_glossary=manual, user_index=user_index,
            )
        except Exception as exc:
            log.warning("chat send failed: %s", exc)
            self.notify(f"Could not start: {exc}")
            self._on_run_changed(cid)
            return None
        self._on_run_changed(cid)
        return run

    def add_without_translating(self) -> None:
        if not self.bound:
            return
        text = self.composer.send_text()
        record = self.composer.attachment
        if not text and not record:
            return
        from glossarion_mobile.state.chat_store_adapter import message_fingerprints
        from glossarion_mobile.ui.chat.run_request import user_turn

        settings = self.settings()
        index = self.env.chats.record_user_turn(
            self.cid, user_turn(text, record, settings.attachment_prompt_role), (record or {}).get("name") or text
        )
        fps = message_fingerprints(self.env.chats.messages(self.cid))
        add_only = list(self.env.chats.meta(self.cid).get("add_only") or [])
        if 0 <= index < len(fps):
            add_only.append(fps[index])
        self.env.chats.set_meta(self.cid, "add_only", add_only)
        self._after_send_ui(self.state.output_mode.value.mode)

    # ---- Plan card ---------------------------------------------------------------------------

    def _pending_plan(self) -> Optional[dict]:
        if not self.bound:
            return None
        plan = self.env.chats.meta(self.cid).get("pending_plan")
        return plan if isinstance(plan, dict) else None

    def _plan_controls(self, plan: dict) -> list:
        context = self.state.chat_context.value
        settings = self.settings()
        glossary = effective_glossary_label(settings.glossary_override_mode,
                                            self.env.config_get if self.env is not None else (lambda k, d=None: d))
        chips = [
            ft.Chip(label=ft.Text(context.model), on_click=lambda e: self.open_model_sheet("model")),
            ft.Chip(label=ft.Text(context.profile), on_click=lambda e: self.open_model_sheet("profile")),
            ft.Chip(label=ft.Text(f"→ {context.target_language}"), on_click=lambda e: self.open_model_sheet("language")),
            ft.Chip(label=ft.Text(glossary), on_click=lambda e: self.open_chat_settings()),
            ft.Chip(label=ft.Text(f"Output: {plan.get('output_mode') or 'text'}")),
            ft.Chip(label=ft.Text("Save to: This chat")),
        ]
        controls: list = [
            ft.Row(chips, wrap=True, spacing=6, run_spacing=6),
            ft.ExpansionTile(
                title="Run options",
                subtitle=ft.Text("Settings › Translation defaults apply to this run",
                                 theme_style=ft.TextThemeStyle.LABEL_SMALL),
                controls=[ft.TextButton(content="Open translation settings",
                                        on_click=lambda e: self.navigate("settings"))],
            ),
        ]
        if self.env is not None and self.env.is_ios:
            # UI_SPEC §7.6 / §2.12.1: where a job starts on iOS, say that it may pause in the background
            controls.append(ft.Text(IOS_BACKGROUND_NOTICE, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        return controls

    def start_plan(self) -> None:
        plan = self._pending_plan()
        if plan is None:
            return
        env = self.env
        cid = self.cid
        env.chats.set_meta(cid, "pending_plan", None)
        manual_data = plan.get("manual_glossary")
        manual = None
        if isinstance(manual_data, dict):
            manual = ManualGlossarySource(
                kind=str(manual_data.get("kind") or "content"), path=str(manual_data.get("path") or ""),
                content=str(manual_data.get("content") or ""), extension=str(manual_data.get("extension") or ".txt"),
            )
        record = plan.get("attachment") or None
        self.sent.append((cid, plan.get("text") or "", record))
        once = plan.get("once") if isinstance(plan.get("once"), dict) else None
        self._spawn(self._submit(cid, str(plan.get("text") or ""), record, self.settings(cid),
                                 str(plan.get("output_mode") or "text"), manual, int(plan.get("user_index") or 0),
                                 once))

    def cancel_plan(self) -> None:
        """Cancel removes the plan and the unsent user_file turn (§2.12.1)."""
        plan = self._pending_plan()
        if plan is None:
            return
        chats = self.env.chats
        chats.set_meta(self.cid, "pending_plan", None)
        index = int(plan.get("user_index") if plan.get("user_index") is not None else -1)
        if index >= 0 and index == len(chats.messages(self.cid)) - 1:
            chats.truncate_messages(self.cid, index)  # the unsent user_file turn goes with the plan
        record = plan.get("attachment")
        if record:
            self.composer.set_attachment(record)
            chats.set_attachment(self.cid, record)
        self.composer.set_text(str(plan.get("text") or ""))
        self.render_transcript(follow=True)
        self.refresh_send()

    def _on_job_action(self, action: str, item: Any = None) -> None:
        if action == "start":
            self.start_plan()
        elif action == "cancel_plan":
            self.cancel_plan()
        elif action in ("stop", "cancel_queued"):
            self.on_send_action(SendAction.STOP)
        elif action == "force_stop":
            self.on_send_action(SendAction.FORCE_STOP)
        elif action == "log":
            run = self.env.runs.run_for(self.cid) if self.bound else None
            if run is not None and run.job_id is not None:
                self.navigate("jobs.detail", {"jid": str(run.job_id)})
            else:
                self.navigate("jobs")
        elif action == "export":
            self._spawn(self._export_outputs())
        elif action == "compile":
            self._spawn(self._compile())
        elif action == "migrate":
            self.notify("Migrating attachment workspaces arrives with the attachments manager (U7)")
        elif action in ("read", "open_reader"):
            self._spawn(self._open_reader(item))
        elif action == "open_output":
            folder = self._last_output_folder()
            if folder and self.env.open_output is not None:
                self.env.open_output(folder)
            else:
                self.notify("No output folder for this chat yet")
        elif action in ("resume", "retry"):
            self._resume_last()
        else:
            self.notify("This action arrives in a later milestone")

    async def _open_reader(self, item: Any = None) -> None:
        """Job card Read / Open reader (U5): the turn's workspace in the Reader."""
        opener = self.env.open_reader if self.env is not None else None
        if opener is None:
            self.notify("The Reader is not available in this session")
            return
        folder, source = self._reader_target(item)
        result = opener(folder, source)
        if hasattr(result, "__await__"):
            await result

    def _reader_target(self, item: Any = None) -> tuple:
        """(workspace folder, attachment path) of the job card's turn (default: the chat's run)."""
        runs = self.env.runs if self.bound else None
        run = runs.run_for(self.cid) if runs is not None else None
        index = getattr(item, "index", None)
        if index is None and run is not None:
            index = run.user_index
        source = ""
        messages = self._messages()
        if index is not None and 0 <= int(index) < len(messages):
            message = messages[int(index)]
            if message and len(message) > 2 and str(message[0]) == "user_file":
                source = str(message[2] or "")
        folder = ""
        if run is not None and (index is None or run.user_index == index):
            folder = str(run.output_dir or run.output_folder or "")
        return folder or self._last_output_folder(), source

    def _last_output_folder(self) -> str:
        run = self.env.runs.run_for(self.cid) if self.env.runs is not None else None
        if run is not None and run.output_folder:
            return run.output_folder
        for message in reversed(self._messages()):
            if message and message[0] == "assistant" and len(message) > 4 and str(message[4] or ""):
                return str(message[4])
        return self.env.chats.output_folder(self.cid)

    async def _export_outputs(self) -> None:
        """Share / Export the run's compiled documents: the desktop attachment rule
        (``ChatStoreMixin._preferred_attachment_compiled_documents``: the newest top-level
        EPUB and PDF; nested PDFs are resources), EPUB first; both are offered when both exist."""
        folder = self._last_output_folder()
        documents: list = []
        if folder and os.path.isdir(folder):
            def pick() -> list:
                from direct_text_store import ChatStoreMixin  # shared (U3)

                preferred = ChatStoreMixin._preferred_attachment_compiled_documents(folder)
                return [os.path.join(folder, preferred[ext]) for ext in (".epub", ".pdf") if preferred.get(ext)]

            documents = await self.env.run_io(pick)
        if not documents:
            self.notify("No compiled EPUB or PDF in this chat's output yet")
            return
        if len(documents) > 1 and self.page is not None:
            items = [ActionItem(f"{os.path.splitext(path)[1][1:].upper()} · {os.path.basename(path)}",
                                lambda p=path: self._spawn(self._export_document(p)), icon="IOS_SHARE")
                     for path in documents]
            ActionSheet(items, title="Share / Export", tablet=bool(self.layout.persistent_sidebar)).show(self.page)
            return
        await self._export_document(documents[0])

    async def _export_document(self, path: str) -> None:
        export_file = getattr(self.env, "export_file", None)
        if export_file is not None:
            result = export_file(path)
        elif self.env.share_files is not None:
            result = self.env.share_files([path])
        else:
            return
        if asyncio.iscoroutine(result):
            await result

    async def _compile(self) -> None:
        try:
            job_id = await self.env.runs.compile(self.cid)
        except Exception as exc:
            self.notify(f"Could not start compiling: {exc}")
            return
        if job_id is None:
            self.notify("No translated output folder for this chat yet")
        else:
            self.notify("Compiling EPUB · see Jobs", action_label="Jobs", on_action=lambda: self.navigate("jobs"))

    def _resume_last(self) -> None:
        """Resume / Retry failed: resubmit the chat's last run (same run root -> resumes from progress)."""
        if self.env is None or self.env.runs is None:
            return

        async def go() -> None:
            try:
                run = await self.env.runs.resubmit(self.cid)
            except Exception as exc:
                self.notify(f"Could not resume: {exc}")
                return
            if run is None:
                self.notify("Nothing to resume in this chat")
            self._on_run_changed(self.cid)

        self._spawn(go())

    # ---- glossary approval ------------------------------------------------------------------

    def _answer_glossary(self, accepted: bool) -> None:
        if self.bound and self.env.runs is not None:
            self.env.runs.answer_glossary(self.cid, accepted)
        self.refresh_send()

    def open_glossary_editor(self, path: str) -> None:
        if self.env is None:
            return

        async def run() -> None:
            info = await self.env.run_io(glossary_preview, path)
            if not info.get("exists"):
                self.notify("The generated glossary file could not be found.")
                return
            table = self.glossary_table_opener  # GlossaryFeature (U6): the Glossary Manager's editor
            editor = GlossaryEditorView(path, info.get("text") or "", has_bom=bool(info.get("bom")), mono=self.env.mono,
                                        on_close=lambda: self._close_overlay(editor.view),
                                        on_saved=lambda p: self.notify(f"Saved edited glossary: {os.path.basename(p)}"),
                                        on_table=(lambda: table(path)) if table is not None else None)
            if self.env.push_overlay is not None:
                self.env.push_overlay(editor.view)

        self._spawn(run())

    def _close_overlay(self, view: Any) -> None:
        if self.env is not None and self.env.pop_overlay is not None:
            self.env.pop_overlay(view)

    # ---- composer events ---------------------------------------------------------------------

    # ---- output mode / token hint ---------------------------------------------------------

    def _set_mode(self, mode_state: OutputModeState) -> None:
        """Programmatic mode change (chat load, auto Vision, after send): never persisted."""
        self._applying_mode = True
        try:
            self.state.output_mode.set(mode_state)
        finally:
            self._applying_mode = False

    def _on_output_mode(self, mode_state: Any) -> None:
        """A toggle tap: persist like ``_set_direct_output_mode(persist=True)`` (chat override when set)."""
        if self._applying_mode or getattr(mode_state, "automatic", False) or not self.bound:
            return
        mode = normalize_mode(getattr(mode_state, "mode", "text"))
        if self.env.chats.overrides(self.cid).get("output_mode") is not None:
            self.env.chats.set_override(self.cid, "output_mode", mode)
        elif self.env.config_get("direct_text_output_mode", None) != mode:
            self.env.config_set_many({"direct_text_output_mode": mode})

    def _on_content_changed(self, _has_content: bool = False) -> None:
        self.refresh_send()
        text = self.composer.text
        if len(text) < TOKEN_HINT_MIN_CHARS:
            if self.composer.token_hint.visible:
                self.composer.set_token_hint("")
                self._push(self.composer.token_hint)
            return
        self._token_text = text
        if self.env is not None and (self._token_task is None or self._token_task.done()):
            self._token_task = self._spawn(self._update_token_hint())

    async def _update_token_hint(self) -> None:
        """tiktoken count in a worker, debounced 450 ms (UI_SPEC §2.3)."""
        await asyncio.sleep(0.45)
        text = self._token_text
        model = self.state.chat_context.value.model
        try:
            count = await self.env.run_io(count_tokens, text, model)
        except Exception:
            return
        if self.composer.text != text and len(self.composer.text) >= TOKEN_HINT_MIN_CHARS:
            self._token_task = None
            self._on_content_changed()
            return
        self.composer.set_token_hint(token_hint(count) if len(self.composer.text) >= TOKEN_HINT_MIN_CHARS else "")
        self._push(self.composer.token_hint)

    def _on_draft_changed(self, text: str) -> None:
        if self.bound:
            self.env.chats.set_draft(self.cid, text)

    def _on_attachment_removed(self) -> None:
        if self.bound:
            self.env.chats.set_attachment(self.cid, None)
        self._set_mode(self.state.output_mode.value.attachment_changed(None))
        self.refresh_send()

    def attach_file(self, path: str) -> bool:
        """``_set_attachment``: one supported file per turn; images/CBZ switch to Vision · auto."""
        record = attachment_record(path)
        if record is None or not is_supported_attachment(path):
            self.notify("Unsupported attachment")
            return False
        previous = self.composer.attachment
        mode = self.state.output_mode.value
        if previous and is_vision_attachment(previous.get("path")) and not is_vision_attachment(path):
            mode = mode.attachment_changed(None)
        mode = mode.attachment_changed(path) if is_vision_attachment(path) else mode
        self._set_mode(mode)
        self.composer.set_attachment(record)
        if self.bound:
            self.env.chats.set_attachment(self.cid, record)
        name = record["name"]
        self.caption.show(f"Attached {name} · Vision enabled" if is_vision_attachment(path) else f"Attached {name}")
        self.refresh_send()
        return True

    async def pick_and_attach(self, images: bool = False) -> Optional[str]:
        if self.env is None or self.env.pick_files is None:
            self.notify("File picking is not available in this build")
            return None
        paths = await self.env.pick_files(_IMAGE_EXTENSIONS if images else _ATTACH_EXTENSIONS, False)
        if not paths:
            return None
        path = paths[0]
        if self.env.import_file is not None:
            path = await self.env.run_io(self.env.import_file, path)
        if path and self.attach_file(path):
            return path
        return None

    def _on_pill(self, pill_id: str) -> None:
        self.open_chat_settings()

    def _on_pill_reset(self, pill_id: str) -> None:
        if not self.bound:
            return
        field_name = {"glossary": "glossary_override_mode", "thinking": "disable_thinking",
                      "multipass": "force_multipass_off", "target": "target_language", "model": "model"}.get(pill_id)
        if field_name:
            self.env.chats.set_override(self.cid, field_name, None)
            self.apply_settings_changed()

    def option_pills(self) -> list:
        """Pills for options that differ from the defaults (UI_SPEC §2.3)."""
        if not self.bound:
            return []
        settings = self.settings()
        overrides = self.env.chats.overrides(self.cid)
        pills = []
        if settings.glossary_override_mode in ("manual", "no_glossary", "none"):
            pills.append(("glossary", {"manual": "Glossary: Manual", "no_glossary": "Glossary: Off",
                                       "none": "Glossary: Main"}[settings.glossary_override_mode]))
        if settings.disable_thinking:
            pills.append(("thinking", "Thinking off"))
        if not settings.force_multipass_off:
            pills.append(("multipass", "Multipass on"))
        if overrides.get("target_language"):
            pills.append(("target", f"→ {overrides['target_language']}"))
        if overrides.get("model"):
            pills.append(("model", f"Model: {overrides['model']}"))
        return pills

    def apply_settings_changed(self) -> None:
        if not self.bound:
            return
        self.state.chat_context.set(self.chat_context_for(self.cid))
        self.composer.set_pills(self.option_pills())
        self.render_transcript()
        self.refresh_send()

    def _on_signin_changed(self) -> None:
        if self.env is not None and self.env.oauth is not None:
            signed = frozenset(self.env.oauth.signed_in)
            current = self.state.signed_in.value
            if signed != current:
                self.state.signed_in.set(signed)
        self.refresh_send()

    # ---- fixes / header actions -------------------------------------------------------------

    def run_fix(self, fix_action: Optional[str]) -> None:
        """Fix buttons of a blocked Send (§2.4) and of the status caption."""
        if fix_action == "sign_in_chatgpt":
            if self.env is not None and self.env.oauth is not None:
                self.open_login_sheet()
            else:
                self.navigate("settings.accounts")
        elif fix_action == "choose_model":
            self.open_model_sheet("model")
        elif fix_action == "open_diagnostics":
            self.navigate("settings.logs")
        elif fix_action == "add_key":
            self.navigate("settings.keys")
        elif fix_action == "provide_glossary":
            self.open_manual_glossary(lambda source: None)
        elif fix_action == "remove_attachment":
            self.composer._remove_attachment()
        elif fix_action:
            self.notify("This fix arrives in a later milestone")

    async def _on_menu(self, e: Any = None) -> None:
        if self.open_drawer is not None:
            result = self.open_drawer()
            if asyncio.iscoroutine(result):
                await result

    def _on_new_chat(self, e: Any = None) -> None:
        # Desktop _new_chat rule: an empty current chat is reused.
        if not self.bound:
            if self.transcript.is_empty and not self.composer.has_content:
                self.notify("This chat is already empty")
            else:
                self.notify("New chats need the chat store, which is not available in this build")
            return
        cid = self.env.chats.new_chat()
        if self.state.current_chat.value != cid:
            self.state.current_chat.set(cid)
        else:
            self.load_chat(cid)

    def _on_new_scratch(self, e: Any = None) -> None:
        self.notify("Scratch chats arrive in U7")

    def _on_menu_action(self, action: str) -> None:
        cid = self.state.current_chat.value
        if action == "chat_settings":
            if self.bound:
                self.open_chat_settings()
            else:
                self.navigate("chat.settings", {"cid": cid})
        elif action == "attachments":
            self.navigate("chat.attachments", {"cid": cid})
        elif action == "delete":
            self.confirm_delete()
        elif action == "text_size":
            self.open_chat_settings()
        else:
            self.notify("This chat action arrives in U7")

    def _on_suggestion(self, suggestion: str) -> None:
        if suggestion == "open_library":
            self.navigate("library")
        elif suggestion == "manga_page":
            self.navigate("tools.manga")
        elif suggestion == "attach_book":
            self._spawn(self.pick_and_attach())
        elif suggestion == "paste_text":
            try:
                asyncio.ensure_future(self.composer.text_field.focus())
            except Exception:
                pass

    def _on_expand(self) -> None:
        self.navigate("chat.compose", {"cid": self.state.current_chat.value})

    def _on_strip_stop(self) -> None:
        if self.env is not None and self.env.jobs is not None and self.env.jobs.snapshot() is not None:
            self.env.jobs.request_stop(force=False)
        else:
            self.notify("No job is running")

    def confirm_delete(self) -> Optional[ConfirmDialog]:
        if not self.bound:
            self.notify("Deleting chats needs the chat store")
            return None
        if self.env.runs is not None and self.env.runs.live_run(self.cid) is not None:
            self.notify("Stop or finish the current translation before deleting this chat.")
            return None
        title, body = self.env.chats.delete_notice(self.cid)
        cid = self.cid

        async def confirm() -> None:
            ok, error = await self.env.run_io(self.env.chats.delete, cid)
            if not ok:
                self.notify(error or "Could not delete chat output")
                return
            new_cid = self.env.chats.current_cid()
            self.state.current_chat.set(new_cid)
            self.load_chat(new_cid)

        dialog = ConfirmDialog(title=title, body=body, confirm_label="Delete", cancel_label="Cancel",
                               destructive=True, on_confirm=confirm)
        dialog.show(self.page)
        return dialog

    def open_rename(self) -> Optional[ft.AlertDialog]:
        if not self.bound:
            self.notify("Renaming chats needs the chat store")
            return None
        session = self.env.chats.session(self.cid) or {}
        field = ft.TextField(label="Chat name:", value=str(session.get("title") or ""), autofocus=True)

        def save(e: Any = None) -> None:
            self.page.pop_dialog()
            if self.env.chats.rename(self.cid, field.value or ""):
                self.header.set_title(self.env.chats.session(self.cid).get("title"))
                self._push(self.header.wrapper)

        dialog = ft.AlertDialog(
            title=ft.Text("Rename chat"),
            content=field,
            actions=[ft.TextButton(content="Cancel", on_click=lambda e: self.page.pop_dialog()),
                     ft.FilledButton(content="OK", on_click=save)],
        )
        self.page.show_dialog(dialog)
        return dialog

    # ---- message actions -------------------------------------------------------------------

    def _copy_message(self, card: AssistantMessage) -> None:
        text = card.content_text
        if self.env is not None and self.env.copy_text is not None:
            result = self.env.copy_text(text)
            if asyncio.iscoroutine(result):
                self._spawn(result)
        card.show_copied()

        async def reset() -> None:
            await asyncio.sleep(COPIED_SECONDS)
            card.reset_copied()

        self._spawn(reset())

    def _source_for(self, index: int) -> Optional[tuple]:
        for message in reversed(self._messages()[:index]):
            if message and message[0] in ("user", "user_file"):
                return message
        return None

    def _retranslate(self, card: AssistantMessage) -> None:
        source = self._source_for(card.index)
        if source is None:
            self.notify("The source of this response is no longer in the chat")
            return
        if source[0] == "user":
            self.composer.set_text(str(source[1] or ""))
        else:
            if not self.attach_file(str(source[2])):
                return
            self.composer.set_text(str(source[4] or "") if len(source) > 4 else "")
        self.on_send_action(SendAction.SEND)

    def _message_more(self, card: AssistantMessage) -> ActionSheet:
        mid = self._mid(card.index)
        reason = None
        try:
            from glossarion_mobile.ui.screens.output_editor import edit_availability

            reason = edit_availability(self.env.chats, self.cid, card.index) if self.bound else "Unavailable"
        except Exception:
            reason = "Unavailable"
        message = self._messages()[card.index] if self.bound and 0 <= card.index < len(self._messages()) else None
        folder = str(message[4] or "") if message is not None and len(message) > 4 else ""
        items = [
            ActionItem("Edit translation", (lambda: self.navigate("chat.message.edit", {"cid": self.cid, "mid": mid}))
                       if mid else None, icon="EDIT", disabled_reason=reason),
            ActionItem("Copy", lambda: self._copy_message(card), icon="CONTENT_COPY"),
            ActionItem("Open output folder", (lambda: self.env.open_output(folder)) if (folder and self.env is not None
                       and self.env.open_output is not None) else None, icon="FOLDER_OPEN",
                       disabled_reason=None if folder else "No output folder for this response"),
            self._add_term_item(folder),
            ActionItem("Delete message", disabled_reason="Arrives in U7", icon="DELETE_OUTLINE", destructive=True),
        ]
        sheet = ActionSheet(items, title=card.request_label or "Response", tablet=bool(self.layout.persistent_sidebar))
        sheet.show(self.page)
        return sheet

    def _add_term_item(self, folder: str) -> ActionItem:
        """Response ⋯ › Add term to glossary: a new entry in the workspace's glossary.csv (GlossaryFeature)."""
        adder = self.glossary_term_adder
        glossary = os.path.join(folder, "glossary.csv") if folder else ""
        if adder is None:
            return ActionItem("Add term to glossary", disabled_reason="The Glossary Manager is not available",
                              icon="BOOKMARK_ADD")
        if not glossary or not os.path.isfile(glossary):
            return ActionItem("Add term to glossary", disabled_reason="This response's workspace has no glossary.csv",
                              icon="BOOKMARK_ADD")
        return ActionItem("Add term to glossary", lambda: adder("", glossary_path=glossary), icon="BOOKMARK_ADD")

    def _user_actions(self, bubble: UserBubble) -> ActionSheet:
        sheet = ActionSheet(
            [
                ActionItem("Copy", lambda: self._spawn_copy(bubble.text), icon="CONTENT_COPY"),
                ActionItem("Translate again", lambda: (self.composer.set_text(bubble.text),
                                                       self.on_send_action(SendAction.SEND)), icon="REFRESH"),
                ActionItem("Edit & resend", disabled_reason="Arrives in U7", icon="EDIT"),
            ],
            title="Message",
        )
        sheet.show(self.page)
        return sheet

    def _spawn_copy(self, text: str) -> None:
        if self.env is not None and self.env.copy_text is not None:
            result = self.env.copy_text(text)
            if asyncio.iscoroutine(result):
                self._spawn(result)

    def _show_full(self, card: AssistantMessage) -> None:
        InfoSheet(title=card.request_label or "Translation", body=card.content_text).show(self.page)

    def _open_request(self, segment: dict) -> None:
        index = segment.get("index")
        if isinstance(index, int) and self.bound:
            segment = dict(segment)
            segment["content"] = self.env.chats.message_text(self.cid, index, "content")
            segment["thinking"] = self.env.chats.message_text(self.cid, index, "thinking")
        RequestSheet(segment).show(self.page)

    # ---- sheets --------------------------------------------------------------------------

    def open_plus_sheet(self) -> PlusSheet:
        self.plus_sheet = PlusSheet(
            mode_signal=self.state.output_mode,
            row_style=self.layout.output_row,
            on_attach=self._on_attach,
            on_attach_long_press=self._on_attach_long_press,
            on_tool=self._on_tool,
            on_this_chat=self._on_this_chat,
            on_open_mode_options=self.open_mode_options,
            on_dismiss=lambda e: self.composer.set_plus_open(False),
        )
        self._haptic("light_impact")
        self.composer.set_plus_open(True)
        self.plus_sheet.show(self.page)
        return self.plus_sheet

    def _on_attach(self, tile_id: str) -> None:
        self.composer.set_plus_open(False)
        if tile_id == "library":
            self.navigate("library")
        elif tile_id in ("files", "photos"):
            self._spawn(self.pick_and_attach(images=tile_id == "photos"))
        elif tile_id == "clipboard":
            self._spawn(self._paste_clipboard())
        else:
            self.notify("This source arrives in a later milestone")

    async def _paste_clipboard(self) -> None:
        reader = getattr(self.env, "read_clipboard", None) if self.env is not None else None
        if reader is None:
            self.notify("Clipboard is not available")
            return
        try:
            text = reader()
            if asyncio.iscoroutine(text):
                text = await text
        except Exception:
            text = None
        if text:
            self.composer.handle_text((self.composer.text or "") + text)

    def _on_attach_long_press(self, tile_id: str) -> None:
        self.composer.set_plus_open(False)
        if tile_id == "files":
            self.notify("Pick folder… arrives in U5")

    def _on_tool(self, tool_id: str) -> None:
        self.composer.set_plus_open(False)
        route = TOOL_ROUTES.get(tool_id)
        if route is not None:
            self.navigate(route)
        else:
            self.notify(_NOT_YET.get(tool_id, "This tool arrives in a later milestone"))

    def _on_this_chat(self, item_id: str) -> None:
        self.composer.set_plus_open(False)
        if item_id == "chat_settings" or item_id == "glossary_policy":
            if self.bound:
                self.open_chat_settings()
            else:
                self.navigate("chat.settings", {"cid": self.state.current_chat.value})

    def open_mode_options(self, mode_id: str) -> ModeOptionsSheet:
        from glossarion_mobile.ui.sheets.model_sheet import sheet_env

        self.mode_sheet = ModeOptionsSheet(mode_id, ctx=sheet_env().ctx)  # ctx: the settings tiles (U4)
        self.mode_sheet.show(self.page)
        return self.mode_sheet

    def _profiles(self) -> list:
        if self.env is None:
            return []
        try:
            names = list(self.env.profiles() or [])
        except Exception:
            names = []
        return names

    def open_chat_settings(self) -> Any:
        if not self.bound:
            self.notify("Chat settings need the chat store")
            return None
        from glossarion_mobile.ui.sheets.chat_settings import ChatSettingsSheet

        self.settings_sheet = ChatSettingsSheet(
            cid=self.cid,
            config=self.env.store,
            chats=self.env.chats,
            profiles=self._profiles(),
            languages=self.env.languages,
            on_changed=self.apply_settings_changed,
            on_choose_model=lambda scope: self.open_model_sheet("model", chat_scope=scope == "chat"),
        )
        self.settings_sheet.show(self.page)
        return self.settings_sheet

    def open_model_once(self) -> Any:
        """Long-press Send -> "Translate once with another model…": the ModelSheet in one-shot mode
        ("Use once"); the choice overrides only this send, never the global key or the chat."""
        if not self.bound or self.env is None or self.env.runs is None:
            self.notify("Translating needs the job service, which is not running in this build.")
            return None
        text = self.composer.send_text()
        if not text and not self.composer.attachment:
            self.notify("Enter text or attach a TXT, EPUB, PDF, CBZ, or image file to translate.")
            return None
        return self.open_model_sheet("model", one_shot=True)

    def open_model_sheet(self, tab: str = "model", chat_scope: bool = False, one_shot: bool = False) -> Any:
        """The ModelSheet (UI_SPEC §2.2): catalog search, favourites, provider groups, route row."""
        context = self.state.chat_context.value
        if self.env is None:
            titles = {
                "model": ("Model", context.model),
                "profile": ("Prompt profile", context.profile),
                "language": ("Target language", context.target_language),
            }
            title, value = titles.get(tab, titles["model"])
            self.model_sheet = InfoSheet(
                title=title,
                body=f"Current: {value}\n\nThe model picker needs the chat services, which are not running in this build.",
                actions=[
                    ft.TextButton(content="Manage models", on_click=lambda e: self._sheet_nav("settings.models")),
                    ft.TextButton(content="Accounts", on_click=lambda e: self._sheet_nav("settings.accounts")),
                ],
            )
            self.model_sheet.show(self.page)
            return self.model_sheet
        from glossarion_mobile.services.oauth import sign_in_satisfied
        from glossarion_mobile.ui.sheets.model_sheet_min import ModelSheetMin, load_model_catalog

        signed = self.state.signed_in.value
        sheet = ModelSheetMin(
            current_model=context.model,
            current_profile=context.profile,
            current_language=context.target_language,
            profiles=self._profiles(),
            languages=self.env.languages,
            signed_in=lambda model: sign_in_satisfied(model, signed),
            tab=tab,
            chat_scope=chat_scope and not one_shot,
            one_shot=one_shot,
            on_select=self._on_model_once_select if one_shot else self._on_model_sheet_select,
            on_sign_in=self.open_login_sheet,
            on_accounts=lambda: self.navigate("settings.accounts"),
        )
        self.model_sheet = sheet
        sheet.show(self.page)

        async def load() -> None:
            try:
                models = await self.env.run_io(load_model_catalog, self.env.config_get)
            except Exception as exc:
                log.info("model catalog unavailable: %s", exc)
                return
            sheet.set_models(models)

        self._spawn(load())
        return sheet

    def _on_model_once_select(self, field_name: str, value: str, this_chat_only: bool = False) -> None:
        """One-shot choice: send now with ``{override key: value}`` for this run only."""
        key = _SHEET_OVERRIDE_KEYS.get(field_name)
        if key and value:
            self.begin_send(once={key: value})

    def _on_model_sheet_select(self, field_name: str, value: str, this_chat_only: bool) -> None:
        key = {"model": "model", "profile": "active_profile", "language": "output_language"}[field_name]
        override_key = {"model": "model", "profile": "profile", "language": "target_language"}[field_name]
        if this_chat_only and self.bound:
            self.env.chats.set_override(self.cid, override_key, value)
        else:
            self.env.config_set_many({key: value})
        self.apply_settings_changed()

    def _sheet_nav(self, route_name: str) -> None:
        if self.model_sheet is not None:
            self.model_sheet.close()
        self.navigate(route_name)

    def open_manual_glossary(self, on_use: Callable[[ManualGlossarySource], Any]) -> Any:
        from glossarion_mobile.ui.sheets.manual_glossary import ManualGlossarySheet

        def use(source: ManualGlossarySource) -> None:
            self.last_manual_glossary = source
            on_use(source)

        async def pick() -> Optional[str]:
            if self.env is None or self.env.pick_files is None:
                return None
            paths = await self.env.pick_files(["csv", "json", "txt", "md"], False)
            return paths[0] if paths else None

        self.glossary_sheet = ManualGlossarySheet(
            on_use=use,
            on_cancel=lambda: self.caption.show("Manual glossary required"),
            pick_file=pick,
            initial=self.last_manual_glossary,
            mono=getattr(self.env, "mono", "monospace") if self.env is not None else "monospace",
        )
        self.glossary_sheet.show(self.page)
        return self.glossary_sheet

    def open_login_sheet(self) -> Any:
        """"Sign in with ChatGPT" (blocked Send, status caption): the LoginSheet of the ChatGPT slot
        the current model uses (``authgpt2/`` -> #2; ``authgpt/`` and the ``authgpt0/`` pool -> #0)."""
        if self.env is None or self.env.oauth is None:
            self.navigate("settings.accounts")
            return None
        from glossarion_mobile.services.oauth import sign_in_slot
        from glossarion_mobile.ui.screens.accounts import LoginSheet

        account_id = sign_in_slot(self.state.chat_context.value.model, "authgpt")
        self.login_sheet = LoginSheet(self.env.oauth, provider="authgpt", account_id=account_id,
                                      on_done=lambda status: self._on_signin_changed())
        self.login_sheet.show(self.page)
        return self.login_sheet
