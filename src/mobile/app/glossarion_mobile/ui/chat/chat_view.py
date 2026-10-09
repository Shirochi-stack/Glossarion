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
  the manual glossary sheet and the ChatGPT LoginSheet;
* U7: generated media in the cards (image gallery / VideoCard / AudioCard, the full-screen
  MediaViewer), "Generate from prompt (no input)" (a ``generate_media`` job), Refine's
  "Compare with original", message versions ‹2/3› (Edit & resend, Retranslate), Delete
  message, scratch chats (New / Send as scratch / Duplicate as scratch, Save / Discard), the
  Attachments manager, Jump to…, Search in chat, Export chat, and the ＋ sheet's
  "Retranslate chapters" (the Progress manager's Chapters on this chat's workspace).

Device fixes (2026-10-08): a finished book's workspace moves into the Library by itself (ChatFeature
auto-migrate calls ``on_workspace_migrated``), so the Result card offers "Open in Library" instead of
Migrate; a QA scan runs from the Result card, the ＋ sheet and ``/qa`` (Quick Scan, the job's card is a
"QA scan" message, progress in the JobStrip); "Always accept" on the glossary card; a Library book is
attached from inside the chat (＋ › From Library, the empty-chat chip, ``/library [title]``: the
searchable picker) and Send then defaults to "Save to: Library" (the book's own workspace).

U9: Chat settings › Text size (85–150 %, per chat, sidecar ``text_scale``) scales the
transcript and composer text: the column sits in a Container whose nested theme carries the
text styles at Appearance × chat scale (Flet merges a nested theme onto the page theme, so
the colours stay the app's). On tablets chat settings and Compare open in the SidePanel.

All Flet mutation happens on the UI loop; worker-thread notifications arrive
through ``ChatEnv.on_ui``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import time
import types
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.services.background import IOS_BACKGROUND_NOTICE
from glossarion_mobile.state.app_state import AppState, ChatContext, JobStripModel
from glossarion_mobile.ui.chat.cards import (
    LIBRARY_WAIT_REASON,
    GlossaryApprovalCard,
    GlossaryEditorView,
    JobCard,
    RequestSheet,
    glossary_preview,
)
from glossarion_mobile.ui.chat.chat_ops import (
    COPY_FORMATS,
    build_chat_export_zip,
    copy_text_for,
    glossary_terms_markdown,
    jump_entries,
    safe_file_stem,
    search_matches,
    source_text_for,
    transcript_markdown,
    turn_workspace,
    version_view,
    workspace_outputs,
)
from glossarion_mobile.ui.chat.chat_ops import moved_workspace as _moved_workspace
from glossarion_mobile.ui.chat.composer import Composer, StatusCaption
from glossarion_mobile.ui.chat.direct_text_rules import (
    AUTO_ACCEPT_GLOSSARY_PREF,
    TOKEN_HINT_MIN_CHARS,
    DirectTextSettings,
    ManualGlossarySource,
    attachment_text_chars,
    count_tokens,
    effective_glossary_label,
    is_supported_attachment,
    merge_meta_overrides,
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
    job_kind,
    progress_counts,
    progress_line,
    running_label,
    state_name,
)
from glossarion_mobile.ui.chat.media_cards import AudioHub, CompareSheet, MediaActions, media_section
from glossarion_mobile.ui.chat.media_model import GENERATIVE_MODES, find_unrefined_backup, media_items, ocr_entries
from glossarion_mobile.ui.chat.messages import AssistantMessage, UserBubble, UserFileCard, VersionSwitcher
from glossarion_mobile.ui.chat.mode_options_sheet import ModeOptionsContent, ModeOptionsSheet
from glossarion_mobile.ui.chat.output_modes import (
    IMAGE_ATTACHMENT_EXTENSIONS,
    OutputModeState,
    is_vision_attachment,
    normalize_mode,
)
from glossarion_mobile.ui.chat.plan_card import ChooseChaptersSheet, RunOptionsPanel, destination_label, destination_sheet
from glossarion_mobile.ui.chat.plan_model import NO_FILES_TEXT, batch_files, glossary_chip_label, plan_facts, range_preview
from glossarion_mobile.ui.chat.run_request import attachment_record
from glossarion_mobile.ui.chat.send_state import (
    BLOCK_ATTACHMENT_MISSING,
    BLOCK_GLOSSARY_PENDING,
    SendAction,
    SendInputs,
    SendState,
    excluded_route_reason,
)
from glossarion_mobile.ui.chat.quick_chips import QuickChips
from glossarion_mobile.ui.chat.stream_bridge import label_request_number, segment_request_number
from glossarion_mobile.ui.chat.transcript import CardSlot, Transcript
from glossarion_mobile.ui.chat.transcript_model import (
    ACTIONS_LABEL,
    QA_LABEL,
    REPORT_LABEL,
    build_items,
    item_position,
    item_sizes,
    shift_window,
    tail_start,
    window_after_append,
    window_around,
)
from glossarion_mobile.ui.components.action_sheet import ActionItem, ActionSheet
from glossarion_mobile.ui.components.dialogs import ConfirmDialog, close_dialog
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.foreground import park_while_hidden
from glossarion_mobile.ui.responsive import Layout, layout_for
from glossarion_mobile.ui.sheets.plus_sheet import PlusSheet
from glossarion_mobile.ui.shell.job_strip import JobStrip
from glossarion_mobile.ui.theme import build_text_theme

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
    "retranslate": None,  # the Progress manager's Chapters on this chat's workspace (open_retranslate)
}
_NOT_YET = {
    "extract_glossary": "Glossary extraction is not available in this session",
}
_LATER: dict = {}
SCRATCH_BANNER = "Scratch chat — not saved"
HIGHLIGHT_SECONDS = 1.5
#: Chat settings › Text size bounds (UI_SPEC §2.14: 85–150 %, per chat).
CHAT_TEXT_SCALE_RANGE = (0.85, 1.5)
# ModelSheet field -> chat override key (run_request.OVERRIDE_CONFIG_KEYS maps them to config keys)
_SHEET_OVERRIDE_KEYS = {"model": "model", "profile": "profile", "language": "target_language"}
_ATTACH_EXTENSIONS = [
    "txt", "epub", "pdf", "md", "markdown", "html", "htm", "xhtml", "xml", "json", "csv", "tsv", "srt", "ass",
    "lrc", "vtt", "log", "sdlxliff", "zip", "cbz", "mp4", "png", "jpg", "jpeg", "gif", "bmp", "webp", "tif",
    "tiff", "svg", "ico", "heic", "heif", "avif", "jxl",
]
_IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "gif", "bmp", "webp", "tif", "tiff", "heic", "heif", "avif", "jxl"]
COPIED_SECONDS = 1.6
_TERMINAL_JOB_STATES = ("DONE", "FAILED", "STOPPED", "CANCELLED", "INTERRUPTED")
NOTHING_TO_SCAN = "Nothing to scan in this turn's workspace"
#: chat sidecar meta key: the Library book attached to this chat (``{"bid", "path"}``)
LIBRARY_ATTACHMENT_META = "library_attachment"
#: chat sidecar meta key: ``{created_at: data}`` of the chat's tool-job card messages (Library job, QA scan)
TOOL_JOBS_META = "tool_jobs"


def _same_path(a: Any, b: Any) -> bool:
    if not a or not b:
        return False
    return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))


def library_attach_reason(target: Any) -> Optional[str]:
    """Why a Library row cannot be attached to a chat (None: it can): its raw file must be on this
    device and be a file the chat takes (blocking: the picker computes it on the io pool)."""
    source = str(getattr(target, "source", "") or "")
    if not source or not os.path.isfile(source):
        return "No raw file on this device"
    if not is_supported_attachment(source):
        return "Not an attachable file type"
    return None


def library_matches(service: Any, query: str) -> list:
    """Blocking: the attachable Library books (ToolTargets) whose title / raw title / tags match
    ``query`` (the Library search: ``targets.order_library_rows``, newest first like the in-chat
    picker); a single exact title match wins over the partial ones (``/library <title>``)."""
    from glossarion_mobile.ui.tools import targets as tg

    rows = [target for target in tg.library_targets(service) if library_attach_reason(target) is None]
    found = tg.order_library_rows(service, rows, str(query or ""))
    wanted = str(query or "").strip().casefold()

    def name(target: Any) -> str:
        book = (getattr(target, "extra", None) or {}).get("book")
        return str((book or {}).get("name") or getattr(target, "title", "") or "").strip().casefold()

    exact = [target for target in found if wanted and name(target) == wanted]
    return exact if len(exact) == 1 else found


def _library_plan_fields(view: Any, cid: str, record: Optional[dict]) -> dict:
    """Plan fields of a send whose attachment is the chat's Library book (``library_attachment`` meta):
    ``{"destination": "library", "library_bid": bid}`` (the book's own workspace), else {}."""
    if not record:
        return {}
    env = getattr(view, "env", None)
    meta_of = getattr(getattr(env, "chats", None), "meta", None)
    if not callable(meta_of):
        return {}
    try:
        attached = (meta_of(cid) or {}).get(LIBRARY_ATTACHMENT_META)
    except Exception:
        return {}
    if not isinstance(attached, Mapping) or not _same_path(attached.get("path"), record.get("path")):
        return {}
    jobs = getattr(env, "jobs", None)
    if jobs is None or not getattr(jobs, "has_kind", lambda _k: True)("translate"):
        return {}
    return {"destination": "library", "library_bid": str(attached.get("bid") or "")}


def library_line(service: Any, bid: str) -> str:
    """Blocking: the Plan card's "📚 Library · <pill>" for the attached Library book (its Library card's
    progress pill, ``models.card_model_for``: the Library's own card model)."""
    book = service.book_for_bid(bid) if service is not None and bid else None
    if not book:
        return "📚 Library"
    from glossarion_mobile.ui.library.models import card_model_for

    model = card_model_for(service, book, views=getattr(getattr(service, "snapshot", None), "views", None) or {},
                           raw_titles=False, dark=False)
    pill = " ".join(p for p in (model.pill_text or "", model.pct_text or "") if p).strip()
    return f"📚 Library · {pill or model.title}"


def append_tool_message(chats: Any, cid: str, body: str, folder: str, label: str, data: dict) -> None:
    """Append the card message of a job a chat handed to a tool (``LIBRARY_LABEL`` / ``QA_LABEL``).

    The desktop history keeps only its own storage keys (``ChatStore._normalize_message_storage``
    drops a job id, source or book id on the first save), so ``data`` also goes into the chat's
    sidecar (``TOOL_JOBS_META``) under the message's ``created_at``, which the history keeps
    (microseconds make it unique); ``ChatView._tool_storage`` reads it back."""
    from datetime import datetime

    created = datetime.now().astimezone().isoformat(timespec="microseconds")
    chats.append_messages(cid, [("assistant", body, "", "", folder, label, dict(data, created_at=created))])
    meta_of = getattr(chats, "meta", None)
    saved = meta_of(cid).get(TOOL_JOBS_META) if callable(meta_of) else None
    saved = dict(saved) if isinstance(saved, dict) else {}
    saved[created] = dict(data)
    chats.set_meta(cid, TOOL_JOBS_META, saved)


def _in_attachments(folder: str) -> bool:
    """A chat's ``Attachments/<stem>`` workspace (not yet moved into the Library)."""
    return bool(folder) and os.path.basename(os.path.dirname(os.path.normpath(folder))).lower() == "attachments"


def qa_scannable(folder: str, source: str = "") -> bool:
    """Blocking: the QA Scanner's folder check for a chat scan (text mode for a TXT / PDF source that
    is still on disk, as the scan will run)."""
    from glossarion_mobile.ui.tools import targets as tg

    if not folder or not os.path.isdir(folder):
        return False
    text_mode = str(source or "").lower().endswith((".txt", ".pdf")) and os.path.isfile(str(source))
    return tg.folder_has_scan_files(folder, text_mode=text_mode)


def workspace_failed_count(folder: str) -> Optional[int]:
    """Blocking: failed + QA-failed chapters in a workspace's ``translation_progress.json`` (the Result
    card's "N QA failed" chip after a QA scan marked chapters); None without a progress file."""
    path = os.path.join(str(folder or ""), "translation_progress.json")
    if not folder or not os.path.isfile(path):
        return None
    import progress_core  # shared (U5)

    chapters = progress_core.load_progress(path).get("chapters")
    if not isinstance(chapters, Mapping):
        return 0
    return sum(1 for entry in chapters.values()
               if isinstance(entry, Mapping) and str(entry.get("status") or "") in ("failed", "qa_failed"))


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
        # Set by GlossaryFeature (U9): the PlanGlossarySheet behind the Plan / Batch card glossary chip.
        self.plan_glossary_opener: Optional[Callable[..., Any]] = None
        # U9 Plan card: Run options panels kept across card rebuilds, (cid, plan created) -> panel
        self.run_panels: dict = {}
        self.chapters_sheet: Any = None
        self.batch_card: Any = None
        self.model_sheet: Any = None
        self.settings_sheet: Any = None
        self.glossary_sheet: Any = None
        self.login_sheet: Any = None
        self.cid = str(state.current_chat.value)
        # The rendered window over the transcript's cards (``_window_items``: the whole chat's items
        # without hidden versions), (start, end); None = the newest cards. ``_window_total`` is the card
        # count it was computed for: the desktop rule re-tails only after cards were added to a window
        # that ended at the old tail (``window_after_append``).
        self.window: Optional[tuple] = None
        self._window_total = 0
        self.items: list = []  # the items of the latest render (the window indexes them)
        self._row_page = 20  # a JobCard's page of request rows (the rendered-card limit at the last render)
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
        self.audio_hub = AudioHub()  # one flet_audio service for every AudioCard of the chat
        self.media_viewer: Any = None
        self.jump_sheet: Any = None
        self.version_anchor: Optional[int] = None  # Edit & resend / Retranslate: the turn the next send versions
        self.hidden_indices: set = set()
        self.search_hits: list = []
        self.search_position = -1
        self._search_task: Any = None
        self.focus_index: Optional[int] = None
        # Card extras that need file I/O (a Vision run's OCR texts, the Refine backup): looked up
        # on the io pool once per key and cached, never read on the loop during a render.
        self._extras: dict = {}
        self._extras_pending: set = set()
        self._extra_targets: dict = {}  # key -> (render generation, [(apply, card)])
        self._render_gen = 0
        # Saved cards of the latest render, slot key -> (signature, card): a card whose inputs did not
        # change is passed again as the same object (Transcript.CardSlot: never a frozen copy).
        self._cards: dict = {}
        self._next_cards: dict = {}
        self.library_picker: Any = None
        self._tool_states: dict = {}  # chat QA job id -> last state seen (its card re-renders on a change)
        self._library_starting: set = set()  # Plan "created" stamps whose Library start is under way

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
            on_save_scratch=lambda e: self.save_scratch(),
            on_search=self._on_search_query,
            on_search_step=self.step_search,
            on_search_close=self.close_search,
        )
        self.transcript = Transcript(
            on_suggestion=self._on_suggestion,
            on_scroll=self.header.on_transcript_scroll,
            # a loader row's tap shows the loaded cards; a scroll at the edge (auto) keeps the viewport
            on_load_earlier=lambda auto=False: self.slide(-1, keep_view=auto),
            on_load_later=lambda auto=False: self.slide(1, keep_view=auto),
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
        # UI_SPEC §2.7 quick-action chips (with an attachment): Translate · Extract glossary first ·
        # Translate as manga · Open in Reader
        self.quick_chips = QuickChips(on_select=self._on_quick_chip)
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
            on_slash=self.run_slash,
        )
        self.column = ft.Column(
            [self.transcript_stack, self.job_strip, self.quick_chips.control, self.caption,
             self.composer.slash.control, self.composer],
            spacing=0,
            expand=True,
        )
        # per-chat Text size: a nested theme over the column (``apply_chat_text_scale``)
        self.chat_text_scale = 1.0
        self._chat_theme_key: tuple = (1.0, 1.0)
        self.scale_box = ft.Container(content=self.column, expand=True, key="chat-scale-box")
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
            inner: ft.Control = self.scale_box
        else:
            inner = ft.Row(
                [ft.Container(width=layout.chat_max_width, content=self.scale_box)],
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
        self.header.set_text_scale(layout.text_scale)  # the bar grows with large text (UI_SPEC §7.5)
        self.header.set_width(layout.width)  # narrow phones: compact scratch-chat actions

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
            state.text_scale.subscribe(lambda _v: self.apply_chat_text_scale()),
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
        # On the chat home the strip shows only for another chat's job (§1.7) - and for this chat's
        # jobs that have no Job card here (Extract glossary, Compile, a Library translation started
        # from the Plan card): the strip is their progress line, Stop and "Done · Open" (U9).
        if (model is not None and model.owner_chat == self.state.current_chat.value
                and getattr(model, "chat_card", True)):
            model = None
        self.job_strip.set_model(model)

    def _on_job_strip(self, model: Optional[JobStripModel]) -> None:
        self._apply_job_strip(model)
        self.refresh_send()  # another chat's job running -> Send queues

    # ---- chat loading / settings ---------------------------------------------------

    def settings(self, cid: Optional[str] = None) -> DirectTextSettings:
        """The chat's effective Direct Text settings: config.json, the mobile-only Prefs globals
        ("Always accept generated glossaries"), then the chat's overrides and sidecar meta fields."""
        cid = str(cid or self.cid)
        if self.env is None:
            return DirectTextSettings()
        prefs = getattr(self.env, "prefs", None)
        base = DirectTextSettings.from_config(self.env.config_get,
                                              prefs_get=prefs.get if prefs is not None else None)
        if not self.bound:
            return base
        return base.with_overrides(merge_meta_overrides(self.env.chats.overrides(cid), self.env.chats.meta(cid)))

    def chat_context_for(self, cid: str) -> ChatContext:
        """The header's model · profile · language of chat ``cid``: its own (or its series') values, else
        what it inherits - the same values Chat settings shows and the run uses (never the previous chat's
        header: on a fresh config nothing names a global model or profile)."""
        from glossarion_mobile.ui.sheets.chat_settings import inherited_value

        env = self.env
        overrides = env.chats.overrides(cid) if self.bound else {}
        listing = None
        service = getattr(env, "profile_service", None)
        if service is not None:
            try:
                listing = service.listing()
            except Exception:
                log.debug("prompt profile listing failed", exc_info=True)

        def inherited(field_name: str) -> str:
            return inherited_value(env.config_get, field_name, listing,
                                   self._profiles() if listing is None and field_name == "profile" else ())

        model = overrides.get("model") or inherited("model")
        profile = overrides.get("profile") or inherited("profile")
        target = overrides.get("target_language") or inherited("target_language")
        custom = any(overrides.get(k) is not None for k in overrides)
        return ChatContext(model=str(model), profile=str(profile), target_language=str(target), custom=custom)

    def load_chat(self, cid: Any) -> None:
        """Show chat ``cid``: draft, attachment, mode, header, transcript tail (``_load_chat_session``)."""
        cid = str(cid)
        previous = self.cid
        self.cid = cid
        if not self.bound:
            self._on_current_chat(cid)
            return
        chats = self.env.chats
        if previous != cid:
            self._left_chat(previous)
            sheet = self.settings_sheet
            if sheet is not None and str(getattr(sheet, "cid", cid)) != cid:
                sheet.close()  # the other chat's settings (a tablet SidePanel stays open otherwise)
        chats.select(cid)
        session = chats.session(cid)
        title = str((session or {}).get("title") or "New chat")
        self.header.set_title(title)
        self.header.set_scratch(self._is_scratch(cid))
        self.header.set_attachments(self._attachment_count(cid))
        if self.header.searching:
            self.close_search()
        self.version_anchor = None
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
        self.apply_chat_text_scale(push=False)
        self.state.chat_context.set(self.chat_context_for(cid))
        self.composer.set_pills(self.option_pills())
        self.window = None
        self.live_cards = {}
        self.live_job_card = None
        self.approval_card = None
        self._approval_key = None
        self._cards = {}
        self.show_newest()  # the chat opens on its newest cards (window None), following them
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

    def _window_items(self, messages: list, versions: Any) -> list:
        """The transcript's cards: the whole chat's items (a book turn is one file card + one JobCard
        whatever its size) without the hidden versions. The window indexes this list."""
        hidden = versions.hidden
        return [item for item in build_items(messages) if item.index not in hidden]

    def _window_for(self, items: list, messages: list, expanded: Any, settings: DirectTextSettings) -> tuple:
        """This render's window over ``items`` (the desktop ``_history_visible_start/_end`` rules): the
        newest cards when there is none (``_reset_history_window``); after cards were added or removed
        it follows them only if it ended at the old tail (``_update_history_window_after_append``);
        otherwise it stays where the user slid or jumped it (clamped)."""
        total = len(items)
        previous, self._window_total = self._window_total, total
        if self.window is not None and total != previous:
            self.window = window_after_append(self.window, previous, total)  # None: follow the new cards
        if self.window is not None:
            start = max(0, min(int(self.window[0]), total))
            end = max(start, min(int(self.window[1]), total))
            if end > start or not total:
                return (start, end)
        limit = settings.rendered_card_limit
        return (tail_start(item_sizes(items, messages, expanded, limit), limit), total)

    def render_transcript(self, *, follow: bool = False) -> None:
        """Render the window. ``follow``: scroll to the newest card if the window shows it and the user
        follows the tail; it never moves the window (``show_newest`` / ``slide`` / ``jump_to`` do)."""
        if not self.bound:
            return
        self._render_gen += 1  # card extras still loading go to this render's cards
        messages = self._messages()
        settings = self.settings()
        expanded = self.env.chats.expanded(self.cid)
        run = self.env.runs.live_run(self.cid) if self.env.runs is not None else None
        versions = self._version_view(messages)
        self.hidden_indices = set(versions.hidden)
        items = self._window_items(messages, versions)
        self.items = items
        self.window = start, end = self._window_for(items, messages, expanded, settings)
        self._row_page = settings.rendered_card_limit  # a JobCard's page of request rows
        controls: list = []
        if self._is_scratch(self.cid):
            controls.append(self._scratch_banner())
        self._next_cards = {}
        # the run's card is mounted again below when it is in the window (no repaint of a card off screen)
        self.live_job_card = None
        for item in items[start:end]:
            controls.append(self._item_control(item, messages, expanded, run))
            switcher = versions.switchers.get(item.index) if item.kind in ("user", "user_file") else None
            if switcher is not None:
                anchor, selected, count = switcher
                controls.append(VersionSwitcher(anchor, selected, count, on_select=self.select_version,
                                                key=f"versions-{item.index}"))
        self._cards, self._next_cards = self._next_cards, {}
        self.transcript.set_messages([c for c in controls if c is not None], hidden_before=start,
                                     hidden_after=max(0, len(items) - end))
        self.transcript.set_tail(self._tail_controls(run))
        self.header.set_empty(self.transcript.is_empty and not self.composer.has_content)
        self._push(self.transcript)
        if follow and end >= len(items) and self.transcript.follow_tail and not settings.disable_auto_scroll:
            self._spawn(self.transcript.scroll_to_end())

    def show_newest(self) -> None:
        """Back to the newest cards, following them (an action whose card the user should see: a send, a
        Plan, a QA scan, a batch…)."""
        self.window = None
        self.transcript.follow_tail = True
        self.render_transcript(follow=True)

    def slide(self, direction: int, *, keep_view: bool = False) -> bool:
        """One page of older (-1) / newer (+1) cards (desktop ``_shift_history_window``); False when the
        window cannot move. ``keep_view`` (a scroll at the edge): the card that was at that edge stays in
        view (``Transcript.keep_in_view`` ends the edge load); a loader row's tap shows the loaded cards."""
        if not self.bound:
            return False
        if self.window is None:
            self.render_transcript()
        new = shift_window(self.window, len(self.items), self.settings().rendered_card_limit, direction)
        if new is None:
            return False
        anchor = None
        if keep_view:
            slots = [c for c in self.transcript.messages if isinstance(c, CardSlot)]
            if slots:
                anchor = (slots[0] if direction < 0 else slots[-1]).slot_key
        self.window = new
        if direction < 0:
            # reading older cards: a live run's repaints no longer pull the view to the end (↓ shows instead)
            self.transcript.follow_tail = False
        self.transcript.hold_edge_loads()
        self.render_transcript()
        if keep_view and self._spawn(self.transcript.keep_in_view(anchor)) is None:
            self.transcript.release_edge_load()
        return True

    def _item_control(self, item: Any, messages: list, expanded: set, run: Any) -> Optional[ft.Control]:
        message = messages[item.index] if 0 <= item.index < len(messages) else None
        if item.kind == "user":
            width = self.layout.width or 400
            return self._card(self._mid(item.index) or item.key, ("user", self.cid, item.index, message[1], width),
                              lambda: UserBubble(message[1], index=item.index, available_width=width,
                                                 on_long_press=self._user_actions))
        if item.kind == "user_file":
            missing = not os.path.isfile(str(message[2] if len(message) > 2 else ""))
            thumbnail = os.path.splitext(str(message[1] or ""))[1].lower() in IMAGE_ATTACHMENT_EXTENSIONS
            return self._card(
                self._mid(item.index) or item.key,
                ("user_file", self.cid, item.index, tuple(message[1:6]), missing, thumbnail),
                lambda: UserFileCard(
                    message[1], message[2] if len(message) > 2 else "", message[3] if len(message) > 3 else 0,
                    message[4] if len(message) > 4 else "", message[5] if len(message) > 5 else "user",
                    index=item.index, missing=missing, on_long_press=self._user_file_actions, thumbnail=thumbnail,
                ),
            )
        if item.kind == "assistant":
            return self._assistant_control(item.index, messages[item.index], expanded)
        if item.kind == "job":
            return self._job_control(item, messages, run)
        if item.kind == "qa":
            return self._qa_job_control(item, messages)
        return None

    def _card(self, key: Any, signature: Any, build: Callable[[], ft.Control]) -> Any:
        """The ``CardSlot`` of one saved card: the previous render's card object while ``signature``
        (everything the card is built from) is unchanged, otherwise a new card from ``build``."""
        key = str(key)
        cached = self._cards.get(key)
        card = cached[1] if cached is not None and cached[0] == signature else build()
        self._next_cards[key] = (signature, card)
        return self.transcript.slot(key, card)

    def _mid(self, index: int) -> Optional[str]:
        try:
            return self.env.chats.mid_for_index(self.cid, index)
        except Exception:
            return None

    def _assistant_control(self, index: int, message: tuple, expanded: set) -> Any:
        cid = self.cid
        chats = self.env.chats
        storage = message[6] if len(message) > 6 and isinstance(message[6], dict) else {}
        items = self._media_for(index)
        width = self.layout.width or 400
        is_expanded = index in expanded
        signature = (
            "assistant", cid, index, tuple(message[3:6]), storage.get("created_at"),
            chats.message_text(cid, index, "content"),  # bodies are cached by the store
            chats.message_text(cid, index, "thinking") if is_expanded else None,
            is_expanded, tuple(items), width, bool(self.layout.persistent_sidebar),
        )

        def build() -> AssistantMessage:
            media_control = media_section(items, actions=self.media_actions(), hub=self.audio_hub,
                                          available_width=width, spawn=self._spawn) if items else None
            return AssistantMessage(
                media=items,
                media_control=media_control,
                index=index,
                request_label=str(message[5] if len(message) > 5 else ""),
                created_at=str(storage.get("created_at") or ""),
                processing_label=str(message[3] if len(message) > 3 else "Processing") or "Processing",
                content=lambda i=index: chats.message_text(cid, i, "content"),
                thinking=lambda i=index: chats.message_text(cid, i, "thinking"),
                expanded=is_expanded,
                on_toggle_thinking=lambda card, value: chats.set_expanded(cid, card.index, value),
                on_copy=self._copy_message,
                on_retranslate=self._retranslate,
                on_more=self._message_more,
                on_show_full=self._show_full,
            )

        slot = self._card(self._mid(index) or f"m-{index}", signature, build)
        card = slot.card
        folder = str(message[4] or "") if len(message) > 4 else ""
        if folder:  # Refine › "Compare with original" when the workspace kept the unrefined backup
            label = str(message[5] if len(message) > 5 else "")
            self._io_extra(("refine", cid, folder, label), lambda m=message: self._refine_backup(m),
                           lambda backup, c=card: c.set_compare(
                               (lambda target, b=backup: self.open_compare(target, b)) if backup else None),
                           card)
        return slot

    @staticmethod
    def _saved_request_segments(item: Any, messages: list) -> list:
        """The JobCard rows of a turn's committed request cards (its Result; while it runs, the cards the
        run already froze into the chat - the glossary gate's - before its live ones)."""
        segments = []
        for index in item.requests:
            if not 0 <= index < len(messages):
                continue
            msg = messages[index]
            segments.append({
                "label": str(msg[5] if len(msg) > 5 else ""),
                "content": str(msg[1] or ""),
                "thinking": str(msg[2] or "") if len(msg) > 2 else "",
                "phase": "processing",
                "complete": True,
                "index": index,
            })
        return segments

    def _job_control(self, item: Any, messages: list, run: Any) -> Any:
        # ``item.index`` is always the turn's user_file (build_items): the card keeps its title, size
        # line, live run, Result state and Resume wherever the window starts (owner issues 12 + 13)
        file_message = messages[item.index] if 0 <= item.index < len(messages) else None
        record = None
        if file_message is not None:
            name = str(file_message[1])
            record = {"name": name, "path": str(file_message[2] if len(file_message) > 2 else ""),
                      "extension": os.path.splitext(name)[1].lower(), "size": file_message[3] if len(file_message) > 3 else 0}
        slot_key = f"job-{item.index}"
        live = run is not None and item.index >= 0 and run.user_index == item.index
        row_page = getattr(self, "_row_page", None) or 20
        plan = self._pending_plan()
        if plan is not None and plan.get("user_index") == item.index and not live:
            context = self.state.chat_context.value
            glossary = self._plan_glossary_label()

            def build_plan() -> JobCard:
                card = JobCard(attachment=record, phase=CardPhase("plan"), on_action=self._on_job_action)
                card.set_plan(self._plan_controls(plan, card), start_menu=self._plan_start_menu(plan))
                return card

            return self._card(slot_key, ("plan", self.cid, item.index, record, self._plan_signature(plan), context,
                                         glossary), build_plan)
        if getattr(item, "library", None) is not None and not live:
            return self._library_job_control(item, messages, record)
        if live:
            # The running card stays the same object for the whole run: progress, phase and its
            # request rows are updated in place (here and on every job snapshot). Its rows are the
            # turn's committed cards (the glossary gate freezes its phase into the chat) + the live ones.
            slot = self._card(slot_key, ("live", self.cid, item.index, record),
                              lambda: JobCard(attachment=record, phase=CardPhase("running"), on_action=self._on_job_action,
                                              on_open_request=self._open_request, row_page=row_page,
                                              requests_expanded=True))
            slot.card.saved_requests = self._saved_request_segments(item, messages)
            self.live_job_card = slot.card
            self._update_live_job_card(run)
            return slot
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
        if last is not None and not last.live and self._run_turn(last) == item.index:
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
        segments = self._saved_request_segments(item, messages)
        report = self.env.chats.message_text(self.cid, item.report, "content") if item.report is not None else None

        failed = 0
        if ended is not None:
            snapshot = last.last_snapshot if (last is not None and not last.live and self._run_turn(last) == item.index) \
                else (runs.persisted_job(self.cid, item.index) if runs is not None else None)
            failed = progress_counts(snapshot).get("failed", 0) if snapshot is not None else 0

        def build() -> JobCard:
            card = JobCard(attachment=record, phase=phase, on_action=lambda a, it=item: self._on_job_action(a, it),
                           on_open_request=self._open_request,
                           on_open_output=lambda path, kind, it=item: self._open_output_file(path, kind, it),
                           row_page=row_page)
            card.set_phase(phase, status=status)
            card.set_failed(failed)
            card.set_requests(segments)
            if report is not None:
                card.set_report(report)
            return card

        slot = self._card(slot_key, ("job", self.cid, item, record, phase, status, segments, report, failed, row_page),
                          build)
        card = slot.card
        # UI_SPEC §2.12.4 Result: the turn's output files as chips (its own workspace, on the io pool)
        key = ("outputs", self.cid, self._mid(item.index) or item.index, tuple(item.requests), item.report,
               item.actions, phase.name)
        self._io_extra(key, lambda it=item: self._outputs_for(it),
                       lambda files, c=card: c.set_outputs(files or []), card)
        # "Open in Library" once the workspace left Attachments/ (auto-migrate); the "N QA failed" chip
        # from the workspace's progress file, where a chat QA scan writes its marks
        key = ("library", self.cid, self._mid(item.index) or item.index, tuple(item.requests), item.report,
               item.actions, phase.name)
        self._io_extra(key, lambda it=item: self._library_reason(it),
                       lambda reason, c=card: c.set_action_reason("library", reason), card)
        if not phase.live and phase.name != "plan":
            key = ("failed", self.cid, self._mid(item.index) or item.index, tuple(item.requests), item.report,
                   item.actions, phase.name)
            self._io_extra(key, lambda it=item: workspace_failed_count(self._job_workspace(it)),
                           lambda count, c=card, n=failed: c.set_failed(n if count is None else count), card)
        if item.requests:  # Vision: the run's cached OCR texts (its workspace's OCR folder)
            key = ("ocr", self.cid, self._mid(item.index) or item.index, tuple(item.requests), item.report)
            self._io_extra(key, lambda it=item: self._ocr_for(it),
                           lambda entries, c=card: c.set_ocr(entries or []), card)
        return slot

    def _ocr_for(self, item: Any) -> list:
        """Blocking: ``[(name, text)]`` of a job card's cached OCR (empty when its run had none)."""
        workspace = self._job_workspace(item)
        if workspace and os.path.isdir(os.path.join(workspace, "OCR")):
            return ocr_entries(workspace)
        return []

    def _io_extra(self, key: tuple, compute: Callable[[], Any], apply: Callable[[Any], Any], card: Any) -> None:
        """Put a blocking lookup's result on a card without file I/O on the loop.

        ``compute`` runs on the io pool once per ``key`` and the result is cached (a render
        applies a cached value at once); ``apply`` sets it on the cards the newest render built
        for ``key``, which are then updated. A run of this chat that ends drops the chat's cache.
        """
        if key in self._extras:
            apply(self._extras[key])
            return
        generation = getattr(self, "_render_gen", 0)
        known = self._extra_targets.get(key)
        targets = known[1] if known is not None and known[0] == generation else []
        targets.append((apply, card))
        self._extra_targets[key] = (generation, targets)
        if key in self._extras_pending or self.env is None:
            return
        self._extras_pending.add(key)
        cid = self.cid

        async def load() -> None:
            try:
                value = await self.env.run_io(compute)
            except Exception:
                log.debug("card extra %s failed", key[0], exc_info=True)
                value = None
            finally:
                self._extras_pending.discard(key)
            if self.cid != cid:
                self._extra_targets.pop(key, None)
                return
            if len(self._extras) > 512:
                self._extras.clear()
            self._extras[key] = value
            _generation, targets = self._extra_targets.pop(key, (0, []))
            for fn, target_card in targets:
                try:
                    fn(value)
                    target_card.update()
                except Exception:  # the card left the transcript meanwhile
                    pass

        self._spawn(load())

    def _drop_extras(self, cid: Any) -> None:
        """A run of chat ``cid`` ended: its OCR / Refine lookups may have changed."""
        for key in [k for k in self._extras if len(k) > 1 and k[1] == str(cid)]:
            self._extras.pop(key, None)

    def _tail_controls(self, run: Any) -> list:
        controls: list = []
        if run is None or not run.live:
            self.live_cards = {}
            blocked = self._safety_block()
            if blocked:
                controls.append(self._safety_card(blocked))
            batch = self._pending_batch()
            if batch is not None:
                controls.append(self._batch_control(batch))
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

    #: UI_SPEC §2.13 "Safety block" (exact copy)
    SAFETY_TITLE = "Blocked by the provider's safety filter"

    def _safety_block(self) -> str:
        """The last run of this chat ended on a provider safety / prohibited-use block: its log line."""
        runs = self.env.runs if self.bound else None
        last = runs.run_for(self.cid) if runs is not None else None
        if last is None or last.live:
            return ""
        result = getattr(getattr(last, "last_snapshot", None), "result", None) or {}
        return str(result.get("safety_block") or "") if isinstance(result, Mapping) else ""

    def _safety_card(self, line: str) -> Any:
        from glossarion_mobile.ui.components.error_card import ErrorCard

        key = ("safety", self.cid, line)
        cached = getattr(self, "_safety_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        card = ErrorCard(title=self.SAFETY_TITLE, message=line,
                         actions=[("Retranslate with…", self.open_model_once),
                                  ("Settings › Provider options & safety", self.open_safety_settings)],
                         key="safety-block")
        self._safety_cache = (key, card)
        return card

    def open_safety_settings(self) -> Any:
        """Settings › Chapter Extraction Settings › Disable API Safety Filters (+ threshold)."""
        env = self.env
        opener = getattr(env, "open_setting", None) if env is not None else None
        if callable(opener):
            return opener("other.processing.extraction", "disable_gemini_safety")
        return self.navigate("settings")

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
        prefs = getattr(self.env, "prefs", None) if self.env is not None else None
        self.approval_card = GlossaryApprovalCard(
            path=path, info=info, on_answer=self._answer_glossary, on_edit=self.open_glossary_editor,
            on_always=self._always_accept_glossary if prefs is not None else None,
        )
        self._approval_key = key
        return self.approval_card

    def _always_accept_glossary(self) -> None:
        """The approval card's "Always accept": the All-chats switch of Chat settings ("Always accept
        generated glossaries", Prefs ``chat_auto_accept_glossary``) turns on, then this question is
        answered Yes like ✓ Yes (``RunController.answer_glossary``). Later sends carry it into their job."""
        prefs = getattr(self.env, "prefs", None) if self.env is not None else None
        if prefs is not None:
            prefs.set(AUTO_ACCEPT_GLOSSARY_PREF, True)
        if self.bound and self.env.chats.meta(self.cid).get("auto_accept_glossary") is False:
            self.env.chats.set_meta(self.cid, "auto_accept_glossary", None)  # this chat's "ask" gives way too
        self._answer_glossary(True)
        self.notify("Generated glossaries are accepted automatically from now on", action_label="Chat settings",
                    on_action=lambda: self.open_chat_settings("global"))

    # ---- live updates ----------------------------------------------------------------------

    def _on_run_changed(self, cid: str) -> None:
        if str(cid) != self.cid:
            self.refresh_send()
            return
        run = self.env.runs.run_for(cid)
        if run is not None and not run.live:
            # finished: show the committed cards from the store
            self._drop_extras(cid)
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
        self._track_tool_job(snapshot)
        run = self.env.runs.live_run(self.cid) if self.env.runs is not None else None
        if run is not None and self.live_job_card is not None:
            self._update_live_job_card(run)
            self.live_job_card.push()

    def _track_tool_job(self, snapshot: Any) -> bool:
        """A chat's QA scan changed state: its "QA scan" card shows the new state, and once it ends the
        Result cards re-read their workspaces ("N QA failed" from the progress file the scan marked)."""
        if snapshot is None or job_kind(snapshot) != "qa_scan":
            return False
        origin = getattr(getattr(snapshot, "spec", None), "origin", None)
        cid = str(origin.get("cid") or "") if isinstance(origin, Mapping) else ""
        if not cid:
            return False
        job_id, state = str(getattr(snapshot, "id", "") or ""), state_name(snapshot)
        if self._tool_states.get(job_id) == state:
            return False
        self._tool_states[job_id] = state
        if state in _TERMINAL_JOB_STATES:
            self._drop_extras(cid)
        if cid == self.cid and self.bound:
            self.render_transcript()
        return True

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
        # the turn's committed cards (the glossary gate's) stay listed above the run's live ones (desktop);
        # once the finish has committed the run's cards (the run is still live while it finishes) they are
        # both: the committed ones are listed, as the Result will list them (matched by request number - the
        # commit's label ends in "Request N" and may be renamed, e.g. the header batch's; the shared commit
        # drops lifecycle-only rows such as the EPUB metadata request's)
        saved = card.saved_requests
        rows = segments
        if saved and segments:
            live_numbers = {segment_request_number(s) for s in segments} - {0}
            if live_numbers & {label_request_number(s.get("label")) for s in saved}:
                rows = []
            else:
                live_labels = {str(s.get("label") or "") for s in segments} - {""}
                saved = [s for s in saved if str(s.get("label") or "") not in live_labels]
        card.set_requests(saved + rows)
        active_issue = getattr(snapshot, "active_issue", None)
        try:
            from glossarion_mobile.services.jobs import issue_label

            card.set_issue(issue_label(active_issue()) if callable(active_issue) else "")
        except Exception:
            card.set_issue("")

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
            await park_while_hidden(self.page)  # no repaints in the background; one catch-up on return
            if not run.live or str(run.cid) != self.cid:
                break  # finished meanwhile: _on_run_changed renders the committed cards
            if run.stream.drain():
                self.transcript.set_tail(self._tail_controls(run))
                if self.live_job_card is not None:
                    self._update_live_job_card(run)
                self._push(self.transcript)
                # follow only while the window shows the newest cards and the user is at the bottom: after
                # "↑ earlier" or a jump the view stays where the user is reading, ↓ brings it back
                at_tail = self.transcript.follow_tail and not self.transcript.hidden_after
                if at_tail and not self.settings().disable_auto_scroll:
                    await self.transcript.scroll_to_end(0)
                elif not at_tail:
                    self.new_fab.visible = True
                    self._push(self.new_fab)

    def _on_follow_change(self, follow: bool) -> None:
        if follow and self.new_fab.visible and not self.transcript.hidden_after:
            self.new_fab.visible = False
            self._push(self.new_fab)

    async def _jump_to_end(self, e: Any = None) -> None:
        self.window = None
        self.transcript.follow_tail = True
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
        chips_changed = self._refresh_quick_chips(state)
        if push:
            self._push(self.caption, self.header.wrapper)
            if chips_changed:
                self._push(self.quick_chips.control)
        return state

    # ---- quick-action chips (UI_SPEC §2.7) ----------------------------------------------------------

    def _refresh_quick_chips(self, state: Optional[SendState] = None) -> bool:
        attachment = self.composer.attachment or {}
        path = str(attachment.get("path") or "") if isinstance(attachment, dict) else ""
        busy = state in (SendState.RUNNING, SendState.FINISHING, SendState.STOPPING)
        return self.quick_chips.set_attachment(path or None, hidden=busy)

    def _on_quick_chip(self, chip_id: str) -> None:
        attachment = self.composer.attachment or {}
        path = str(attachment.get("path") or "") if isinstance(attachment, dict) else ""
        if chip_id == "translate":
            self.on_send_action(SendAction.SEND)
        elif chip_id in ("extract_glossary", "manga"):
            self._on_tool(chip_id)  # the glossary / manga features take the attachment from the composer
        elif chip_id == "open_reader":
            opener = self.env.open_reader if self.env is not None else None
            if opener is None or not path:
                self.notify("The Reader is not available in this session")
                return
            result = opener("", path)
            if hasattr(result, "__await__"):
                self._spawn(result)
        else:
            self.notify("This action is not available here")

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
        if action is SendAction.SEND_AS_SCRATCH:
            self.send_as_scratch()
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
        anchor, self.version_anchor = self.version_anchor, None
        # §2.12.1 "a TXT/MD over 20,000 characters": the attached file's characters, not only the message
        chars = max(len(text), attachment_text_chars(record)) if record else len(text)
        # A Library book attached in this chat continues its own workspace: "Save to: Library" (§2.5)
        library = _library_plan_fields(self, cid, record)
        planned = bool(record) and needs_plan(record.get("extension") or "", chars, skip_plan=settings.skip_plan)
        if planned or library:
            from glossarion_mobile.ui.chat.run_request import user_turn

            index = env.chats.record_user_turn(
                cid, user_turn(text, record, settings.attachment_prompt_role), record.get("name") or text
            )
            self._link_version(cid, anchor, index)
            plan = {
                "user_index": index, "text": text, "attachment": dict(record), "output_mode": output_mode,
                "manual_glossary": manual.as_dict() if manual else None, "created": time.time(),
                "once": dict(once) if once else None, **library,
            }
            if planned:
                env.chats.set_meta(cid, "pending_plan", plan)
                self._after_send_ui(output_mode)
                return
            # no Plan card for this send (Chat settings › skip the plan, a short TXT): the book's Library run
            self._after_send_ui(output_mode)
            self._spawn(self._start_library_plan(plan))
            return
        self._after_send_ui(output_mode)
        self.sent.append((cid, text, dict(record) if record else None))
        self._spawn(self._submit(cid, text, record, settings, output_mode, manual, None, once, anchor))

    def _after_send_ui(self, output_mode: str) -> None:
        """Clear the composer and restore the mode non-automatically (desktop after recording the turn)."""
        self.composer.clear()
        self._set_mode(OutputModeState(normalize_mode(output_mode)))
        self.show_newest()
        self.refresh_send()

    def run_overrides(self, cid: str) -> dict:
        """The chat's overrides for a run (``ChatStoreAdapter.overrides``), its prompt profile checked at
        Send: the re-check that follows a renamed profile / clears a deleted one is posted (and waits for
        the Library at start-up), so a chat may still name a profile that is gone - it runs with what it
        inherits (series / All chats), never with the backend's silent first profile."""
        chats = self.env.chats
        overrides = dict(chats.overrides(cid))
        name = overrides.get("profile")
        service = getattr(self.env, "profile_service", None)
        if not name or service is None:
            return overrides
        try:
            listing = service.core_listing()  # None: no desktop core here, the names are unknown
        except Exception:
            listing = None
        if listing is None or not listing.names or name in listing.names:
            return overrides
        reconcile = getattr(self.env, "reconcile_profiles", None)
        if callable(reconcile):
            try:
                reconcile()  # the posted re-check, now: renames followed, gone ones cleared (with its notice)
            except Exception:
                log.exception("re-checking the chats' prompt profiles failed")
            overrides = dict(chats.overrides(cid))
        if overrides.get("profile") and overrides["profile"] not in listing.names:
            overrides.pop("profile", None)
        return overrides

    async def _submit(self, cid: str, text: str, record: Optional[dict], settings: DirectTextSettings,
                      output_mode: str, manual: Optional[ManualGlossarySource], user_index: Optional[int],
                      once: Optional[dict] = None, anchor: Optional[int] = None,
                      config_extra: Optional[dict] = None) -> Any:
        overrides = self.run_overrides(cid)
        if once:
            overrides = {**overrides, **{k: v for k, v in once.items() if v}}
        expected = len(self.env.chats.messages(cid)) if user_index is None else None
        kwargs: dict = {"config_extra": dict(config_extra)} if config_extra else {}
        try:
            run = await self.env.runs.send(
                cid, text=text, attachment=record, settings=settings, output_mode=output_mode,
                overrides=overrides, manual_glossary=manual, user_index=user_index, **kwargs,
            )
        except Exception as exc:
            log.warning("chat send failed: %s", exc)
            self.notify(f"Could not start: {exc}")
            self._link_version(cid, anchor, expected)
            self._on_run_changed(cid)
            return None
        self._link_version(cid, anchor, getattr(run, "user_index", expected))
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

    def _plan_glossary_label(self) -> str:
        return effective_glossary_label(self.settings().glossary_override_mode,
                                        self.env.config_get if self.env is not None else (lambda k, d=None: d))

    @staticmethod
    def _plan_signature(plan: dict) -> tuple:
        """What the Plan card is built from: Run options / range edits update the card in place (they
        are saved with the plan without rebuilding it), a destination change rebuilds it."""
        return (plan.get("user_index"), plan.get("output_mode"), plan.get("destination") or "chat",
                plan.get("created"), str(plan.get("text") or ""), str(plan.get("library_bid") or ""))

    def _glossary_chip_text(self) -> str:
        """The Plan / Batch glossary chip: the effective mode and, when the main settings drive the run,
        the loaded manual glossary's file name (desktop loaded-glossary label)."""
        effective = self._plan_glossary_label()
        manual = self.env.config_get("manual_glossary_path", "") if self.env is not None else ""
        return glossary_chip_label(effective, manual if effective.endswith("(auto)") else "")

    def _update_plan(self, cid: str, created: Any, **changes: Any) -> None:
        """Save fields of the pending plan (the card is not rebuilt for them)."""
        if not self.bound:
            return
        plan = self.env.chats.meta(cid).get("pending_plan")
        if not isinstance(plan, dict) or plan.get("created") != created:
            return
        plan = dict(plan)
        plan.update(changes)
        self.env.chats.set_meta(cid, "pending_plan", plan)

    def _run_panel(self, plan: dict, *, batch: bool = False) -> RunOptionsPanel:
        cid = self.cid
        created = plan.get("created")
        key = (cid, "batch" if batch else "plan", created)
        panel = self.run_panels.get(key)
        if panel is None:
            values = plan.get("run_config") if isinstance(plan.get("run_config"), dict) else {}
            only = plan.get("only_this_run")
            if batch:
                save = lambda values, only_run: self._update_batch(cid, created, run_config=values,  # noqa: E731
                                                                   only_this_run=only_run)
            else:
                save = lambda values, only_run: self._update_plan(cid, created, run_config=values,  # noqa: E731
                                                                  only_this_run=only_run)

            def on_change(values: dict, only_run: bool) -> None:
                save(values, only_run)
                self._refresh_range_chips()

            panel = RunOptionsPanel(self.env, values=values, only_this_run=True if only is None else bool(only),
                                    on_change=on_change, key="batch-run-options" if batch else "plan-run-options")
            for stale in [k for k in self.run_panels if k[0] == cid and k[1] == key[1] and k != key]:
                self.run_panels.pop(stale, None)
            self.run_panels[key] = panel
        return panel

    def _refresh_range_chips(self) -> None:
        for chip, panel in list(getattr(self, "_range_chips", {}).values()):
            try:
                chip.label = ft.Text(panel.range_chip_label())
                chip.update()
            except Exception:
                pass

    def _plan_controls(self, plan: dict, card: Any = None) -> list:
        from glossarion_mobile.ui.chat.output_modes import output_mode as _output_mode

        context = self.state.chat_context.value
        settings = self.settings()
        record = plan.get("attachment") if isinstance(plan.get("attachment"), dict) else {}
        path = str(record.get("path") or "")
        panel = self._run_panel(plan)
        mode = str(plan.get("output_mode") or "text")
        destination = str(plan.get("destination") or "chat")
        range_chip = ft.Chip(label=ft.Text(panel.range_chip_label()), leading=ft.Icon(ft.Icons.FORMAT_LIST_NUMBERED, size=16),
                             on_click=lambda e: self.open_choose_chapters(plan), key="plan-range")
        self._range_chips = {"plan": (range_chip, panel)}
        chips = [
            ft.Chip(label=ft.Text(context.model), on_click=lambda e: self.open_model_sheet("model"), key="plan-model"),
            ft.Chip(label=ft.Text(context.profile), on_click=lambda e: self.open_model_sheet("profile"),
                    key="plan-profile"),
            ft.Chip(label=ft.Text(f"→ {context.target_language}"), on_click=lambda e: self.open_model_sheet("language"),
                    key="plan-language"),
            ft.Chip(label=ft.Text(self._glossary_chip_text()), leading=ft.Icon(ft.Icons.MENU_BOOK, size=16),
                    on_click=lambda e: self.open_plan_glossary([path]), key="plan-glossary"),
        ]
        if settings.glossary_override_mode in ("manual", "no_glossary"):
            # this chat's own glossary policy (Chat settings) overrides the main glossary
            chips.append(ft.Chip(label=ft.Text("Policy: " + ("Manual" if settings.glossary_override_mode == "manual"
                                                              else "Off")),
                                 on_click=lambda e: self.open_chat_settings(), key="plan-policy"))
        chips += [
            ft.Chip(label=ft.Text(f"{_output_mode(mode).emoji} {_output_mode(mode).label}"),
                    on_click=lambda e: self.open_mode_options(mode), key="plan-output"),
            range_chip,
            ft.Chip(label=ft.Text(destination_label(destination)),
                    leading=ft.Icon(ft.Icons.LOCAL_LIBRARY if destination == "library" else ft.Icons.CHAT, size=16),
                    on_click=lambda e: self.open_destination(plan), key="plan-destination"),
        ]
        facts = ft.Text("estimating…", theme_style=ft.TextThemeStyle.LABEL_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                        key="plan-facts")
        conversion = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.TERTIARY, visible=False,
                             key="plan-conversion")
        if path and card is not None:
            def apply_facts(value: Any, t=facts, c=conversion) -> None:
                info = value if isinstance(value, dict) else {}
                t.value = str(info.get("line") or "")
                c.value = str(info.get("conversion") or "")
                c.visible = bool(c.value)

            self._io_extra(("facts", self.cid, path), lambda p=path: plan_facts(p), apply_facts, card)
        loaded = self.env.config_get("manual_glossary_path", "") if self.env is not None else ""
        buttons: list = [ft.TextButton(content="Choose chapters", icon=ft.Icons.FORMAT_LIST_NUMBERED,
                                       on_click=lambda e: self.open_choose_chapters(plan), key="plan-choose")]
        if loaded and self.glossary_table_opener is not None:
            buttons.append(ft.TextButton(content="Review glossary", icon=ft.Icons.EDIT_NOTE, key="plan-review",
                                         on_click=lambda e, g=str(loaded): self.glossary_table_opener(g)))
        bid = str(plan.get("library_bid") or "")
        library_line_text = ft.Text("📚 Library", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.PRIMARY,
                                    visible=bool(bid), key="plan-library")
        if bid:
            # the attached Library book: where it stands (its card's pill) and its Book page
            buttons.append(ft.TextButton(content="Open book", icon=ft.Icons.LOCAL_LIBRARY, key="plan-open-book",
                                         on_click=lambda e, b=bid: self.navigate("library.book", {"bid": b})))
            if card is not None:
                def apply_line(value: Any, t=library_line_text) -> None:
                    t.value = str(value or "📚 Library")

                service = self._library_service()
                self._io_extra(("libplan", self.cid, bid), lambda b=bid, s=service: library_line(s, b), apply_line,
                               card)
        controls: list = [
            library_line_text,
            facts,
            conversion,
            ft.Row(chips, wrap=True, spacing=6, run_spacing=6),
            panel.control,
            ft.Row(buttons, wrap=True, spacing=4),
        ]
        if self.env is not None and self.env.is_ios:
            # UI_SPEC §7.6 / §2.12.1: where a job starts on iOS, say that it may pause in the background
            controls.append(ft.Text(IOS_BACKGROUND_NOTICE, theme_style=ft.TextThemeStyle.BODY_SMALL,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        return controls

    def _plan_start_menu(self, plan: dict) -> list:
        """Start ▾ (UI_SPEC §2.12.1 "Async variant"): Run as async batch (50% off)."""
        record = plan.get("attachment") if isinstance(plan.get("attachment"), dict) else {}
        if not str(record.get("path") or ""):
            return []
        return [("start_async", "Run as async batch (50% off)")]

    def open_plan_glossary(self, inputs: list) -> Any:
        """The glossary chip: the PlanGlossarySheet (desktop main-window glossary row: 📄 Load Glossary,
        the loaded-glossary label and ✕, Map glossaries for several EPUBs); without the Glossary
        feature, the chat's glossary policy in Chat settings."""
        opener = self.plan_glossary_opener
        if opener is None:
            return self.open_chat_settings()
        return opener(effective=self._plan_glossary_label(), inputs=[p for p in inputs if p],
                      on_changed=self.apply_settings_changed)

    def open_choose_chapters(self, plan: dict, *, batch: bool = False) -> Optional[ChooseChaptersSheet]:
        """Choose chapters: range + Spine order + the live preview (desktop 🔍 preview)."""
        if not self.bound:
            return None
        panel = self._run_panel(plan, batch=batch)
        if batch:
            files = list(plan.get("files") or [])
            path = next((p for p in files if str(p).lower().endswith((".epub", ".pdf"))), files[0] if files else "")
        else:
            record = plan.get("attachment") if isinstance(plan.get("attachment"), dict) else {}
            path = str(record.get("path") or "")
        store = panel.store

        def preview(text: str, spine: bool) -> dict:
            snapshot = store.snapshot() if store is not None and hasattr(store, "snapshot") else {}
            return range_preview(snapshot, path, text, spine)

        def applied() -> None:
            panel.refresh_summary()
            self._refresh_range_chips()

        self.chapters_sheet = ChooseChaptersSheet(store=store, path=path, preview=preview, io=self.env.run_io,
                                                  on_applied=applied,
                                                  spine_available=not str(path).lower().endswith(".pdf"))
        self.chapters_sheet.show(self.page)
        return self.chapters_sheet

    def open_destination(self, plan: dict) -> Any:
        """Save to: This chat / Library (the normal pipeline)."""
        cid = self.cid
        created = plan.get("created")
        library_reason = None
        jobs = getattr(self.env, "jobs", None) if self.env is not None else None
        if jobs is None or not getattr(jobs, "has_kind", lambda _k: True)("translate"):
            library_reason = "The job service is not running"

        def choose(destination: str) -> None:
            self._update_plan(cid, created, destination=destination)
            self.render_transcript()

        sheet = destination_sheet(plan.get("destination"), choose, tablet=bool(self.layout.persistent_sidebar),
                                  library_reason=library_reason)
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def start_plan(self) -> None:
        plan = self._pending_plan()
        if plan is None:
            return
        env = self.env
        cid = self.cid
        record = plan.get("attachment") or None
        if str(plan.get("destination") or "chat") == "library" and record:
            # the output-root question comes first; Cancel there keeps this plan
            self._spawn(self._start_library_plan(plan))
            return
        env.chats.set_meta(cid, "pending_plan", None)
        manual_data = plan.get("manual_glossary")
        manual = None
        if isinstance(manual_data, dict):
            manual = ManualGlossarySource(
                kind=str(manual_data.get("kind") or "content"), path=str(manual_data.get("path") or ""),
                content=str(manual_data.get("content") or ""), extension=str(manual_data.get("extension") or ".txt"),
            )
        run_config = self._take_run_config(cid, plan)
        self.sent.append((cid, plan.get("text") or "", record))
        once = plan.get("once") if isinstance(plan.get("once"), dict) else None
        self._spawn(self._submit(cid, str(plan.get("text") or ""), record, self.settings(cid),
                                 str(plan.get("output_mode") or "text"), manual, int(plan.get("user_index") or 0),
                                 once, config_extra=run_config))

    def _take_run_config(self, cid: str, plan: dict) -> dict:
        """The Plan card's Run options for this start (its panel, else the values saved with the plan);
        the panel is released."""
        panel = self.run_panels.pop((cid, "plan", plan.get("created")), None)
        if panel is not None:
            return panel.config_overrides()
        values = plan.get("run_config") if isinstance(plan.get("run_config"), dict) else {}
        return dict(values) if plan.get("only_this_run", True) else {}

    async def _start_library_plan(self, plan: dict) -> Any:
        """Start with "Save to: Library": the desktop Load-for-translation output-root check first
        (``translate_sheet.confirm_output_root``: the book's workspace lies under another output folder);
        Cancel keeps the plan (a send without a Plan card shows one now), else the Library job starts."""
        cid = self.cid
        created = plan.get("created")
        if created in self._library_starting:
            return None  # a second Start tap while the first one asks
        self._library_starting.add(created)
        try:
            if not await self._confirm_output_root(plan):
                if self.bound and not isinstance(self.env.chats.meta(cid).get("pending_plan"), dict):
                    self.env.chats.set_meta(cid, "pending_plan", plan)
                    if self.cid == cid:
                        self.render_transcript(follow=True)
                return None
            if self.bound:
                self.env.chats.set_meta(cid, "pending_plan", None)
            record = dict(plan.get("attachment") or {})
            run_config = self._take_run_config(cid, plan)
            self.sent.append((cid, plan.get("text") or "", record))
            return await self._submit_library(cid, plan, record, run_config)
        finally:
            self._library_starting.discard(created)

    def _library_service(self) -> Any:
        """The Library's LibraryService (ChatEnv.library_service, else the Tools context's), or None."""
        env = self.env
        if env is None:
            return None
        getter = getattr(env, "library_service", None)
        try:
            service = getter() if callable(getter) else None
        except Exception:
            service = None
        if service is None:
            factory = getattr(env, "tools_context", None)
            ctx = factory() if callable(factory) else None
            service = getattr(ctx, "service", None) if ctx is not None else None
        return service

    async def _confirm_output_root(self, plan: dict) -> bool:
        """The Library book's workspace is under another output folder than the current one: ask like
        the desktop (``translate_sheet.confirm_output_root``); True when nothing needs asking."""
        bid = str(plan.get("library_bid") or "")
        factory = getattr(self.env, "tools_context", None) if self.env is not None else None
        ctx = factory() if callable(factory) and bid else None
        service = getattr(ctx, "service", None) if ctx is not None else None
        if service is None:
            return True
        book = service.book_for_bid(bid)
        if not book:
            return True
        from glossarion_mobile.ui.library.translate_sheet import confirm_output_root

        return bool(await confirm_output_root(ctx, [book]))

    async def _submit_library(self, cid: str, plan: dict, record: dict, run_config: dict) -> Any:
        """Save to: Library - a ``translate`` job over the attachment (the normal pipeline: workspace in
        the output root, Library shelf, post-QA scan per settings); the turn gets a "Library job" card.
        A Library book attached in the chat (``library_bid``) is tied to its book (origin / card bid)."""
        from glossarion_mobile.ui.chat.transcript_model import LIBRARY_LABEL

        from glossarion_mobile.state.setting_writes import output_mode_values
        from glossarion_mobile.ui.chat.run_request import config_overrides as chat_config_overrides

        path = str(record.get("path") or "")
        name = str(record.get("name") or os.path.basename(path))
        title = str((self.env.chats.session(cid) or {}).get("title") or "Chat")
        # what the Plan card shows: the chat's model / profile / language chips and its mode chip
        # (the desktop Output Mode selector's flags, settings_rules.output_mode_flags), then Run options
        overrides: dict = {}
        try:
            overrides.update(chat_config_overrides(self.run_overrides(cid)))
        except Exception:
            log.debug("chat overrides unavailable", exc_info=True)
        overrides.update(output_mode_values(plan.get("output_mode")))
        overrides.update(run_config or {})
        params: dict = {"config_overrides": overrides} if overrides else {}
        bid = str(plan.get("library_bid") or "")
        origin = {"type": "chat", "cid": cid, "label": f"Chat · {title}"}
        if bid:
            origin["bid"] = bid  # the Book page's last-run line matches on it
        try:
            job_id = await self.env.jobs.submit("translate", name, (path,), params, origin)
        except Exception as exc:
            self.notify(f"Could not start: {exc}")
            self.env.chats.set_meta(cid, "pending_plan", plan)  # the plan stays, Start can be tapped again
            self.render_transcript()
            return None
        if bid:
            body = (f"📚 **Translating in the Library:** {name}\n\nThe book continues in its own workspace "
                    "(its Library card and Book page follow it). Follow it in Jobs.")
        else:
            body = (f"📚 **Translating in the Library:** {name}\n\nThe workspace goes to the output folder and the "
                    "book appears on the Library shelf (In progress). Follow it in Jobs.")
        storage = {"library_job": str(job_id), "source": path}
        if bid:
            storage["bid"] = bid
        append_tool_message(self.env.chats, cid, body, "", LIBRARY_LABEL, storage)
        self.render_transcript(follow=True)  # the turn's own card (its Plan card's slot) shows where it went
        self.notify(f"Translating · {name}", action_label="Jobs", on_action=lambda: self.navigate("jobs"))
        return job_id

    def _library_job_control(self, item: Any, messages: list, record: Optional[dict]) -> Any:
        """A turn sent to the Library (Plan "Save to: Library"): where the run went, with Jobs / Library."""
        message = messages[item.library] if 0 <= item.library < len(messages) else None
        storage = self._tool_storage(message)
        job_id = str(storage.get("library_job") or "")
        bid = str(storage.get("bid") or "")
        record = self._follow_library_rename(item.library, message, storage, job_id, record)

        def build() -> JobCard:
            return self._tool_job_card(record, "Sent to the Library · translating as a book", [
                ("Job", "WORK_HISTORY", "libjob-job", lambda: self._on_library_action("job", job_id), True),
                ("Library", "LOCAL_LIBRARY", "libjob-library",
                 lambda: self._on_library_action("library", job_id, bid), False),
            ])

        return self._card(f"job-{item.index}", ("library", self.cid, item.index, job_id, bid, record), build)

    def _tool_storage(self, message: Any) -> dict:
        """A tool-job card message's data: its sidecar entry (``append_tool_message``) under what the
        message still carries in memory."""
        storage = message[6] if message is not None and len(message) > 6 and isinstance(message[6], dict) else {}
        created = str(storage.get("created_at") or "")
        saved = None
        if created and self.bound:
            jobs = self.env.chats.meta(self.cid).get(TOOL_JOBS_META)
            saved = jobs.get(created) if isinstance(jobs, dict) else None
        return {**(saved if isinstance(saved, dict) else {}), **storage}

    @staticmethod
    def _tool_job_card(record: Optional[dict], status: str, buttons: list, *, icon: Optional[str] = None,
                       meta: Optional[str] = None) -> JobCard:
        """The static card of a job this chat handed to a tool (a "Save to: Library" translation, a QA
        scan): where it went and its buttons ``[(label, icon name, key, handler, primary)]``; its live
        progress, Stop and "Done · Open" are the chat's JobStrip (``chat_card`` False)."""
        card = JobCard(attachment=record, phase=CardPhase("done"), icon=icon, meta=meta)
        card.set_phase(CardPhase("done"), status=status)
        controls: list = []
        for label, icon_name, key, handler, primary in buttons:
            kind = ft.FilledTonalButton if primary else ft.TextButton
            controls.append(kind(content=label, icon=getattr(ft.Icons, icon_name), key=key,
                                 on_click=lambda e, h=handler: h()))
        card.buttons.controls = controls
        return card

    def _follow_library_rename(self, index: int, message: Any, storage: dict, job_id: str,
                               record: Optional[dict]) -> Optional[dict]:
        """A Library job renamed its input for a workspace collision (``renamed_inputs``, the desktop
        selection rule): the turn's stored source and the chat's attachment record follow the new name."""
        if not job_id or self.env is None or self.env.jobs is None or not self.bound:
            return record
        try:
            snap = self.env.jobs.snapshot_of(job_id)
        except Exception:
            snap = None
        renamed = dict((getattr(snap, "result", None) or {}).get("renamed_inputs") or {}) if snap is not None else {}
        source = str(storage.get("source") or "")
        new = renamed.get(source)
        if not new or new == source:
            return record
        try:
            updated = dict(storage, source=new)
            self.env.chats.replace_message(self.cid, index, tuple(message[:6]) + (updated,) + tuple(message[7:]))
            created = str(storage.get("created_at") or "")
            saved = self.env.chats.meta(self.cid).get(TOOL_JOBS_META)
            if created and isinstance(saved, dict) and isinstance(saved.get(created), dict):
                saved = dict(saved)
                saved[created] = dict(saved[created], source=new)  # what survives the history's save
                self.env.chats.set_meta(self.cid, TOOL_JOBS_META, saved)
            current = self.env.chats.attachment(self.cid)
            if isinstance(current, dict) and str(current.get("path") or "") == source:
                self.env.chats.set_attachment(self.cid, dict(current, path=new, name=os.path.basename(new)))
            attached = self.env.chats.meta(self.cid).get(LIBRARY_ATTACHMENT_META)
            if isinstance(attached, dict) and _same_path(attached.get("path"), source):
                # the chat's Library book follows its renamed raw file (the next Send still continues it)
                self.env.chats.set_meta(self.cid, LIBRARY_ATTACHMENT_META, dict(attached, path=new))
        except Exception:
            log.debug("following the renamed Library input failed", exc_info=True)
            return record
        if isinstance(record, dict) and str(record.get("path") or "") == source:
            record = dict(record, path=new, name=os.path.basename(new))
        return record

    def _on_library_action(self, action: str, job_id: str = "", bid: str = "") -> None:
        if action == "job" and job_id:
            self.navigate("jobs.detail", {"jid": job_id})
        elif action == "job":
            self.navigate("jobs")
        elif bid:
            self.navigate("library.book", {"bid": bid})  # the attached Library book's own page
        else:
            self.navigate("library")

    # ---- QA scan from the chat (owner request 2026-10-08) -------------------------------------------

    async def start_chat_qa(self, item: Any = None) -> Optional[str]:
        """QA scan of a turn's workspace (Result card › QA scan) or of the chat's latest book workspace
        (＋ › QA scan, ``/qa``; Tools › QA Scanner when the chat has none): the shared scan through a
        ``qa_scan`` job (``qa_model.chat_qa_job``: Quick Scan, the mobile duplicate-check sample size).
        Progress, Stop and "Done · Open" are the chat's JobStrip; the chat gets a "QA scan" card."""
        if not self.bound or self.env.jobs is None:
            self.navigate("tools.qa")
            return None
        if item is not None:
            folder = await self.env.run_io(self._job_workspace, item)
            source = self._turn_source(item)
        else:
            folder, source = await self.env.run_io(self._latest_workspace)
            if not folder:
                self.navigate("tools.qa")  # no book in this chat yet: pick one in Tools › QA Scanner
                return None
        if not folder or not await self.env.run_io(qa_scannable, folder, source):
            self.notify(NOTHING_TO_SCAN)
            return None
        from glossarion_mobile.ui.tools import qa_model

        # the one-time mobile move of a saved desktop 1000 to 0 (Tools › QA Scanner runs it too): a chat
        # scan uses the same effective duplicate-check sample size the QA screen shows
        store, prefs = getattr(self.env, "store", None), getattr(self.env, "prefs", None)
        if store is not None and prefs is not None and callable(getattr(store, "set", None)):
            qa_model.migrate_quick_sample_size(self.env.config_get, store.set, prefs)
        cid = self.cid
        chat_title = str((self.env.chats.session(cid) or {}).get("title") or "Chat")
        job = qa_model.chat_qa_job(folder, source, cid=cid, chat_title=chat_title)
        if isinstance(job, str):  # why this workspace cannot be scanned here
            self.notify(job)
            return None
        try:
            job_id = await self.env.jobs.submit(*job)
        except Exception as exc:
            self.notify(f"Could not start the QA scan: {exc}")
            return None
        name = os.path.basename(os.path.normpath(folder))
        summary = self._qa_summary_line()
        body = (f"🔎 **QA scan:** {name}\n\n{summary}. The bar above shows its progress; "
                "open the report or the chapters here when it is done.")
        # message[4] is the scanned workspace: a later move into the Library rewrites it with the cards
        append_tool_message(self.env.chats, cid, body, folder, QA_LABEL,
                            {"qa_job": str(job_id), "folder": folder, "source": source, "summary": summary})
        self.show_newest()
        self.notify(f"QA scan · {name}", action_label="Jobs",
                    on_action=lambda j=str(job_id): self.navigate("jobs.detail", {"jid": j}))
        return str(job_id)

    def _qa_summary_line(self) -> str:
        """``qa_model.chat_qa_summary_line``: the mode and the duplicate-check sample size a chat scan
        runs with (the saved Tools › QA / Settings › QA value, else the mobile default 0)."""
        from glossarion_mobile.ui.tools import qa_model

        saved = self.env.config_get("qa_scanner_settings", None) if self.env is not None else None
        return str(qa_model.chat_qa_summary_line({"qa_scanner_settings": saved} if isinstance(saved, Mapping) else {}))

    def _qa_job_control(self, item: Any, messages: list) -> Any:
        """A "QA scan" message: the scan's state and Job · Report · Chapters (the tool-job card)."""
        message = messages[item.index] if 0 <= item.index < len(messages) else None
        storage = self._tool_storage(message)
        job_id = str(storage.get("qa_job") or "")
        folder = str((message[4] if message is not None and len(message) > 4 else "") or storage.get("folder") or "")
        source = str(storage.get("source") or "")
        snap = self.env.jobs.snapshot_of(job_id) if (job_id and self.env.jobs is not None) else None
        state = state_name(snap) if snap is not None else ""
        status = {"QUEUED": "Queued · starts after the current job", "STARTING": "Scanning…",
                  "RUNNING": "Scanning…", "STOPPING": "Stopping…", "FORCE_STOPPING": "Stopping…",
                  "DONE": "Done", "FAILED": "Failed", "STOPPED": "Stopped", "CANCELLED": "Cancelled",
                  "INTERRUPTED": "Interrupted"}.get(state, "Sent to Jobs")
        name = os.path.basename(os.path.normpath(folder)) if folder else "workspace"
        record = {"name": f"QA scan · {name}", "path": folder}
        summary = str(storage.get("summary") or "") or self._qa_summary_line()  # what the scan ran with

        def build() -> JobCard:
            return self._tool_job_card(record, status, [
                ("Job", "WORK_HISTORY", "qajob-job",
                 lambda: self.navigate("jobs.detail", {"jid": job_id}) if job_id else self.navigate("jobs"), True),
                ("Report", "ASSESSMENT", "qajob-report", lambda: self._spawn(self.open_qa_report(folder, job_id)), False),
                ("Chapters", "CHECKLIST", "qajob-chapters",
                 lambda: self._spawn(self._open_qa_chapters(folder, source)), False),
            ], icon="FACT_CHECK", meta=summary)

        return self._card(self._mid(item.index) or f"m-{item.index}",  # the Jump-to / search scroll key
                          ("qa", self.cid, item.index, job_id, folder, status, summary), build)

    async def open_qa_report(self, folder: str, job_id: str = "") -> Optional[str]:
        """QA card › Report: the scan's ``validation_results.html`` in the QA report viewer."""
        from glossarion_mobile.ui.tools import qa_model

        path = qa_model.report_path_for(folder) if folder else ""
        if not path or not await self.env.run_io(os.path.isfile, path):
            snap = self.env.jobs.snapshot_of(job_id) if (job_id and self.env.jobs is not None) else None
            ended = state_name(snap) in _TERMINAL_JOB_STATES if snap is not None else False
            self.notify("This scan wrote no report" if ended else "The scan has not finished yet")
            return None
        factory = getattr(self.env, "tools_context", None)
        ctx = factory() if callable(factory) else None
        if ctx is None:
            self.notify("Reports cannot be opened in this session")
            return None
        from glossarion_mobile.ui.tools.qa_screen import open_qa_report

        return open_qa_report(ctx, path)

    async def _open_qa_chapters(self, folder: str, source: str) -> Optional[str]:
        """QA card › Chapters: the scanned workspace in the Progress manager (Retranslate the QA-failed)."""
        opener = getattr(self.env, "open_progress", None) if self.env is not None else None
        if not folder or opener is None:
            self.notify("No output folder for this chat yet")
            return None
        return await self._await(opener(folder, source))

    def start_plan_async(self) -> None:
        """Start ▾ › Run as async batch (50% off): Tools › Async batch with this file as the source (its
        screen asks the async dialog's questions); the plan stays until the batch is submitted there."""
        plan = self._pending_plan()
        record = plan.get("attachment") if isinstance(plan, dict) and isinstance(plan.get("attachment"), dict) else {}
        path = str(record.get("path") or "")
        opener = getattr(self.env, "open_tool_with_source", None) if self.env is not None else None
        if not path or opener is None:
            self.notify("Async batch is not available in this session")
            return
        opener("tools.async", path)

    def _open_progress(self, item: Any = None) -> Any:
        """Job card › Progress: the turn's workspace in the Progress manager (Chapters)."""
        return self._spawn(self._open_progress_async(item))

    async def _open_progress_async(self, item: Any = None) -> Optional[str]:
        opener = getattr(self.env, "open_progress", None) if self.env is not None else None
        folder, source = await self.env.run_io(self._reader_target, item) if self.env is not None else ("", "")
        if not folder or opener is None:
            self.notify("No output folder for this chat yet")
            return None
        return await self._await(opener(folder, source))

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
        self.show_newest()
        self.refresh_send()

    def _on_job_action(self, action: str, item: Any = None) -> None:
        if action == "start":
            self.start_plan()
        elif action == "start_async":
            self.start_plan_async()
        elif action == "progress":
            self._open_progress(item)
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
            self._spawn(self._export_outputs(item))
        elif action == "compile":
            self._compile_menu(item)
        elif action == "library":
            self._spawn(self.open_in_library(item))
        elif action == "qa":
            self._spawn(self.start_chat_qa(item))
        elif action in ("read", "open_reader"):
            self._spawn(self._open_reader(item))
        elif action == "open_output":
            self._spawn(self._open_job_output(item))
        elif action in ("resume", "retry"):
            if item is None:
                self._resume_last()
            else:
                self._spawn(self._resume_turn(item))
        else:
            self.notify("This action is not available here")

    async def open_in_library(self, item: Any = None) -> Optional[str]:
        """Result card › Open in Library: the Book page of the Library book this turn's workspace became
        (auto-migrate, ChatEnv.library_book); the Library home when the book cannot be told."""
        folder = await self.env.run_io(self._job_workspace, item) if self.env is not None else ""
        finder = getattr(self.env, "library_book", None) if self.env is not None else None
        bid = None
        if folder and finder is not None:
            try:
                bid = await self._await(finder(folder))
            except Exception:
                log.debug("looking up the Library book failed", exc_info=True)
                bid = None
        if bid:
            self.navigate("library.book", {"bid": str(bid)})
            return str(bid)
        self.navigate("library")
        return None

    async def _resume_turn(self, item: Any) -> Any:
        """Resume / Retry failed of a Result card: a workspace that moved into the Library continues in
        the Library's translate sheet (``ChatEnv.library_translate``: its own progress file); the chat's
        run root otherwise (``ChatRuns.resubmit``)."""
        folder, in_attachments = await self.env.run_io(self._job_workspace_state, item)
        hand_off = getattr(self.env, "library_translate", None)
        if folder and not in_attachments and hand_off is not None:
            return await self._await(hand_off(folder, self._turn_source(item)))
        self._resume_last()
        return None

    async def _open_reader(self, item: Any = None) -> None:
        """Job card Read / Open reader (U5): the turn's workspace in the Reader."""
        opener = self.env.open_reader if self.env is not None else None
        if opener is None:
            self.notify("The Reader is not available in this session")
            return
        folder, source = await self.env.run_io(self._reader_target, item)
        result = opener(folder, source)
        if hasattr(result, "__await__"):
            await result

    def _run_turn(self, run: Any) -> Optional[int]:
        """The current index of ``run``'s user turn (Delete message may have moved it)."""
        if run is None:
            return None
        fn = getattr(self.env.runs, "turn_index", None) if self.env is not None else None
        return fn(self.cid, run) if callable(fn) else getattr(run, "user_index", None)

    def _turn_source(self, item: Any = None) -> str:
        """The attachment path of the job card's turn (default: the turn of the chat's run)."""
        runs = self.env.runs if self.bound else None
        index = getattr(item, "index", None)
        if index is None and runs is not None:
            index = self._run_turn(runs.run_for(self.cid))
        messages = self._messages()
        if index is not None and 0 <= int(index) < len(messages):
            message = messages[int(index)]
            if message and len(message) > 2 and str(message[0]) == "user_file":
                return str(message[2] or "")
        return ""

    def _reader_target(self, item: Any = None) -> tuple:
        """Blocking: (workspace folder, attachment path) of the job card's turn (default: the chat's run).

        The turn's own workspace comes first (``_job_workspace``: its ``Attachments/<stem>`` folder, or the
        Library folder it moved to); the run's pipeline folder only while it exists: a finished run's
        temp root is deleted, which left Progress / the QA-failed chip on a missing folder (U9)."""
        source = self._turn_source(item)
        folder = self._job_workspace(item) if item is not None else ""
        if not folder:
            runs = self.env.runs if self.bound else None
            run = runs.run_for(self.cid) if runs is not None else None
            index = getattr(item, "index", None)
            if run is not None and (index is None or self._run_turn(run) == index):
                folder = next((str(c) for c in (run.output_dir, run.output_folder) if c and os.path.isdir(str(c))), "")
        return folder or self._last_output_folder(), source

    def _last_output_folder(self) -> str:
        run = self.env.runs.run_for(self.cid) if self.env.runs is not None else None
        if run is not None and run.output_folder:
            return run.output_folder
        for message in reversed(self._messages()):
            if message and message[0] == "assistant" and len(message) > 4 and str(message[4] or ""):
                return str(message[4])
        return self.env.chats.output_folder(self.cid)

    async def _export_outputs(self, item: Any = None) -> None:
        """Share / Export the job card's output files (its own ``Attachments/<stem>`` workspace,
        ``chat_ops.workspace_outputs``: compiled EPUB / PDF first, then ``*_translated.txt``,
        subtitles, SDLXLIFF and the glossary); a choice when there are several. Without a turn
        workspace: the chat's newest output folder (the desktop attachment rule, newest EPUB / PDF)."""
        folder = await self.env.run_io(self._job_workspace, item) if item is not None else ""
        if not folder:
            folder = self._last_output_folder()
        documents: list = []
        if folder and os.path.isdir(folder):
            documents = [path for path, _kind in await self.env.run_io(workspace_outputs, folder)]
        if not documents:
            self.notify("No output files in this turn's workspace yet")
            return
        if len(documents) > 1 and self.page is not None:
            items = [ActionItem(f"{os.path.splitext(path)[1][1:].upper()} · {os.path.basename(path)}",
                                lambda p=path: self._spawn(self._export_document(p)), icon="IOS_SHARE",
                                key=f"export-output-{index}")
                     for index, path in enumerate(documents)]
            sheet = ActionSheet(items, title="Share / Export", tablet=bool(self.layout.persistent_sidebar))
            self.last_sheet = sheet
            sheet.show(self.page)
            return
        await self._export_document(documents[0])

    def _outputs_for(self, item: Any) -> list:
        """Blocking: ``[(path, kind)]`` of a job card's turn workspace (``workspace_outputs``)."""
        folder = self._job_workspace(item)
        return workspace_outputs(folder) if folder else []

    async def _open_job_output(self, item: Any = None) -> None:
        """Job card › Open output: the turn's ``Attachments/<stem>`` workspace in Files (else the chat's
        newest output folder)."""
        folder = await self.env.run_io(self._job_workspace, item) if item is not None else ""
        if not folder:
            folder = self._last_output_folder()
        if folder and self.env.open_output is not None:
            self.env.open_output(folder)
        else:
            self.notify("No output folder for this chat yet")

    def _open_output_file(self, path: str, kind: str, item: Any = None) -> Optional[ActionSheet]:
        """An output chip: Open (the Reader for the turn's EPUB, the text editor for text, subtitle,
        SDLXLIFF and glossary files) · Share / Save… (ExportSheet)."""
        from glossarion_mobile.ui.tools import text_editor

        items: list = []
        if kind == "epub":
            items.append(ActionItem("Read", lambda: self._spawn(self._open_reader(item)), icon="AUTO_STORIES",
                                    key="output-read"))
        elif kind in ("txt", "subtitle", "sdlxliff", "glossary", "html"):
            reason = None if text_editor.is_text_file(path) else "Not a text file"
            # the editor route takes a query; the chat's navigate(route, params) has none
            editor_ctx = types.SimpleNamespace(prefs=getattr(self.env, "prefs", None),
                                               go=lambda name, params=None, query=None: self.navigate(name, params))
            items.append(ActionItem("Open in text editor", lambda: text_editor.open_text_editor(editor_ctx, path),
                                    icon="EDIT_NOTE", key="output-edit", disabled_reason=reason))
        items.append(ActionItem("Share / Save…", lambda: self._spawn(self._export_document(path)), icon="IOS_SHARE",
                                key="output-share"))
        if self.page is None:
            return None
        sheet = ActionSheet(items, title=os.path.basename(path), tablet=bool(self.layout.persistent_sidebar))
        self.last_sheet = sheet
        sheet.show(self.page)
        return sheet

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

    def _compile_menu(self, item: Any = None) -> Optional[ActionSheet]:
        """Job card › Compile ▾: EPUB or PDF of the card's own workspace (UI_SPEC §2.12.4)."""
        if self.page is None:
            self._spawn(self._compile(item))
            return None
        sheet = ActionSheet([
            ActionItem("Compile EPUB", lambda: self._spawn(self._compile(item, "compile_epub")), icon="MENU_BOOK",
                       key="compile-epub"),
            ActionItem("Compile PDF", lambda: self._spawn(self._compile(item, "compile_pdf")), icon="PICTURE_AS_PDF",
                       key="compile-pdf"),
        ], title="Compile", tablet=bool(self.layout.persistent_sidebar))
        self.last_sheet = sheet
        sheet.show(self.page)
        return sheet

    async def _compile(self, item: Any = None, kind: str = "compile_epub") -> None:
        folder = await self.env.run_io(self._job_workspace, item) if item is not None else ""
        try:
            job_id = await self.env.runs.compile(self.cid, kind, folder=folder or None)
        except Exception as exc:
            self.notify(f"Could not start compiling: {exc}")
            return
        if job_id is None:
            self.notify("No translated output folder for this chat yet")
        else:
            label = "PDF" if kind == "compile_pdf" else "EPUB"
            self.notify(f"Compiling {label} · see Jobs", action_label="Jobs", on_action=lambda: self.navigate("jobs"))

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
            self._forget_library_attachment()
        self._set_mode(self.state.output_mode.value.attachment_changed(None))
        self.refresh_send()

    def _forget_library_attachment(self, keep_path: str = "") -> None:
        """The composer no longer holds the chat's Library book (removed, or another file attached)."""
        if not self.bound:
            return
        attached = self.env.chats.meta(self.cid).get(LIBRARY_ATTACHMENT_META)
        if attached is not None and not (keep_path and isinstance(attached, dict)
                                         and _same_path(attached.get("path"), keep_path)):
            self.env.chats.set_meta(self.cid, LIBRARY_ATTACHMENT_META, None)

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
            self._forget_library_attachment(keep_path=str(record.get("path") or path))
        name = record["name"]
        self.caption.show(f"Attached {name} · Vision enabled" if is_vision_attachment(path) else f"Attached {name}")
        self.refresh_send()
        return True

    async def pick_and_attach(self, images: bool = False, extensions: Optional[list] = None) -> Optional[str]:
        if self.env is None or self.env.pick_files is None:
            self.notify("File picking is not available in this build")
            return None
        allowed = list(extensions) if extensions else (_IMAGE_EXTENSIONS if images else _ATTACH_EXTENSIONS)
        multiple = not images and not extensions  # ＋ › Files: several files make a BatchPlanCard (§2.12.5)
        paths = await self.env.pick_files(allowed, multiple)
        if not paths:
            return None
        if self.env.import_file is not None:
            imported = []
            for picked in paths:
                imported.append(await self.env.run_io(self.env.import_file, picked))
            paths = [p for p in imported if p]
        if len(paths) > 1:
            self.set_batch([str(p) for p in paths])
            return None
        path = paths[0] if paths else None
        if path and self.attach_file(path):
            return path
        return None

    async def pick_folder_batch(self) -> Optional[str]:
        """Files long-press › Pick folder… (UI_SPEC §2.5): the folder's supported files as a batch
        (``FileBridge.pick_folder`` copies it into the Inbox); where the platform cannot hand over a
        folder (Android SAF trees: ``FolderPickUnavailable``) the folder comes in as a .zip / .cbz."""
        picker = getattr(self.env, "pick_folder", None) if self.env is not None else None
        if picker is None:
            self.notify("Folder picking is not available here: pick the folder as a .zip / .cbz")
            await self.pick_and_attach(extensions=["zip", "cbz"])
            return None
        try:
            folder = await picker()
        except Exception as exc:
            reason = str(getattr(exc, "reason", "") or exc or "")
            if exc.__class__.__name__ != "FolderPickUnavailable":
                log.warning("folder pick failed: %s", exc)
            if reason == "No folder was chosen":
                return None
            self.notify(f"{reason or 'This device cannot pick folders'} · pick the folder as a .zip / .cbz instead")
            await self.pick_and_attach(extensions=["zip", "cbz"])
            return None
        if not folder:
            return None
        files = await self.env.run_io(batch_files, folder, False)
        if not files:
            self.notify(NO_FILES_TEXT.format(folder=os.path.basename(str(folder))))
            return None
        self.set_batch(files, folder=str(folder))
        self.notify(f"📁 Found {len(files)} supported files in: {os.path.basename(str(folder))}")
        return str(folder)

    # ---- BatchPlanCard (UI_SPEC §2.12.5) ------------------------------------------------------------

    def _pending_batch(self) -> Optional[dict]:
        if not self.bound:
            return None
        batch = self.env.chats.meta(self.cid).get("pending_batch")
        return batch if isinstance(batch, dict) and batch.get("files") else None

    def set_batch(self, files: list, *, folder: str = "", include_subfolders: bool = False) -> Optional[dict]:
        """Put a batch plan in this chat (replaces a previous one)."""
        if not self.bound:
            self.notify("Batches need the chat store")
            return None
        batch = {"files": [str(p) for p in files], "folder": str(folder or ""),
                 "include_subfolders": bool(include_subfolders), "created": time.time()}
        self.env.chats.set_meta(self.cid, "pending_batch", batch)
        self.show_newest()
        return batch

    def _update_batch(self, cid: str, created: Any, **changes: Any) -> None:
        if not self.bound:
            return
        batch = self.env.chats.meta(cid).get("pending_batch")
        if not isinstance(batch, dict) or batch.get("created") != created:
            return
        batch = dict(batch)
        batch.update(changes)
        self.env.chats.set_meta(cid, "pending_batch", batch)

    def _batch_control(self, batch: dict) -> Any:
        from glossarion_mobile.ui.chat.batch_plan import BatchPlanCard

        panel = self._run_panel(batch, batch=True)
        mapping = self.env.config_get("manual_glossary_map", {}) if self.env is not None else {}
        mapped = {}
        if isinstance(mapping, dict):
            for epub, glossary in mapping.items():
                mapped[os.path.normcase(os.path.abspath(str(epub)))] = str(glossary)
        jobs = getattr(self.env, "jobs", None)
        start_reason = None if jobs is not None else "The job service is not running"
        signature = (self.cid, batch.get("created"), tuple(batch.get("files") or ()), bool(batch.get("include_subfolders")),
                     self._glossary_chip_text(), tuple(sorted(mapped.items())), start_reason)
        if self.batch_card is not None and getattr(self.batch_card, "_signature", None) == signature:
            return self.batch_card
        card = BatchPlanCard(files=list(batch.get("files") or ()), folder=str(batch.get("folder") or ""),
                             include_subfolders=bool(batch.get("include_subfolders")),
                             glossary_label=self._glossary_chip_text(), mapped=mapped, run_options=panel,
                             on_action=lambda action, *args, b=batch: self._on_batch_action(b, action, *args),
                             start_reason=start_reason, key=f"batch-plan-{int(float(batch.get('created') or 0) * 1000)}")
        card._signature = signature
        self.batch_card = card
        return card

    def _on_batch_action(self, batch: dict, action: str, *args: Any) -> Any:
        cid = self.cid
        created = batch.get("created")
        files = list(batch.get("files") or ())
        if action == "clear":
            self.env.chats.set_meta(cid, "pending_batch", None)
            self.run_panels.pop((cid, "batch", created), None)
            self.batch_card = None
            self.render_transcript()
            return None
        if action == "subfolders":
            include = bool(args[0]) if args else False

            async def rescan() -> None:
                found = await self.env.run_io(batch_files, str(batch.get("folder") or ""), include)
                self._update_batch(cid, created, include_subfolders=include, files=found or files)
                self.batch_card = None
                self.render_transcript()

            return self._spawn(rescan())
        if action in ("glossary", "file"):
            return self.open_plan_glossary(files)
        if action == "map":
            opener = self.plan_glossary_opener
            feature_map = getattr(getattr(opener, "__self__", None), "open_map_glossaries", None)
            epubs = [p for p in files if str(p).lower().endswith(".epub")]
            if feature_map is None:
                self.notify("Map glossaries needs the Glossary feature")
                return None
            return feature_map(epubs, on_saved=lambda _r: (setattr(self, "batch_card", None), self.render_transcript()))
        if action == "start":
            return self._spawn(self._start_batch(batch))
        return None

    async def _start_batch(self, batch: dict) -> Any:
        """One Start: one ``translate`` job over every file (JobSpec inputs; the desktop multi-file run)."""
        from glossarion_mobile.ui.chat.transcript_model import LIBRARY_LABEL

        cid = self.cid
        created = batch.get("created")
        files = [p for p in (batch.get("files") or ()) if p]
        panel = self.run_panels.get((cid, "batch", created))
        run_config = panel.config_overrides() if panel is not None else (
            dict(batch.get("run_config") or {}) if batch.get("only_this_run", True) else {})
        title = os.path.basename(files[0]) + (f" +{len(files) - 1}" if len(files) > 1 else "") if files else "Batch"
        chat_title = str((self.env.chats.session(cid) or {}).get("title") or "Chat")
        params: dict = {"config_overrides": dict(run_config)} if run_config else {}
        try:
            job_id = await self.env.jobs.submit("translate", title, tuple(files), params,
                                                {"type": "chat", "cid": cid, "label": f"Chat · {chat_title}"})
        except Exception as exc:
            self.notify(f"Could not start: {exc}")
            return None
        self.env.chats.set_meta(cid, "pending_batch", None)
        self.run_panels.pop((cid, "batch", created), None)
        self.batch_card = None
        names = "\n".join(f"- {os.path.basename(p)}" for p in files[:50]) + (
            f"\n- … +{len(files) - 50} more" if len(files) > 50 else "")
        self.env.chats.record_user_turn(cid, ("user", f"📚 Batch translation ({len(files)} files):\n{names}"),
                                        f"Batch · {len(files)} files")
        append_tool_message(self.env.chats, cid, f"📚 **Batch started:** {len(files)} file(s) translate as books in "
                                                 "the output folder (Library shelf). Follow it in Jobs.", "",
                            LIBRARY_LABEL, {"library_job": str(job_id)})
        self.show_newest()
        self.notify(f"Translating · {title}", action_label="Jobs", on_action=lambda: self.navigate("jobs"))
        return job_id

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

    def chat_scale(self, cid: Optional[str] = None) -> float:
        """This chat's Text size (sidecar ``text_scale``; 1.0 when unset or unreadable)."""
        if not self.bound:
            return 1.0
        try:
            value = float(self.env.chats.meta(str(cid or self.cid)).get("text_scale") or 1.0)
        except (TypeError, ValueError, AttributeError):
            return 1.0
        low, high = CHAT_TEXT_SCALE_RANGE
        return max(low, min(high, value))

    def apply_chat_text_scale(self, push: bool = True) -> float:
        """Chat settings › Text size (UI_SPEC §2.14, §2.18): the transcript and composer text styles
        at Appearance × chat scale through a nested theme (no theme at 100 %)."""
        scale = self.chat_scale()
        try:
            app = float(self.state.text_scale.value or 1.0)
        except (TypeError, ValueError):
            app = 1.0
        plain = abs(scale - 1.0) < 0.005  # 100 %: no nested theme (the page theme already has the app scale)
        key = (round(scale, 3), round(app, 3))
        if key == self._chat_theme_key:
            return scale
        self._chat_theme_key = key
        self.chat_text_scale = scale
        self.composer.set_text_scale(app * scale)
        if plain:
            self.scale_box.theme = None
            self.scale_box.dark_theme = None
        else:
            nested = ft.Theme(text_theme=build_text_theme(round(app * scale, 3)))
            self.scale_box.theme = nested
            self.scale_box.dark_theme = nested
        if push:
            self._push(self.scale_box)
        return scale

    def keyboard_text_scale(self, step: int) -> float:
        """Hardware keyboard Ctrl+= / Ctrl+- / Ctrl+0 (``ui.keyboard``; desktop zoom shortcuts): this
        chat's Text size one 5 % step up / down (clamped to CHAT_TEXT_SCALE_RANGE) or back to 100 %,
        saved in the sidecar like Chat settings › Text size. Returns the scale now applied."""
        if not self.bound:
            return self.chat_text_scale
        low, high = CHAT_TEXT_SCALE_RANGE
        current = self.chat_scale()
        scale = 1.0 if step == 0 else max(low, min(high, round(current + 0.05 * step, 2)))
        if abs(scale - current) < 0.005:
            return current
        self.env.chats.set_meta(self.cid, "text_scale", round(scale, 2))
        return self.apply_chat_text_scale()

    def apply_settings_changed(self) -> None:
        if not self.bound:
            return
        self.state.chat_context.set(self.chat_context_for(self.cid))
        self.apply_chat_text_scale()
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
            self.notify("This fix is not available here")

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

    def _on_new_scratch(self, e: Any = None) -> Optional[str]:
        """Drawer / header "New scratch chat" (UI_SPEC §2.16): an unsaved chat."""
        if not self.bound or not hasattr(self.env.chats, "new_scratch"):
            self.notify("Scratch chats need the chat store")
            return None
        cid = self.env.chats.new_scratch()
        self._switch_to(cid)
        return cid

    def _on_menu_action(self, action: str) -> None:
        cid = self.state.current_chat.value
        if action == "chat_settings":
            if self.bound:
                self.open_chat_settings()
            else:
                self.navigate("chat.settings", {"cid": cid})
        elif action == "attachments":
            self.navigate("chat.attachments", {"cid": self.cid if self.bound else cid})
        elif action == "delete":
            self.confirm_delete()
        elif action == "text_size":
            self.open_chat_settings()
        elif action == "jump_to":
            self.open_jump_to()
        elif action == "search":
            self.open_search()
        elif action == "export":
            self.open_export()
        else:
            self.notify("This chat action is not available here")

    def _on_suggestion(self, suggestion: str) -> None:
        if suggestion == "from_library":
            self.open_library_picker()  # in-chat: the book is attached here
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

    def compose_screen(self, match: Any, on_close: Optional[Callable[[], Any]] = None) -> Any:
        """``/chat/<cid>/compose``: the full-screen composer seeded from the composer text (UI_SPEC §2.3)."""
        from glossarion_mobile.ui.chat.compose_screen import ComposeScreen

        env = self.env
        return ComposeScreen(
            match, text=self.composer.text, model=self.state.chat_context.value.model,
            on_done=self.apply_composed_text,
            on_close=on_close,
            run_io=env.run_io if env is not None else None, spawn=self._spawn,
        )

    def apply_composed_text(self, text: str) -> None:
        """Done in the full-screen composer: the composer text and the chat draft."""
        self.composer.set_text(text)
        if self.bound:
            self.env.chats.set_draft(self.cid, text)
        self._on_content_changed()
        self._push(self.composer)

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
        scratch = self._is_scratch(cid)

        async def confirm() -> None:
            ok, error = await self.env.run_io(self.env.chats.delete, cid)
            if not ok:
                self.notify(error or "Could not delete chat output")
                return
            new_cid = self.env.chats.current_cid()
            self.cid = new_cid  # the deleted chat is gone: nothing to confirm on leaving it
            self.state.current_chat.set(new_cid)
            self.load_chat(new_cid)

        dialog = ConfirmDialog(title=title, body=body, confirm_label="Discard" if scratch else "Delete",
                               cancel_label="Cancel", destructive=True, on_confirm=confirm)
        dialog.show(self.page)
        return dialog

    def open_rename(self) -> Optional[ft.AlertDialog]:
        if not self.bound:
            self.notify("Renaming chats needs the chat store")
            return None
        session = self.env.chats.session(self.cid) or {}
        field = ft.TextField(label="Chat name:", value=str(session.get("title") or ""), autofocus=True)
        saved: list = []

        def save(e: Any = None) -> None:
            if saved:  # a second tap while the dialog closes
                return
            saved.append(True)
            # This dialog itself: page.pop_dialog() would close a snackbar shown since it opened.
            close_dialog(self.page, dialog)
            if self.env.chats.rename(self.cid, field.value or ""):
                self.header.set_title(self.env.chats.session(self.cid).get("title"))
                self._push(self.header.wrapper)

        dialog = ft.AlertDialog(
            title=ft.Text("Rename chat"),
            content=field,
            actions=[ft.TextButton(content="Cancel", on_click=lambda e: close_dialog(self.page, dialog)),
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

    def _source_index(self, index: int) -> Optional[int]:
        messages = self._messages()
        for position in range(min(index, len(messages)) - 1, -1, -1):
            if messages[position] and messages[position][0] in ("user", "user_file"):
                return position
        return None

    def _retranslate(self, card: AssistantMessage) -> None:
        """Retranslate: the source turn again, recorded as a new version of it (UI_SPEC §2.10)."""
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
        self.version_anchor = self._source_index(card.index)
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
            ActionItem("Copy as…", lambda: self._copy_as_menu(card), icon="CONTENT_PASTE_GO", key="copy-as"),
            ActionItem("Open output folder", (lambda: self.env.open_output(folder)) if (folder and self.env is not None
                       and self.env.open_output is not None) else None, icon="FOLDER_OPEN",
                       disabled_reason=None if folder else "No output folder for this response"),
            self._glossary_terms_item(card, folder),
            self._add_term_item(folder),
            *self._media_menu_items(card.index),
            ActionItem("Delete message", lambda: self.confirm_delete_messages([card.index]), icon="DELETE_OUTLINE",
                       destructive=True, disabled_reason=self._delete_reason()),
        ]
        sheet = ActionSheet(items, title=card.request_label or "Response", tablet=bool(self.layout.persistent_sidebar))
        sheet.show(self.page)
        return sheet

    def _glossary_targets(self, folder: str) -> list:
        """``[(label, path)]`` glossaries a response's term can go to (UI_SPEC §2.10 target picker): the
        workspace's glossary.csv, this chat's (or its series') manual glossary, the global manual
        glossary; each once, existing files only."""
        targets: list = []
        seen: set = set()

        def add(label: str, path: Any) -> None:
            path = str(path or "")
            key = os.path.normcase(os.path.abspath(path)) if path else ""
            if path and key not in seen and os.path.isfile(path):
                seen.add(key)
                targets.append((label, path))

        if folder:
            add("This response's workspace glossary", os.path.join(folder, "glossary.csv"))
        if self.bound:
            try:
                own = self.env.chats.own_overrides(self.cid) or {}
                merged = self.env.chats.overrides(self.cid) or {}
            except Exception:
                own, merged = {}, {}
            manual = merged.get("manual_glossary_path")
            add("This chat's manual glossary" if own.get("manual_glossary_path") else "Series manual glossary", manual)
            store = getattr(self.env, "store", None)
            try:
                add("Manual glossary (All chats)", store.get("manual_glossary_path", "") if store is not None else "")
            except Exception:
                pass
        return targets

    def _add_term_item(self, folder: str) -> ActionItem:
        """Response ⋯ › Add term to glossary: a new entry in the chosen glossary (GlossaryFeature's editor /
        new-entry sheet); a picker when the workspace, the chat / series and the global manual glossary
        offer more than one target."""
        adder = self.glossary_term_adder
        if adder is None:
            return ActionItem("Add term to glossary", disabled_reason="The Glossary Manager is not available",
                              icon="BOOKMARK_ADD")
        targets = self._glossary_targets(folder)
        if not targets:
            return ActionItem("Add term to glossary", icon="BOOKMARK_ADD",
                              disabled_reason="No glossary for this response (workspace, chat or manual glossary)")
        if len(targets) == 1:
            return ActionItem("Add term to glossary", lambda p=targets[0][1]: adder("", glossary_path=p),
                              icon="BOOKMARK_ADD")
        return ActionItem("Add term to glossary…", lambda: self._glossary_target_picker(targets), icon="BOOKMARK_ADD")

    def _glossary_target_picker(self, targets: list) -> Optional[ActionSheet]:
        adder = self.glossary_term_adder
        if adder is None or self.page is None:
            return None
        sheet = ActionSheet([
            ActionItem(label, lambda p=path: adder("", glossary_path=p), icon="BOOKMARK_ADD",
                       key=f"glossary-target-{index}")
            for index, (label, path) in enumerate(targets)
        ], title="Add term to", tablet=bool(self.layout.persistent_sidebar))
        self.last_sheet = sheet
        sheet.show(self.page)
        return sheet

    def _glossary_terms_item(self, card: Any, folder: str) -> ActionItem:
        """Response ⋯ › Glossary terms used (FEATURE_MAP glossary #77): the glossary entries the turn's
        source mentions, confirmed in this output (``glossary_usage.build_chapter_footnote``)."""
        targets = self._glossary_targets(folder)
        if not targets:
            return ActionItem("Glossary terms used", icon="SPELLCHECK", key="glossary-terms",
                              disabled_reason="No glossary for this response (workspace, chat or manual glossary)")
        return ActionItem("Glossary terms used", lambda p=targets[0][1]: self._spawn(self.show_glossary_terms(card, p)),
                          icon="SPELLCHECK", key="glossary-terms")

    async def show_glossary_terms(self, card: Any, glossary_path: str) -> Optional[str]:
        """Build the footnote on the io pool and show it in a Markdown sheet with Copy."""
        if not self.bound:
            return None
        index = int(card.index)
        cid = self.cid
        chats = self.env.chats
        source_index = self._source_index(index)
        messages = self._messages()
        source_message = messages[source_index] if source_index is not None else None
        label = str(getattr(card, "request_label", "") or "")

        def build() -> str:
            output = chats.message_text(cid, index, "content")
            source = source_text_for(source_message) if source_message is not None else ""
            return glossary_terms_markdown(glossary_path, source, output, label=label)

        try:
            markdown = await self.env.run_io(build)
        except Exception as exc:
            self.notify(f"Could not read the glossary: {exc}")
            return None
        from glossarion_mobile.ui.library.glossary_tab import _MarkdownSheet

        copy = self.env.copy_text if self.env is not None else None
        sheet = _MarkdownSheet(f"Glossary terms used · {os.path.basename(glossary_path)}", markdown,
                               copy=(lambda text: self._spawn_copy(text)) if copy is not None else None)
        self.last_sheet = sheet
        if self.page is not None:
            sheet.show(self.page)
        return markdown

    def _copy_as_menu(self, card: Any) -> Optional[ActionSheet]:
        """Response ⋯ › Copy as… Markdown / HTML / plain text (UI_SPEC §2.10)."""
        items = [ActionItem(label, lambda f=fmt: self._spawn(self.copy_as(card, f)), icon="CONTENT_COPY",
                            key=f"copy-as-{fmt}") for fmt, label, _key in COPY_FORMATS]
        if self.page is None:
            return None
        sheet = ActionSheet(items, title="Copy as", tablet=bool(self.layout.persistent_sidebar))
        self.last_sheet = sheet
        sheet.show(self.page)
        return sheet

    async def copy_as(self, card: Any, fmt: str) -> str:
        """Copy one response as Markdown / HTML / plain text (``chat_ops.copy_text_for``: the desktop's
        per-response ``.md`` / ``.html`` / ``.txt`` copies, else the dialog's converters)."""
        if not self.bound:
            return ""
        index = int(card.index)
        cid = self.cid
        chats = self.env.chats
        messages = self._messages()
        message = messages[index] if 0 <= index < len(messages) else None
        storage = message[6] if message is not None and len(message) > 6 and isinstance(message[6], dict) else {}

        def resolve(reference: str) -> str:
            try:
                return chats.binding_for(cid).resolve_reference(reference)
            except Exception:
                return reference

        def build() -> str:
            return copy_text_for(fmt, chats.message_text(cid, index, "content"), storage, resolve)

        text = await self.env.run_io(build)
        self._spawn_copy(text)
        label = dict((f, l) for f, l, _k in COPY_FORMATS).get(fmt, fmt)
        self.notify(f"Copied as {label}")
        return text

    def _user_actions(self, bubble: UserBubble) -> ActionSheet:
        sheet = ActionSheet(
            [
                ActionItem("Copy", lambda: self._spawn_copy(bubble.text), icon="CONTENT_COPY"),
                ActionItem("Translate again", lambda: self.translate_again(bubble.index, bubble.text), icon="REFRESH"),
                ActionItem("Edit & resend", lambda: self.edit_and_resend(bubble.index, bubble.text), icon="EDIT"),
                ActionItem("Delete", lambda: self.confirm_delete_messages(self._turn_indices(bubble.index)),
                           icon="DELETE_OUTLINE", destructive=True, disabled_reason=self._delete_reason()),
            ],
            title="Message",
        )
        sheet.show(self.page)
        return sheet

    def _user_file_actions(self, card: UserFileCard) -> ActionSheet:
        sheet = ActionSheet(
            [
                ActionItem("Copy instruction", lambda: self._spawn_copy(card.prompt), icon="CONTENT_COPY",
                           disabled_reason=None if card.prompt else "No instruction"),
                ActionItem("Run again", lambda: self._run_file_again(card), icon="REFRESH",
                           disabled_reason=None if os.path.isfile(card.path) else "The attached file is missing"),
                ActionItem("Delete", lambda: self.confirm_delete_messages(self._turn_indices(card.index)),
                           icon="DELETE_OUTLINE", destructive=True, disabled_reason=self._delete_reason()),
            ],
            title=card.name,
        )
        sheet.show(self.page)
        return sheet

    def _run_file_again(self, card: UserFileCard) -> None:
        if self.attach_file(card.path):
            self.composer.set_text(card.prompt)
            self.version_anchor = card.index
            self.on_send_action(SendAction.SEND)

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
            mode_content=lambda mode: self.mode_options_content(mode, inline=True).column,
        )
        self._haptic("light_impact")
        self.composer.set_plus_open(True)
        self.plus_sheet.show(self.page)
        return self.plus_sheet

    def _on_attach(self, tile_id: str) -> None:
        self.composer.set_plus_open(False)
        if tile_id == "library":
            self.open_library_picker()
        elif tile_id in ("files", "photos"):
            self._spawn(self.pick_and_attach(images=tile_id == "photos"))
        elif tile_id == "clipboard":
            self._spawn(self._paste_clipboard())
        else:
            self.notify("This source is not available in this build")

    def open_library_picker(self, query: str = "") -> Any:
        """＋ › From Library, the empty-chat "From Library" chip and ``/library`` (UI_SPEC §2.5 item 1):
        the Library's books in the Tools SourcePicker, inside the chat - searchable (the Library search),
        newest first, the Library's own rows with covers. The picked book is attached to this chat
        (``attach_library_book``); Send then continues it in its own workspace ("Save to: Library").
        A long-press starts a multi-selection (the Library shelves' long-press): "Use N" puts the books
        in one batch plan (``set_batch``; its Start runs one translate job, each book in its own
        workspace). Without the Tools feature the Library opens instead."""
        factory = getattr(self.env, "tools_context", None) if self.env is not None else None
        ctx = factory() if callable(factory) else None
        if ctx is None or self.page is None:
            self.navigate("library")
            return None
        from glossarion_mobile.ui.tools.source_picker import SourcePicker

        def chosen(targets: list) -> None:
            if len(targets) > 1:
                self.set_batch([str(getattr(t, "source", "") or "") for t in targets])
            elif targets:
                self.attach_library_book(targets[0])

        # (the picker closes itself before its header action runs)
        picker = SourcePicker(ctx, title="Attach from Library", multi=False, eligible=library_attach_reason,
                              on_done=chosen, segment="library", segments=("library",), searchable=True,
                              book_rows=True, include_unresolved=True, query=str(query or ""),
                              header_action=("Open Library", lambda: self.navigate("library")), find_folder=False,
                              long_press_selects=True)
        self.library_picker = picker
        picker.show(self.page)
        return picker

    def attach_library_book(self, target: Any) -> bool:
        """Attach a Library book's raw file to this chat and remember it as the chat's Library book
        (sidecar meta ``library_attachment`` ``{bid, path}``: the next Send defaults to "Save to:
        Library"); a snackbar with Remove."""
        source = str(getattr(target, "source", "") or "")
        if not source or not self.attach_file(source):
            return False
        bid = str(getattr(target, "bid", "") or "")
        if self.bound:
            self.env.chats.set_meta(self.cid, LIBRARY_ATTACHMENT_META, {"bid": bid, "path": source})
        self.notify(f"Attached {os.path.basename(source)} from the Library", action_label="Remove",
                    on_action=self.composer._remove_attachment)
        return True

    async def attach_library_query(self, query: str) -> Optional[str]:
        """``/library <title>``: one matching Library book is attached at once (an exact title wins over
        partial matches); several open the picker with the query; none says so (with Open Library)."""
        query = str(query or "").strip()
        service = self._library_service()
        if service is None or not query:
            self.open_library_picker(query)
            return None
        snapshot = getattr(service, "snapshot", None)
        if snapshot is not None and not getattr(snapshot, "scanned_at", None) and hasattr(service, "refresh"):
            await service.refresh(quiet=True, reason="chat /library")  # first use: scan quietly
        matches = await self.env.run_io(library_matches, service, query)
        if len(matches) == 1:
            return "attached" if self.attach_library_book(matches[0]) else None
        if not matches:
            self.notify(f"No Library book matches “{query}”", action_label="Open Library",
                        on_action=lambda: self.navigate("library"))
            return None
        self.open_library_picker(query)
        return "picker"

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
        """Files long-press (UI_SPEC §2.5 "Pick folder, ZIP fallback"): the folder's files become a
        BatchPlanCard (FileBridge.pick_folder); where the platform cannot pick folders, a .zip / .cbz."""
        self.composer.set_plus_open(False)
        if tile_id == "files":
            self._spawn(self.pick_folder_batch())

    def _on_tool(self, tool_id: str) -> None:
        self.composer.set_plus_open(False)
        if tool_id == "retranslate":
            self._spawn(self.open_retranslate())
            return
        if tool_id == "qa":
            # this chat's latest book workspace (Tools › QA Scanner when it has none)
            self._spawn(self.start_chat_qa())
            return
        route = TOOL_ROUTES.get(tool_id)
        if route is not None:
            self.navigate(route)
        else:
            self.notify(_NOT_YET.get(tool_id, "This tool is not available in this build"))

    # ---- U9: slash commands (UI_SPEC §2.7) ------------------------------------------------------------

    def run_slash(self, text: str) -> Optional[str]:
        """Run a composer slash command through the handlers its button uses; returns the command name
        (None for an unknown command)."""
        from glossarion_mobile.ui.chat import slash

        parsed = slash.parse_command(text)
        if parsed is None:
            self.notify(f"Unknown command: {text}")
            return None
        command, arg = parsed
        name = command.name
        tools = {"glossary": "extract_glossary", "qa": "qa", "compile epub": "compile", "compile pdf": "compile",
                 "headers": "headers", "metadata": "headers", "manga": "manga", "review": "review",
                 "async": "async", "progress": "progress"}
        if name in tools:
            self._on_tool(tools[name])
        elif name == "retranslate":
            chapters = None
            if arg:
                chapters = slash.parse_chapter_range(arg)
                if chapters is None:  # the desktop chapter-range syntax: N or N-M
                    self.notify("Use /retranslate N or /retranslate N-M (chapter numbers)")
                    return name
            self._spawn(self.open_retranslate(chapters))
        elif name == "mode":
            mode = arg.strip().lower()
            if mode not in slash.MODE_ARGS:
                self.notify("Output modes: " + ", ".join(slash.MODE_ARGS))
                return name
            self.composer.output_row.select(mode)
        elif name == "model":
            sheet = self.open_model_sheet("model")
            if arg and hasattr(sheet, "set_query"):
                sheet.set_query(arg)
        elif name == "profile":
            match = next((p for p in self._profiles() if str(p).casefold() == arg.casefold()), None) if arg else None
            if match is not None and self.env is not None:
                self._on_model_sheet_select("profile", match, False)
                self.notify(f"Prompt profile: {match}")
            else:
                self.open_model_sheet("profile")
        elif name == "lang":
            if arg and self.env is not None:
                self._on_model_sheet_select("language", arg, False)
                self.notify(f"Target language: {arg}")
            else:
                self.open_model_sheet("language")
        elif name == "policy":
            mode = slash.POLICY_ARGS.get(arg.strip().lower())
            if mode is None or mode == "manual" or not self.bound:
                self.open_chat_settings()  # Manual needs a glossary file: chosen in Chat settings
            else:
                self.env.chats.set_override(self.cid, "glossary_override_mode", mode)
                self.apply_settings_changed()
                self.notify(f"Glossary policy: {arg.strip().lower()}")
        elif name == "scratch":
            self._on_new_scratch()
        elif name == "export":
            self.open_export()
        elif name == "library":
            if arg:
                self._spawn(self.attach_library_query(arg))
            else:
                self.open_library_picker("")
        elif name == "jobs":
            self.navigate(name)
        elif name == "settings":
            if arg:
                self.open_settings_search(arg)
            else:
                self.navigate("settings")
        return name

    def open_settings_search(self, query: str) -> Any:
        """/settings <query>: the shared settings search, prefilled, in a sheet (as the section pages'
        search button shows it)."""
        from glossarion_mobile.ui.settings.search import SettingsSearch, search_sheet
        from glossarion_mobile.ui.sheets.model_sheet import sheet_env

        ctx = sheet_env().ctx
        if ctx is None or not hasattr(ctx, "open_setting"):
            self.navigate("settings")
            return None
        holder: dict = {}

        def open_hit(hit: Any) -> None:
            sheet = holder.get("sheet")
            if sheet is not None and getattr(sheet, "open", False):
                ctx.pop_dialog(sheet)
            ctx.open_setting(hit.section_id, hit.key)

        search = SettingsSearch(ctx, on_open=open_hit, autofocus=True)
        search.field.value = query
        search.set_query(query)
        holder["sheet"] = sheet = search_sheet(search)
        ctx.show_dialog(sheet)
        self.settings_search = search
        return sheet

    def _on_this_chat(self, item_id: str) -> None:
        self.composer.set_plus_open(False)
        if item_id == "chat_settings" or item_id == "glossary_policy":
            if self.bound:
                self.open_chat_settings()
            else:
                self.navigate("chat.settings", {"cid": self.state.current_chat.value})

    def _mode_options_kwargs(self) -> dict:
        overrides = self.env.chats.overrides(self.cid) if self.bound else {}
        return {
            "this_chat": overrides.get("output_mode") is not None,
            "on_this_chat": self.set_mode_scope if self.bound else None,
            "has_text": bool(self.composer.send_text()),
            "has_attachment": bool(self.composer.attachment),
            "on_generate": self.generate_from_prompt,
            "on_open_keys": lambda slug: self.navigate("settings.keys.pool", {"pool": slug}),
            "on_open_endpoints": lambda: self.navigate("settings.endpoints"),
            "config_get": self.env.config_get if self.env is not None else None,
        }

    def mode_options_content(self, mode_id: str, inline: bool = False) -> ModeOptionsContent:
        """The options of ``mode_id`` (the ＋ sheet shows them inline, §2.5; its actions close it first)."""
        from glossarion_mobile.ui.sheets.model_sheet import sheet_env

        kwargs = self._mode_options_kwargs()
        if inline:
            for name in ("on_generate", "on_open_keys", "on_open_endpoints"):
                handler = kwargs.get(name)
                if handler is not None:
                    kwargs[name] = (lambda *a, h=handler: (self._close_plus(), h(*a))[1])
        return ModeOptionsContent(mode_id, ctx=sheet_env().ctx, show_header=not inline, **kwargs)

    def _close_plus(self) -> None:
        if self.plus_sheet is not None:
            self.plus_sheet.close()
        self.composer.set_plus_open(False)

    def open_mode_options(self, mode_id: str) -> ModeOptionsSheet:
        from glossarion_mobile.ui.sheets.model_sheet import sheet_env

        self.mode_sheet = ModeOptionsSheet(mode_id, ctx=sheet_env().ctx,  # ctx: the settings tiles (U4)
                                           **self._mode_options_kwargs())
        self.mode_sheet.show(self.page)
        return self.mode_sheet

    def set_mode_scope(self, this_chat: bool) -> None:
        """Mode options "This chat only": the mode lives in the chat override, else in the global
        ``direct_text_output_mode`` (never the global ``output_mode``, UI_SPEC §2.6)."""
        if not self.bound:
            return
        mode = normalize_mode(self.state.output_mode.value.mode)
        if this_chat:
            self.env.chats.set_override(self.cid, "output_mode", mode)
        else:
            self.env.chats.set_override(self.cid, "output_mode", None)
            if self.env.config_get("direct_text_output_mode", None) != mode:
                self.env.config_set_many({"direct_text_output_mode": mode})

    def _profiles(self) -> list:
        if self.env is None:
            return []
        try:
            names = list(self.env.profiles() or [])
        except Exception:
            names = []
        return names

    def open_chat_settings(self, scope: str = "chat") -> Any:
        """Chat settings (UI_SPEC §2.14); ``scope="global"`` opens it on All chats (Settings › Direct Text)."""
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
            scope=scope,
            prefs=getattr(self.env, "prefs", None),  # the mobile-only All-chats values (Always accept)
            ctx=self.env,  # Edit prompt / New profile… open on the chat's SettingsContext
            profile_service=getattr(self.env, "profile_service", None),
            on_manage_profiles=lambda: (self.settings_sheet.close(), self.navigate("settings.profiles")),
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
            # the desktop combos' side effects: the profile's extraction method
            # (prompt_profiles.select_profile), the target-language fan-out
            # (settings_rules.fan_out_target_language)
            from glossarion_mobile.state.setting_writes import write_setting

            write_setting(self.env.store, key, value)
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

    # ---- U7: generate from prompt ------------------------------------------------------------------

    def generate_from_prompt(self, mode: Optional[str] = None) -> Any:
        """"Generate from prompt (no input)" (UI_SPEC §2.6): the composer text is the prompt; the user
        turn is recorded as ``["user", prompt]`` and a ``generate_media`` job follows."""
        mode = normalize_mode(mode or self.state.output_mode.value.mode)
        if mode not in GENERATIVE_MODES:
            self.notify("Generate from prompt is for the Image, Video and Audio modes")
            return None
        if not self.bound or self.env.runs is None or not hasattr(self.env.runs, "generate"):
            self.notify("Generating needs the job service, which is not running in this build.")
            return None
        if self.composer.attachment:
            self.notify("Remove the attachment to generate from a prompt")
            return None
        prompt = self.composer.send_text()
        if not prompt:
            self.notify("Type a prompt in the composer first")
            return None
        block = self._chat_block()
        if block is not None:
            self.notify(block.message, action_label=block.fix_label,
                        on_action=(lambda a=block.fix_action: self.run_fix(a)) if block.fix_action else None)
            return None
        cid = self.cid
        settings = self.settings(cid)
        self._after_send_ui(self.state.output_mode.value.mode)
        self.sent.append((cid, prompt, None))
        return self._spawn(self._generate(cid, prompt, mode, settings))

    async def _generate(self, cid: str, prompt: str, mode: str, settings: DirectTextSettings) -> Any:
        try:
            run = await self.env.runs.generate(cid, prompt=prompt, output_mode=mode, settings=settings,
                                               overrides=self.run_overrides(cid))
        except Exception as exc:
            log.warning("generate failed: %s", exc)
            self.notify(f"Could not start: {exc}")
            self._on_run_changed(cid)
            return None
        self._on_run_changed(cid)
        return run

    # ---- U7: media ----------------------------------------------------------------------------------------

    def _media_for(self, index: int) -> list:
        chats = self.env.chats if self.bound else None
        if chats is None or not hasattr(chats, "message_media"):
            return []
        try:
            return media_items(chats.message_media(self.cid, index))
        except Exception:
            log.debug("media lookup failed", exc_info=True)
            return []

    def media_actions(self) -> MediaActions:
        return MediaActions(open_viewer=self.open_media_viewer, save=self.save_media, share=self.share_media,
                            open_external=self.open_external, page=self.page,
                            tablet=bool(self.layout.persistent_sidebar))

    def open_media_viewer(self, items: Any, index: int = 0) -> Any:
        from glossarion_mobile.ui.components.media_viewer import MediaViewer

        viewer = MediaViewer(items, index, on_close=lambda: self._close_overlay(viewer.view), on_save=self.save_media,
                             on_share=self.share_media, on_open_external=self.open_external, audio_hub=self.audio_hub,
                             spawn=self._spawn)
        self.media_viewer = viewer
        if self.env is not None and self.env.push_overlay is not None:
            self.env.push_overlay(viewer.view)
        return viewer

    def save_media(self, path: str) -> Any:
        """Save as… (FileBridge "Save to…"; the export sheet when the app has no save hook)."""
        saver = getattr(self.env, "save_file", None) if self.env is not None else None
        if saver is not None:
            return self._spawn(self._await(saver(path)))
        return self._spawn(self._export_document(path))

    def share_media(self, path: str) -> Any:
        if self.env is None or self.env.share_files is None:
            self.notify("Sharing is not available here")
            return None
        return self._spawn(self._await(self.env.share_files([path])))

    def open_external(self, path: str) -> Any:
        """Open externally: the app's hook (system viewer); otherwise the share sheet's Open in…."""
        opener = getattr(self.env, "open_external", None) if self.env is not None else None
        if opener is not None:
            return self._spawn(self._await(opener(path)))
        return self.share_media(path)

    @staticmethod
    async def _await(result: Any) -> Any:
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Future):
            return await result
        return result

    def _media_menu_items(self, index: int) -> list:
        items = [i for i in self._media_for(index) if i.exists]
        if not items:
            return []
        first = items[0]
        return [
            ActionItem("Save media as…", lambda: self.save_media(first.path), icon="SAVE_ALT"),
            ActionItem("Share file", lambda: self.share_media(first.path), icon="IOS_SHARE"),
        ]

    # ---- U7: refine compare -------------------------------------------------------------------------------

    def _refine_backup(self, message: tuple) -> Optional[str]:
        """Blocking: the ``unrefined_backup_file`` of a refined response's workspace (Refine › Compare
        with original); ``_assistant_control`` runs it on the io pool."""
        folder = str(message[4] or "") if len(message) > 4 else ""
        if not folder or not os.path.isdir(os.path.join(folder, "unrefined_backup")):
            return None
        try:
            return find_unrefined_backup(folder, str(message[5] if len(message) > 5 else ""))
        except Exception:
            return None

    def open_compare(self, card: AssistantMessage, backup: str) -> Any:
        async def run() -> None:
            def read() -> str:
                with open(backup, "r", encoding="utf-8", errors="replace") as handle:
                    return handle.read()

            try:
                original = await self.env.run_io(read)
            except Exception as exc:
                self.notify(f"Could not read the original: {exc}")
                return
            CompareSheet(original, card.content_text).show(self.page)

        return self._spawn(run())

    # ---- U7: versions, delete message -----------------------------------------------------------------

    def _version_view(self, messages: list) -> Any:
        chats = self.env.chats
        groups = chats.version_groups(self.cid) if hasattr(chats, "version_groups") else {}
        if not groups:
            return version_view(messages, [], {})
        from glossarion_mobile.state.chat_store_adapter import message_fingerprints

        return version_view(messages, message_fingerprints(messages), groups)

    def _link_version(self, cid: str, anchor: Optional[int], index: Optional[int]) -> None:
        if anchor is None or index is None or not self.bound or not hasattr(self.env.chats, "add_version"):
            return
        messages = self.env.chats.messages(cid)
        if 0 <= int(index) < len(messages) and messages[int(index)][0] in ("user", "user_file"):
            self.env.chats.add_version(cid, int(anchor), int(index))

    def select_version(self, anchor: str, selected: int) -> None:
        if self.bound and self.env.chats.select_version(self.cid, anchor, selected):
            self.render_transcript()

    def edit_and_resend(self, index: int, text: str) -> None:
        """Edit & resend: refill the composer; the next send becomes a version of this turn."""
        self.composer.set_text(text)
        self.version_anchor = index
        self.caption.show("Editing a message · send to add a version")
        try:
            asyncio.ensure_future(self.composer.text_field.focus())
        except Exception:
            pass
        self._push(self.caption)

    def translate_again(self, index: int, text: str) -> None:
        self.composer.set_text(text)
        self.version_anchor = index
        self.on_send_action(SendAction.SEND)

    def _turn_indices(self, index: int) -> list:
        from glossarion_mobile.ui.chat.chat_ops import turn_span

        return turn_span(self._messages(), index) or [index]

    def _delete_reason(self) -> Optional[str]:
        if self.bound and self.env.runs is not None and self.env.runs.live_run(self.cid) is not None:
            return "Stop or finish the current translation first"
        return None

    def confirm_delete_messages(self, indices: list) -> Optional[ConfirmDialog]:
        """Delete message (mobile-only, UI_SPEC §2.10): removes the tuple(s) from the v2 history."""
        if not self.bound:
            return None
        reason = self._delete_reason()
        if reason:
            self.notify(reason)
            return None
        indices = sorted(set(int(i) for i in indices))
        count = len(indices)
        cid = self.cid

        async def go() -> None:
            # Blocking (managed response files are renamed, the history saved): off the loop.
            runs = self.env.runs
            jobs = runs.chat_job_turns(cid) if runs is not None and hasattr(runs, "chat_job_turns") else ()
            chats = self.env.chats
            try:
                deleted = await self.env.run_io(lambda: chats.delete_messages(cid, indices, jobs=jobs))
            except Exception as exc:
                log.exception("deleting messages failed")
                self.notify(f"Could not delete the message: {exc}")
                return
            if deleted and self.cid == cid:
                self.window = None
                self.render_transcript()
                self.refresh_send()

        body = ("Delete this message from the conversation?" if count == 1 else
                f"Delete this message and its {count - 1} response{'s' if count > 2 else ''} from the conversation?")
        dialog = ConfirmDialog(title="Delete message?", body=body + " Saved output files stay on disk.",
                               confirm_label="Delete", cancel_label="Cancel", destructive=True, on_confirm=go)
        if self.page is None:
            self._spawn(go())
            return dialog
        dialog.show(self.page)
        return dialog

    # ---- U7: scratch chats ------------------------------------------------------------------------------

    def _is_scratch(self, cid: Any) -> bool:
        chats = self.env.chats if self.env is not None else None
        return bool(chats is not None and hasattr(chats, "is_scratch") and chats.is_scratch(cid))

    def _attachment_count(self, cid: Any) -> int:
        chats = self.env.chats if self.env is not None else None
        try:
            return int(chats.attachment_count(cid)) if chats is not None and hasattr(chats, "attachment_count") else 0
        except Exception:
            return 0

    def _switch_to(self, cid: str) -> None:
        if self.state.current_chat.value != cid:
            self.state.current_chat.set(cid)  # the attached view loads it from the signal
        if self.cid != str(cid):
            self.load_chat(cid)

    def _scratch_banner(self) -> ft.Control:
        return ft.Container(
            content=ft.Row([ft.Icon(ft.Icons.EDIT_NOTE, color=ft.Colors.ON_TERTIARY_CONTAINER),
                            ft.Text(SCRATCH_BANNER, expand=True, color=ft.Colors.ON_TERTIARY_CONTAINER),
                            ft.TextButton(content="Save", on_click=lambda e: self.save_scratch(),
                                          key="scratch-banner-save")],
                           vertical_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=ft.Colors.TERTIARY_CONTAINER, border_radius=12, padding=ft.Padding.symmetric(horizontal=12, vertical=4),
            key="scratch-banner",
        )

    def save_scratch(self, cid: Optional[str] = None) -> Any:
        """Save a scratch chat into the history (UI_SPEC §2.16)."""
        cid = str(cid or self.cid)
        if not self.bound or not self._is_scratch(cid):
            return None
        if self.env.runs is not None and self.env.runs.live_run(cid) is not None:
            self.notify("Stop or finish the current translation before saving this chat.")
            return None

        async def go() -> None:
            try:
                new_cid = await self.env.run_io(self.env.chats.save_scratch, cid)
            except Exception as exc:
                self.notify(f"Could not save the scratch chat: {exc}")
                return
            if not new_cid:
                self.notify("Could not save the scratch chat")
                return
            if self.cid == cid:
                self.cid = new_cid  # it is no longer a scratch chat: no leave confirmation
                if self.state.current_chat.value != new_cid:
                    self.state.current_chat.set(new_cid)
                self.load_chat(new_cid)
            self.notify("Scratch chat saved")

        return self._spawn(go())

    def discard_scratch(self, cid: str) -> Any:
        async def go() -> None:
            await self.env.run_io(self.env.chats.discard_scratch, cid)

        return self._spawn(go())

    def _left_chat(self, previous: Optional[str]) -> Optional[ConfirmDialog]:
        """Leaving a scratch chat: empty -> discarded; otherwise "Discard scratch chat?" (Discard / Save)."""
        if not previous or not self._is_scratch(previous):
            return None
        chats = self.env.chats
        runs = self.env.runs
        if runs is not None and runs.live_run(previous) is not None:
            return None  # its run finishes into it; the drawer still lists it
        if not chats.messages(previous) and not chats.draft(previous):
            self.discard_scratch(previous)
            return None
        dialog = ConfirmDialog(title="Discard scratch chat?",
                               body="This scratch chat was never saved. Save it to keep its messages.",
                               confirm_label="Discard", cancel_label="Save", destructive=True,
                               on_confirm=lambda: self.discard_scratch(previous),
                               on_cancel=lambda: self.save_scratch(previous))
        if self.page is not None:
            dialog.show(self.page)
        return dialog

    def send_as_scratch(self) -> Optional[str]:
        """Long-press Send › Send as scratch: a scratch chat with this content, sent there."""
        if not self.bound or not hasattr(self.env.chats, "new_scratch"):
            self.notify("Scratch chats need the chat store")
            return None
        text = self.composer.send_text()
        record = self.composer.attachment
        if not text and not record:
            return None
        cid = self.env.chats.new_scratch()
        if record:
            self.env.chats.set_attachment(self.cid, None)
        self.env.chats.set_draft(self.cid, "")
        self._switch_to(cid)
        self.composer.set_text(text)
        if record:
            self.attach_file(str(record.get("path") or ""))
        self.on_send_action(SendAction.SEND)
        return cid

    def duplicate_as_scratch(self, cid: Optional[str] = None) -> Any:
        """Drawer › Duplicate as scratch: the copy gets its own response body files (blocking: off the
        loop), then the chat switches to it. Returns the task (its result: the scratch chat id)."""
        if not self.bound or not hasattr(self.env.chats, "duplicate_as_scratch"):
            self.notify("Scratch chats need the chat store")
            return None
        source = str(cid or self.cid)

        async def go() -> Optional[str]:
            try:
                new_cid = await self.env.run_io(self.env.chats.duplicate_as_scratch, source)
            except Exception as exc:
                log.exception("duplicating the chat failed")
                self.notify(f"Could not duplicate the chat: {exc}")
                return None
            if new_cid:
                self._switch_to(new_cid)
            return new_cid

        return self._spawn(go())

    # ---- U7: attachments / Library hand-off / retranslate -------------------------------------------

    def attachments_screen(self, match: Any = None) -> Any:
        from glossarion_mobile.ui.chat.attachments import AttachmentsScreen

        env = self.env
        cid = self.cid
        return AttachmentsScreen(
            match, chats=env.chats, cid=cid, run_io=env.run_io, notify=self.notify, page=self.page,
            busy=lambda folder: self._workspace_busy(cid, folder),
            open_reader=(lambda folder, source: self._spawn(self._await(env.open_reader(folder, source))))
            if env.open_reader is not None else None,
            open_progress=(lambda folder, source: self._spawn(self._await(env.open_progress(folder, source))))
            if env.open_progress is not None else None,
            share_output=lambda folder: self._spawn(self._share_workspace(folder)),
            on_migrated=self._after_migrate, spawn=self._spawn, tablet=bool(self.layout.persistent_sidebar),
        )

    def _workspace_busy(self, cid: str, folder: str) -> bool:
        """Attachments manager guard (UI_SPEC §2.17): ``attachments.workspace_busy`` (shared with the
        Library auto-migrate): chat ``cid``'s run, or an active / queued job whose folder is the
        workspace or inside it (the job card's Compile, ＋ Retranslate chapters)."""
        from glossarion_mobile.ui.chat.attachments import workspace_busy

        env = self.env
        if env is None:
            return False
        return workspace_busy(env.runs, env.jobs, cid, folder)

    def on_workspace_migrated(self, cid: Any, target: str, source: str = "") -> None:
        """A workspace of chat ``cid`` moved into the Library (ChatFeature's auto-migrate, on the UI
        loop; it records the raw and shows "Added to the Library" itself): the turn's stored paths now
        point at ``target``, so its cards look their workspace up again ("Open in Library" turns on)."""
        self._drop_extras(cid)
        if str(cid) != self.cid or not self.bound:
            return
        self.header.set_attachments(self._attachment_count(self.cid))
        self.render_transcript()
        self._push(self.header.wrapper)

    def _after_migrate(self, target: str, source: str) -> None:
        """The Attachments manager moved a workspace (a name-collision merge it confirmed)."""
        self.on_workspace_migrated(self.cid, target, source)
        hook = getattr(self.env, "after_migrate", None)
        if hook is not None:
            try:
                result = hook(target, source)
                if asyncio.iscoroutine(result):
                    self._spawn(result)  # the hook's snackbar offers "Open book"
                    return
            except Exception:
                log.exception("after-migrate hook failed")
        self.notify("Added to the Library")

    async def _share_workspace(self, folder: str) -> None:
        def pick() -> list:
            from direct_text_store import ChatStoreMixin  # shared (U3)

            preferred = ChatStoreMixin._preferred_attachment_compiled_documents(folder)
            return [os.path.join(folder, preferred[ext]) for ext in (".epub", ".pdf") if preferred.get(ext)]

        documents = await self.env.run_io(pick)
        if not documents:
            self.notify("No compiled EPUB or PDF in this workspace yet")
            return
        await self._export_document(documents[0])

    def _job_workspace(self, item: Any = None) -> str:
        """Blocking: the workspace of a job card's turn (``_job_workspace_state``)."""
        return self._job_workspace_state(item)[0]

    def _job_workspace_state(self, item: Any = None) -> tuple:
        """Blocking: ``(folder, in_attachments)`` of a job card's turn: its ``Attachments/<stem>``
        workspace (its request cards' folder, else the attachment's stem), or the Library folder the
        workspace moved to (auto-migrate rewrote the cards' folders); ``("", True)`` without one."""
        messages = self._messages()
        indices = list(getattr(item, "requests", None) or []) + [getattr(item, "report", None), getattr(item, "actions", None)]
        folder = turn_workspace(messages, indices)
        if folder:
            return folder, True
        moved = _moved_workspace(messages, indices)
        if moved:
            return moved, _in_attachments(moved)
        stem = os.path.splitext(os.path.basename(self._turn_source(item)))[0].lower()
        for folder in self.env.chats.attachment_folders(self.cid) if self.bound else []:
            if os.path.basename(os.path.normpath(folder)).lower() == stem:
                return folder, True
        return "", True

    def _library_reason(self, item: Any) -> Optional[str]:
        """Blocking: why "Open in Library" is off for a turn (None: its workspace is a Library book)."""
        try:
            folder, in_attachments = self._job_workspace_state(item)
        except Exception:
            log.debug("workspace lookup failed", exc_info=True)
            return LIBRARY_WAIT_REASON
        if folder and not in_attachments:
            return None
        try:
            import library_core  # shared (U5)

            books = tuple(getattr(library_core, "RAW_IMPORT_EXTENSIONS", ()) or ())
        except Exception:
            books = ()
        extension = os.path.splitext(self._turn_source(item))[1].lower()
        if books and extension and extension not in books:
            return "Only books (EPUB, TXT, PDF, HTML) are added to the Library"
        return LIBRARY_WAIT_REASON

    def _latest_workspace(self) -> tuple:
        """Blocking: (the chat's most recently written book workspace - an ``Attachments/<stem>`` folder or
        one that moved into the Library -, its attachment path)."""
        messages = self._messages()
        candidates: dict = {}
        for folder in self.env.chats.attachment_folders(self.cid) if self.bound else []:
            candidates.setdefault(os.path.normcase(os.path.abspath(folder)), (folder, ""))
        for item in build_items(messages, 0, len(messages)):
            if item.kind != "job" or item.index < 0:
                continue
            moved = _moved_workspace(messages, list(item.requests) + [item.report, item.actions])
            if moved and not _in_attachments(moved):
                message = messages[item.index]
                source = str(message[2] or "") if len(message) > 2 and str(message[0]) == "user_file" else ""
                candidates[os.path.normcase(os.path.abspath(moved))] = (moved, source)
        if not candidates:
            return "", ""

        def written(entry: tuple) -> float:
            progress = os.path.join(entry[0], "translation_progress.json")
            try:
                return os.path.getmtime(progress if os.path.isfile(progress) else entry[0])
            except OSError:
                return 0.0

        folder, source = max(candidates.values(), key=written)
        if not source:
            stem = os.path.basename(os.path.normpath(folder)).lower()
            for message in reversed(messages):
                if message and str(message[0]) == "user_file" and len(message) > 2:
                    if os.path.splitext(os.path.basename(str(message[2] or "")))[0].lower() == stem:
                        source = str(message[2] or "")
                        break
        return folder, source

    async def open_retranslate(self, chapters: Optional[tuple] = None) -> Optional[str]:
        """＋ › Retranslate chapters: the Progress manager's Chapters on this chat's attachment workspace
        (``chapters``: ``(start, end)`` from ``/retranslate N-M``, selected there and ready for Retranslate)."""
        if self.env is None or self.env.open_progress is None:
            self.navigate("tools.progress")
            return None
        folder, source = await self.env.run_io(self._latest_workspace)
        if not folder:
            folder, source = await self.env.run_io(self._reader_target, None)
        if not folder:
            self.notify("Attach a book and translate it first: Retranslate works on its chapters")
            return None
        if chapters:
            return await self._await(self.env.open_progress(folder, source, chapters))
        return await self._await(self.env.open_progress(folder, source))

    # ---- U7: jump to, search, export ---------------------------------------------------------------------

    def _scroll_key_for(self, index: int) -> Any:
        """The ScrollKey of the card that shows message ``index`` (its turn's JobCard for an attachment's
        request, report, actions and Library cards)."""
        messages = self._messages()
        for item in build_items(messages):
            if item.kind == "job" and item.shows(index):
                return ft.ScrollKey(f"job-{item.index}")
        return ft.ScrollKey(self._mid(index) or f"m-{index}")

    async def jump_to(self, index: int) -> None:
        """The jump procedure (UI_SPEC §2.8): bring the card that shows message ``index`` into the window
        (``window_around`` over the transcript's cards: it stays there, never snapped back to the tail),
        open a request's row in its JobCard, then ``scroll_to(scroll_key=)``."""
        if not self.bound:
            return
        messages = self._messages()
        if not (0 <= index < len(messages)):
            return
        if self.window is None or len(self._window_items(messages, self._version_view(messages))) != len(self.items):
            self.render_transcript()  # the window and its cards for the chat as it is now
        position = item_position(self.items, index)
        if position is None:
            return  # a hidden version
        start, end = self.window
        if not (start <= position < end):
            self.window = window_around(len(self.items), self.settings().rendered_card_limit, position)
            self.render_transcript()
        if position < len(self.items) - 1 and self.transcript.follow_tail:
            # the user reads the jumped-to card now: a live run's repaints no longer pull the view to the
            # end (as after "↑ earlier"; ↓ shows instead) - the client's own scroll report comes too late
            self.transcript.follow_tail = False
        self.focus_index = index
        key = self._scroll_key_for(index)
        card = getattr(self.transcript.slot_for(key), "card", None)
        if isinstance(card, JobCard) and card.reveal_request(index):
            card.push()  # the request's row is listed and its Requests list open
        self.transcript.hold_edge_loads(1.0)  # the jump's own scroll never loads more cards
        await asyncio.sleep(0)
        try:
            await self.transcript.scroll_to(scroll_key=key, duration=250)
        except Exception:
            log.debug("scroll_to failed", exc_info=True)
        await self._highlight(key)

    async def _highlight(self, key: Any) -> None:
        """The jumped-to card pulses for 1.5 s (UI_SPEC §2.8 step 3): its slot (never a frozen copy)."""
        control = self.transcript.slot_for(key)
        if control is None:
            return
        control.opacity = 0.5
        self._push(control)
        await asyncio.sleep(HIGHLIGHT_SECONDS)
        control.opacity = 1.0
        self._push(control)

    def open_jump_to(self) -> Any:
        if not self.bound:
            self.notify("Jump to needs the chat store")
            return None
        from glossarion_mobile.ui.chat.jump_to import JumpToSheet

        inputs, outputs = jump_entries(self._messages(), self.hidden_indices)
        self.jump_sheet = JumpToSheet(inputs, outputs, on_jump=lambda i: self._spawn(self.jump_to(i)),
                                      current=self.focus_index)
        self.jump_sheet.show(self.page)
        return self.jump_sheet

    def open_search(self) -> Any:
        if not self.bound:
            self.notify("Search needs the chat store")
            return None
        bar = self.header.open_search()
        self.search_hits, self.search_position = [], -1
        self._push(self.header.wrapper)
        return bar

    def close_search(self) -> None:
        self.header.close_search()
        self.search_hits, self.search_position = [], -1
        self._push(self.header.wrapper)

    def _on_search_query(self, query: str) -> None:
        self._search_query = query
        if self._search_task is None or self._search_task.done():
            self._search_task = self._spawn(self._run_search())

    async def _run_search(self) -> list:
        await asyncio.sleep(0.25)  # debounce typing
        query = getattr(self, "_search_query", "")
        cid = self.cid
        hidden = set(self.hidden_indices)
        messages = self._messages()
        hits = await self.env.run_io(
            search_matches, messages, query, lambda i: self.env.chats.message_text(cid, i, "content"), hidden)
        if getattr(self, "_search_query", "") != query:
            self._search_task = None
            self._on_search_query(self._search_query)
            return hits
        self.search_hits = list(hits)
        self.search_position = len(self.search_hits) - 1 if self.search_hits else -1
        self._show_search_position()
        if self.search_hits:
            self._spawn(self.jump_to(self.search_hits[self.search_position]))
        return hits

    def _show_search_position(self) -> None:
        total = len(self.search_hits)
        self.header.search_bar.set_count(self.search_position + 1 if total else 0, total)

    def step_search(self, delta: int) -> Optional[int]:
        if not self.search_hits:
            return None
        self.search_position = (self.search_position + int(delta)) % len(self.search_hits)
        self._show_search_position()
        target = self.search_hits[self.search_position]
        self._spawn(self.jump_to(target))
        return target

    def open_export(self, cid: Optional[str] = None) -> Optional[ActionSheet]:
        """Export chat (UI_SPEC §2.18): a Markdown transcript or a ZIP of the chat folder + its v2 JSON."""
        if not self.bound:
            self.notify("Exporting needs the chat store")
            return None
        cid = str(cid or self.cid)
        items = [
            ActionItem("Markdown transcript", lambda: self._spawn(self.export_chat(cid, "markdown")), icon="DESCRIPTION"),
            ActionItem("ZIP (chat folder + history)", lambda: self._spawn(self.export_chat(cid, "zip")),
                       icon="FOLDER_ZIP"),
        ]
        sheet = ActionSheet(items, title="Export chat", tablet=bool(self.layout.persistent_sidebar))
        if self.page is not None:
            sheet.show(self.page)
        return sheet

    def _export_dir(self) -> str:
        import tempfile

        base = getattr(self.env, "temp_dir", None) or tempfile.gettempdir()
        folder = os.path.join(str(base), "chat_exports")
        os.makedirs(folder, exist_ok=True)
        return folder

    def build_export(self, cid: str, kind: str) -> str:
        """Blocking: write the export file and return its path."""
        chats = self.env.chats
        session = chats.session(cid) or {}
        title = str(session.get("title") or "Chat")
        stem = safe_file_stem(title)
        if kind == "markdown":
            path = os.path.join(self._export_dir(), f"{stem}.md")
            text = transcript_markdown(title, chats.messages(cid), lambda i: chats.message_text(cid, i, "content"))
            temp = path + ".part"
            with open(temp, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
            os.replace(temp, path)
            return path
        binding = chats.binding_for(cid) if hasattr(chats, "binding_for") else chats.binding
        return build_chat_export_zip(session, binding.resolve_reference, os.path.join(self._export_dir(), f"{stem}.zip"))

    async def export_chat(self, cid: str, kind: str) -> Optional[str]:
        try:
            path = await self.env.run_io(self.build_export, cid, kind)
        except Exception as exc:
            self.notify(f"Could not export the chat: {exc}")
            return None
        await self._export_document(path)
        return path
