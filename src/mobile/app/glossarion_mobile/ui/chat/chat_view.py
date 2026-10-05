"""ChatView: the chat home (UI_SPEC §1.2, §2) assembled from its parts.

Body: ``SafeArea(Column[Transcript(expand), JobStrip?, StatusCaption?, Composer])``
with the ChatHeader as the View app bar (phone) or a bar on top of the main
area (tablet). The chat column is full width on phones, max 760 dp on large
phones and max 860 dp, centred, on tablets.

The view binds to ``AppState`` Signals (engine readiness, sign-in, the
Model · Profile · Target context, the JobStrip model) and re-derives the
Send/Stop inputs whenever they or the composer content change. Nothing is
wired to jobs in U1: Send actions explain what is missing or which milestone
ships the feature.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile.state.app_state import AppState, JobStripModel
from glossarion_mobile.ui.chat.composer import Composer, StatusCaption
from glossarion_mobile.ui.chat.header import ChatHeader
from glossarion_mobile.ui.chat.mode_options_sheet import ModeOptionsSheet
from glossarion_mobile.ui.chat.plus_sheet import PlusSheet
from glossarion_mobile.ui.chat.send_state import SendAction, SendInputs, SendState
from glossarion_mobile.ui.chat.transcript import Transcript
from glossarion_mobile.ui.components.info_sheet import InfoSheet
from glossarion_mobile.ui.responsive import Layout, layout_for
from glossarion_mobile.ui.shell.job_strip import JobStrip

__all__ = ["ChatView", "TOOL_ROUTES"]

# ＋ sheet tool -> route name of its surface (None: the job card arrives with the pipelines)
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
    "extract_glossary": "Glossary extraction from the chat arrives in U3",
    "retranslate": "Retranslating chapters arrives in U7",
}
_SEND_NOT_YET = {
    SendAction.SEND: "Translating from the chat arrives in U3. Your text stays in the composer.",
    SendAction.QUEUE: "Queued sends arrive in U3",
    SendAction.SEND_ONCE_WITH_MODEL: "One-shot model choice arrives in U4",
    SendAction.ADD_WITHOUT_TRANSLATING: "Adding without translating arrives in U3",
    SendAction.SEND_AS_SCRATCH: "Scratch chats arrive in U3",
    SendAction.STOP_CURRENT_AND_SEND: "Stop current & send arrives in U3",
}


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
    ) -> None:
        self.page = page
        self.state = state
        self.navigate = navigate
        self.notify = notify
        self.open_drawer = open_drawer
        self.haptics = haptics
        self.layout: Layout = layout_for(getattr(page, "width", None) or 0)
        self._unsubs: list[Callable[[], None]] = []
        self.plus_sheet: Optional[PlusSheet] = None
        self.mode_sheet: Optional[ModeOptionsSheet] = None
        self.model_sheet: Optional[InfoSheet] = None

        chat = state.chats.get(state.current_chat.value)
        self.header = ChatHeader(
            title=chat.title if chat else "New chat",
            context=state.chat_context.value,
            on_menu=self._on_menu,
            on_open_model_sheet=self.open_model_sheet,
            on_new_chat=self._on_new_chat,
            on_new_scratch=self._on_new_scratch,
            on_menu_action=self._on_menu_action,
            on_rename=lambda e: self.notify("Renaming chats arrives in U3"),
        )
        self.transcript = Transcript(on_suggestion=self._on_suggestion, on_scroll=self.header.on_transcript_scroll)
        self.job_strip = JobStrip(on_open=lambda: self.navigate("jobs"), on_stop=self._on_strip_stop)
        self.caption = StatusCaption(on_fix=self.run_fix)
        self.composer = Composer(
            mode_signal=state.output_mode,
            row_style=self.layout.output_row,
            on_plus=self.open_plus_sheet,
            on_plus_long_press=lambda: self.notify("The Photos picker arrives in U3"),
            on_send_action=self.on_send_action,
            on_content_changed=lambda _has: self.refresh_send(),
            on_expand=self._on_expand,
            on_open_mode_options=self.open_mode_options,
        )
        self.column = ft.Column(
            [self.transcript, self.job_strip, self.caption, self.composer],
            spacing=0,
            expand=True,
        )
        self.body_control: Any = None
        self.refresh_send(push=False)
        self._apply_job_strip(state.job_strip.value)

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

    def send_inputs(self) -> SendInputs:
        strip = self.state.job_strip.value
        other_running = strip is not None and strip.owner_chat != self.state.current_chat.value and strip.state in (
            "running",
            "finishing",
            "stopping",
        )
        return SendInputs(
            has_content=self.composer.has_content,
            block=self.state.send_block(),
            other_job_running=other_running,
            other_job_title=strip.title if (strip is not None and other_running) else "",
            own_job_state=None,  # JobService (U3)
        )

    def refresh_send(self, push: bool = True) -> SendState:
        inputs = self.send_inputs()
        self.composer.apply_send_inputs(inputs)
        machine = self.composer.send_button.machine
        state = machine.state
        self.caption.show(machine.caption, inputs.block if state is SendState.BLOCKED else None)
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

    # ---- actions ------------------------------------------------------------------

    def _haptic(self, kind: str) -> None:
        if self.haptics is not None:
            self.haptics.fire(kind)

    def on_send_action(self, action: SendAction) -> None:
        if action is SendAction.EXPLAIN_BLOCK:
            block = self.state.send_block()
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
            self.notify("No job is running")
            self.refresh_send()
            return
        if action is SendAction.SEND:
            self._haptic("light_impact")
        message = _SEND_NOT_YET.get(action)
        if message:
            self.notify(message)

    def run_fix(self, fix_action: Optional[str]) -> None:
        """Fix buttons of a blocked Send (§2.4) and of the status caption."""
        if fix_action == "sign_in_chatgpt":
            self.navigate("settings.accounts")
        elif fix_action == "choose_model":
            self.open_model_sheet("model")
        elif fix_action == "open_diagnostics":
            self.navigate("settings.logs")
        elif fix_action == "add_key":
            self.navigate("settings.keys")
        elif fix_action:
            self.notify("This fix arrives in U3")

    async def _on_menu(self, e: Any = None) -> None:
        if self.open_drawer is not None:
            result = self.open_drawer()
            if asyncio.iscoroutine(result):
                await result

    def _on_new_chat(self, e: Any = None) -> None:
        # Desktop _new_chat rule: an empty current chat is reused.
        if self.transcript.is_empty and not self.composer.has_content:
            self.notify("This chat is already empty")
        else:
            self.notify("Multiple chats arrive in U3")

    def _on_new_scratch(self, e: Any = None) -> None:
        self.notify("Scratch chats arrive in U3")

    def _on_menu_action(self, action: str) -> None:
        cid = self.state.current_chat.value
        if action == "chat_settings":
            self.navigate("chat.settings", {"cid": cid})
        elif action == "attachments":
            self.navigate("chat.attachments", {"cid": cid})
        else:
            self.notify("This chat action arrives in U3")

    def _on_suggestion(self, suggestion: str) -> None:
        if suggestion == "open_library":
            self.navigate("library")
        elif suggestion == "manga_page":
            self.navigate("tools.manga")
        elif suggestion == "attach_book":
            self.notify("Attaching files arrives in U3")
        elif suggestion == "paste_text":
            try:
                asyncio.ensure_future(self.composer.text_field.focus())
            except Exception:
                pass

    def _on_expand(self) -> None:
        self.navigate("chat.compose", {"cid": self.state.current_chat.value})

    def _on_strip_stop(self) -> None:
        self.notify("No job is running")

    # ---- sheets --------------------------------------------------------------------

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
        else:
            self.notify("Attaching files arrives in U3")

    def _on_attach_long_press(self, tile_id: str) -> None:
        self.composer.set_plus_open(False)
        if tile_id == "files":
            self.notify("Pick folder… arrives in U3")

    def _on_tool(self, tool_id: str) -> None:
        self.composer.set_plus_open(False)
        route = TOOL_ROUTES.get(tool_id)
        if route is not None:
            self.navigate(route)
        else:
            self.notify(_NOT_YET.get(tool_id, "This tool arrives in a later milestone"))

    def _on_this_chat(self, item_id: str) -> None:
        self.composer.set_plus_open(False)
        if item_id == "chat_settings":
            self.navigate("chat.settings", {"cid": self.state.current_chat.value})
        else:
            self.notify("Glossary policy arrives in U3")

    def open_mode_options(self, mode_id: str) -> ModeOptionsSheet:
        self.mode_sheet = ModeOptionsSheet(mode_id)
        self.mode_sheet.show(self.page)
        return self.mode_sheet

    def open_model_sheet(self, tab: str = "model") -> InfoSheet:
        """ModelSheet stub: shows the current value; the picker arrives in U4."""
        context = self.state.chat_context.value
        titles = {
            "model": ("Model", context.model),
            "profile": ("Prompt profile", context.profile),
            "language": ("Target language", context.target_language),
        }
        title, value = titles.get(tab, titles["model"])
        self.model_sheet = InfoSheet(
            title=title,
            body=f"Current: {value}\n\nSearch, favourites, provider groups and online refresh arrive with the model picker in U4.",
            actions=[
                ft.TextButton(content="Manage models", on_click=lambda e: self._sheet_nav("settings.models")),
                ft.TextButton(content="Accounts", on_click=lambda e: self._sheet_nav("settings.accounts")),
            ],
        )
        self.model_sheet.show(self.page)
        return self.model_sheet

    def _sheet_nav(self, route_name: str) -> None:
        if self.model_sheet is not None:
            self.model_sheet.close()
        self.navigate(route_name)
