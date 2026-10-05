"""Logs & diagnostics (``/settings/logs``, UI_SPEC §4.16) - the U1 version.

Cards: Runtime (bootstrap result, paths, backend warm import) · Self-test
(runs ``diagnostics.selftest`` on a worker thread through ``SelfTestRunner``
and shows a PASS/FAIL card with one row per check) · Device checks (opens the
U0 spike screen: SecureStorage, foreground service, notifications,
Open-with, OAuth loopback, thread stacks, iOS background tasks) · Live log
(LogConsole over the app log buffer).
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile import runtime_bootstrap as rb
from glossarion_mobile.services.diagnostics import SelfTestRunner, summarize
from glossarion_mobile.services.dispatcher import LogBuffer, UiDispatcher
from glossarion_mobile.state.app_state import AppState
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.log_console import LogConsole
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET, mono_family, status_color

__all__ = ["DiagnosticsScreen", "runtime_lines"]

_CHECK_STATUS = {"pass": ("CHECK_CIRCLE", "completed"), "fail": ("ERROR", "failed"), "skip": ("REMOVE_CIRCLE_OUTLINE", "skipped")}


def runtime_lines(page: Any = None) -> list[str]:
    state = rb.get_state()
    if state is None:
        return ["bootstrap() did not run in this process"]
    paths = state.paths
    backend = state.backend_result
    if backend is None:
        backend_text = "warming up…"
    elif backend.get("ok"):
        backend_text = f"ready: {backend.get('modules')} modules in {backend.get('secs')} s"
    else:
        failed = ", ".join(sorted((backend.get("failed") or {}).keys())[:4])
        backend_text = f"FAILED ({failed})"
    platform = getattr(getattr(page, "platform", None), "value", None)
    lines = [
        f"Glossarion {state.version.get('version') or '?'} (build {state.version.get('build') or '?'})",
        f"platform {paths.platform} · page {platform or '?'} · python {sys.version.split()[0]}",
        f"backend ({paths.backend_source}): {paths.backend_dir}",
        f"backend import: {backend_text}",
        f"data: {paths.data}",
        f"docs: {paths.docs}",
        f"boot {state.secs} s · previous crash: {'yes' if state.previous_crash else 'no'}",
    ]
    if state.errors:
        lines.append(f"boot errors: {'; '.join(state.errors)[:300]}")
    return lines


class DiagnosticsScreen(Screen):
    title = "Logs & diagnostics"

    def __init__(
        self,
        match: Optional[RouteMatch],
        *,
        page: Any,
        state: AppState,
        dispatcher: UiDispatcher,
        runner: SelfTestRunner,
        log_buffer: Optional[LogBuffer] = None,
        open_device_checks: Optional[Callable[[], Any]] = None,
        copy_handler: Optional[Callable[[str], Any]] = None,
        dark: bool = False,
    ) -> None:
        super().__init__(match)
        self.page = page
        self.state = state
        self.dispatcher = dispatcher
        self.runner = runner
        self.log_buffer = log_buffer
        self.open_device_checks = open_device_checks
        self.copy_handler = copy_handler
        self.dark = dark
        self._unsubs: list[Callable[[], None]] = []

    # ---- body -----------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        mono = mono_family(self.page)
        self.runtime_text = ft.Text("\n".join(runtime_lines(self.page)), selectable=True, font_family=mono, size=12)
        self.run_button = ft.FilledTonalButton(content="Run self-test", icon=ft.Icons.PLAY_ARROW, on_click=self._on_run)
        self.run_progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.result_text = ft.Text(summarize(self.state.selftest_result.value), selectable=True)
        self.checks_column = ft.Column([], spacing=2, tight=True)
        self.device_button = ft.FilledTonalButton(
            content="Open device checks", icon=ft.Icons.DEVELOPER_MODE, on_click=self._on_device_checks
        )
        self.console = LogConsole(
            buffer=self.log_buffer,
            dispatcher=self.dispatcher,
            copy_handler=self.copy_handler,
            list_height=320,
        ) if self.log_buffer is not None else None
        cards: list[ft.Control] = [
            SectionCard(
                title="Runtime",
                icon="INFO_OUTLINE",
                children=[self.runtime_text],
                trailing=ft.IconButton(
                    icon=ft.Icons.REFRESH, tooltip="Refresh", on_click=self._on_refresh, size_constraints=HIT_TARGET
                ),
                key="diag-runtime",
            ),
            SectionCard(
                title="Self-test",
                icon="FACT_CHECK",
                subtitle="Imports, offline tiktoken, EPUB/lxml, Fernet, openai/pydantic, PyMuPDF, cv2, onnxruntime, env",
                children=[ft.Row([self.run_button, self.run_progress], spacing=12), self.result_text, self.checks_column],
                key="diag-selftest",
            ),
            SectionCard(
                title="Device checks (U0)",
                icon="DEVELOPER_MODE",
                subtitle="SecureStorage, foreground service, notifications, Open-with, OAuth loopback, thread stacks, "
                "iOS background tasks",
                children=[self.device_button],
                key="diag-device",
            ),
        ]
        if self.console is not None:
            cards.append(SectionCard(title="Live log", icon="TERMINAL", children=[self.console], key="diag-log"))
        self._render_result(self.state.selftest_result.value)
        self._render_running(self.state.selftest_running.value)
        return ft.ListView(
            controls=cards,
            expand=True,
            spacing=tokens.SPACING["md"],
            padding=ft.Padding.all(tokens.SPACING["md"]),
        )

    # ---- state ------------------------------------------------------------------------

    def did_show(self) -> None:
        if self._unsubs:
            return
        self._unsubs = [
            self.state.selftest_result.subscribe(self._on_result),
            self.state.selftest_running.subscribe(self._on_running),
            self.state.backend.subscribe(lambda _v: self._on_refresh()),
        ]

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []
        if self.console is not None:
            self.console.detach()

    def _render_running(self, running: bool) -> None:
        self.run_button.disabled = bool(running)
        self.run_progress.visible = bool(running)
        if running:
            self.result_text.value = "Running suite 'smoke'…"

    def _render_result(self, result: Optional[dict]) -> None:
        self.result_text.value = summarize(result)
        rows: list[ft.Control] = []
        for check in (result or {}).get("checks", []):
            icon_name, status = _CHECK_STATUS.get(check.get("status"), ("HELP_OUTLINE", "pending"))
            color = status_color(status, self.dark)
            note = check.get("error") or check.get("reason") or ""
            rows.append(
                ft.Row(
                    [
                        ft.Icon(getattr(ft.Icons, icon_name), color=color, size=16),
                        ft.Text(
                            f"{check.get('name')} ({check.get('secs', 0)} s){(' · ' + note[:160]) if note else ''}",
                            theme_style=ft.TextThemeStyle.BODY_SMALL,
                            selectable=True,
                            expand=True,
                        ),
                    ],
                    spacing=6,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    key=f"check-{check.get('name')}",
                )
            )
        self.checks_column.controls = rows

    def _push(self, *controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass

    def _on_result(self, result: Optional[dict]) -> None:
        self._render_result(result)
        self._push(self.result_text, self.checks_column)

    def _on_running(self, running: bool) -> None:
        self._render_running(running)
        self._push(self.run_button, self.run_progress, self.result_text)

    def _on_refresh(self, e: Any = None) -> None:
        self.runtime_text.value = "\n".join(runtime_lines(self.page))
        self._push(self.runtime_text)

    async def _on_run(self, e: Any = None) -> Optional[dict]:
        return await self.runner.run("smoke", source="diagnostics")

    def _on_device_checks(self, e: Any = None) -> None:
        if self.open_device_checks is not None:
            self.open_device_checks()
