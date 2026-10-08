"""Logs & diagnostics (``/settings/logs``, UI_SPEC §4.16).

U9 additions (``services.logs``): a dismissible previous-crash banner · Logging & dumps (HTTP
logging switch → ``<logs>/http_requests``, Save payloads switch → ``SAVE_PAYLOAD``, the Payloads /
HTTP requests folders with their size, Open in Files and Clear, Memory stats off by default, a
link to Debug mode) · Log files (run.log, crash.log, freeze.log, memory.log: View / Share) and
"Share logs bundle" (secrets redacted) · Developer options (Developer mode: Library Copy Path) ·
Process priority / CPU affinity (not available on mobile, with the reason).

Cards: Runtime (bootstrap result, paths, backend warm import) · Self-test
(runs ``diagnostics.selftest`` on a worker thread through ``SelfTestRunner``
and shows a PASS/FAIL card with one row per check; "Run end-to-end test" runs the
``e2e`` suite: real chat / translate / stop / resume jobs against the built-in fake
OpenAI server, in a sandbox that never touches the user's chats or settings) · Device checks (opens the
U0 spike screen: SecureStorage, foreground service, notifications,
Open-with, OAuth loopback, thread stacks, iOS background tasks) · Live log
(LogConsole over the app log buffer).
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Callable, Optional

import flet as ft

from glossarion_mobile import runtime_bootstrap as rb
from glossarion_mobile.services import logs as dl
from glossarion_mobile.services.diagnostics import SelfTestRunner, summarize
from glossarion_mobile.services.dispatcher import LogBuffer, UiDispatcher
from glossarion_mobile.state.app_state import AppState
from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.components.log_console import LogConsole
from glossarion_mobile.ui.components.reason_chip import ReasonChip
from glossarion_mobile.ui.components.section_card import SectionCard
from glossarion_mobile.ui.components.sheet import scroll_sheet
from glossarion_mobile.ui.router import RouteMatch
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.theme import HIT_TARGET, log_text, mono_family, status_color

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
        prefs: Any = None,
        paths: Any = None,
        files: Any = None,
        config_snapshot: Optional[Callable[[], dict]] = None,
        navigate: Optional[Callable[..., Any]] = None,
        notify: Optional[Callable[..., Any]] = None,
        file_ref: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(match)
        self.prefs = prefs
        self.paths = paths if paths is not None else rb.get_paths()
        self.files = files
        self.config_snapshot = config_snapshot
        self.navigate = navigate
        self.notify = notify
        self.file_ref = file_ref
        self.usage_texts: dict = {}
        self.log_column: Optional[ft.Column] = None
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
        self.e2e_button = ft.FilledTonalButton(
            content="Run end-to-end test", icon=ft.Icons.SCIENCE_OUTLINED, on_click=self._on_run_e2e,
            tooltip="Translate the built-in test book against an offline fake model (a few minutes)",
        )
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
                subtitle="Imports, offline tiktoken, EPUB/lxml, Fernet, openai/pydantic, PyMuPDF, cv2, onnxruntime, env; "
                "end-to-end: chat + glossary approval, translate, stop, resume against an offline fake model",
                children=[
                    ft.Row([self.run_button, self.e2e_button, self.run_progress], spacing=12, wrap=True, run_spacing=8),
                    self.result_text,
                    self.checks_column,
                ],
                key="diag-selftest",
            ),
            SectionCard(
                title="Device checks",
                icon="DEVELOPER_MODE",
                subtitle="SecureStorage, foreground service, notifications, Open-with, OAuth loopback, thread stacks, "
                "iOS background tasks",
                children=[self.device_button],
                key="diag-device",
            ),
        ]
        if self.console is not None:
            cards.append(SectionCard(title="Live log", icon="TERMINAL", children=[self.console], key="diag-log"))
        cards[1:1] = self._u9_cards()
        banner = self._crash_banner()
        if banner is not None:
            cards.insert(0, banner)
        self._render_result(self.state.selftest_result.value)
        self._render_running(self.state.selftest_running.value)
        return ft.ListView(
            controls=cards,
            expand=True,
            spacing=tokens.SPACING["md"],
            padding=ft.Padding.all(tokens.SPACING["md"]),
        )

    # ---- state ------------------------------------------------------------------------

    # ---- U9: logging & dumps, log files, developer options -----------------------------------

    @property
    def data_dir(self) -> str:
        return str(getattr(self.paths, "data", "") or "")

    @property
    def logs_dir(self) -> str:
        return str(getattr(self.paths, "logs", "") or "")

    def _pref(self, key: str, default: Any) -> Any:
        try:
            return self.prefs.get(key, default) if self.prefs is not None else default
        except Exception:
            return default

    def _set_pref(self, key: str, value: Any) -> None:
        if self.prefs is not None:
            try:
                self.prefs.set(key, value)
            except Exception:
                pass

    def _say(self, text: str) -> None:
        if self.notify is not None:
            try:
                self.notify(text)
            except Exception:
                pass

    def _crash_banner(self) -> Optional[ft.Control]:
        state = rb.get_state()
        if state is None or not getattr(state, "previous_crash", False):
            return None
        crash = os.path.join(self.logs_dir, "crash.log") if self.logs_dir else ""
        try:
            mtime = os.path.getmtime(crash) if crash else 0.0
        except OSError:
            mtime = 0.0
        if mtime and float(self._pref(dl.PREF_CRASH_SEEN, 0.0) or 0.0) >= mtime:
            return None
        self.crash_banner = ft.Container(
            content=ft.Column([
                ft.Row([ft.Icon(ft.Icons.WARNING_AMBER, color=ft.Colors.ERROR),
                        ft.Text("Glossarion closed unexpectedly last time", weight=ft.FontWeight.W_600, expand=True)],
                       spacing=8),
                ft.Text("crash.log has the details; share it with the logs bundle when you report the problem.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL),
                ft.Row([ft.TextButton(content="View crash log", on_click=lambda e: self.view_log(crash)),
                        ft.TextButton(content="Dismiss", on_click=lambda e: self.dismiss_crash(mtime))], wrap=True),
            ], spacing=4, tight=True),
            bgcolor=ft.Colors.ERROR_CONTAINER, border_radius=tokens.RADII["card"],
            padding=ft.Padding.all(tokens.SPACING["md"]), key="diag-crash-banner")
        return self.crash_banner

    def dismiss_crash(self, mtime: float) -> None:
        self._set_pref(dl.PREF_CRASH_SEEN, float(mtime or time.time()))
        banner = getattr(self, "crash_banner", None)
        if banner is not None:
            banner.visible = False
            self._push(banner)

    def _u9_cards(self) -> list:
        self.env_check_status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_MEDIUM, visible=False,
                                        key="diag-env-check-verdict")
        self.env_check_lines = ft.Column([], spacing=0, tight=True, visible=False, key="diag-env-check-lines")
        http_on = bool(self._pref(dl.PREF_HTTP_LOG, False))
        payload_on = bool(self._pref(dl.PREF_SAVE_PAYLOAD, True))
        memory_on = bool(self._pref(dl.PREF_MEMORY_STATS, False))
        self.http_switch = ft.Switch(label="HTTP logging", value=http_on, key="diag-http-log",
                                     on_change=lambda e: self.set_http_logging(bool(e.control.value)))
        self.payload_switch = ft.Switch(label="Save payloads", value=payload_on, key="diag-save-payload",
                                        on_change=lambda e: self.set_save_payload(bool(e.control.value)))
        self.memory_switch = ft.Switch(label="Memory stats", value=memory_on, key="diag-memory",
                                       on_change=lambda e: self.set_memory_stats(bool(e.control.value)))
        folder_rows: list = []
        for folder_id, label, path in dl.debug_folders(self.data_dir, self.logs_dir):
            text = ft.Text("…", theme_style=ft.TextThemeStyle.BODY_SMALL)
            self.usage_texts[folder_id] = text
            root = "payloads" if folder_id == "payloads" else "logs"
            folder_rows.append(ft.ListTile(
                title=ft.Text(label), subtitle=text, dense=True, key=f"diag-folder-{folder_id}",
                trailing=ft.Row([
                    ft.IconButton(icon=ft.Icons.FOLDER_OPEN, tooltip="Open in Files", size_constraints=HIT_TARGET,
                                  on_click=lambda e, r=root, p=path: self.open_folder(r, p)),
                    ft.IconButton(icon=ft.Icons.DELETE_SWEEP, tooltip=f"Clear {label}", size_constraints=HIT_TARGET,
                                  on_click=lambda e, i=folder_id, p=path, n=label: self.confirm_clear(i, p, n)),
                ], spacing=0, tight=True)))
        logging_card = SectionCard(
            title="Logging & dumps", icon="BUG_REPORT",
            subtitle="Debug output on this device; shared logs are redacted.",
            children=[
                self.http_switch,
                ft.Text("Every API request and response, written to logs/http_requests for the next requests "
                        "(includes text you translate).", theme_style=ft.TextThemeStyle.BODY_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                self.payload_switch,
                ft.Text("Request/response dumps in Payloads (desktop default on). Capped at 400 MB at launch.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                *folder_rows,
                self.memory_switch,
                ft.Text("Off by default (the desktop memory logger is disabled); writes logs/memory.log.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                ft.TextButton(content="Debug mode (verbose payloads)…", icon=ft.Icons.TUNE, key="diag-debug-mode",
                              on_click=lambda e: self.open_debug_settings()),
                ft.Row([
                    ft.FilledTonalButton(content="Check environment", icon=ft.Icons.FACT_CHECK_OUTLINED,
                                         key="diag-env-check", on_click=lambda e: self._spawn(self.check_environment())),
                    ft.TextButton(content="Env preview", icon=ft.Icons.DATA_OBJECT, key="diag-env-preview",
                                  on_click=lambda e: self.open_env_preview()),
                ], wrap=True, spacing=6),
                self.env_check_status,
                self.env_check_lines,
            ], key="diag-logging")
        self.log_column = ft.Column([ft.Text("…", theme_style=ft.TextThemeStyle.BODY_SMALL)], spacing=0, tight=True)
        self.bundle_button = ft.FilledTonalButton(content="Share logs bundle", icon=ft.Icons.IOS_SHARE,
                                                  key="diag-share-bundle",
                                                  on_click=lambda e: self._spawn(self.share_bundle()))
        logs_card = SectionCard(title="Log files", icon="DESCRIPTION",
                                subtitle="run.log, crash.log, freeze.log; secrets are redacted when shared",
                                children=[self.log_column, self.bundle_button], key="diag-logfiles")
        self.dev_switch = ft.Switch(label="Developer mode", value=bool(self._pref(dl.PREF_DEVELOPER, False)),
                                    key="diag-developer",
                                    on_change=lambda e: self._set_pref(dl.PREF_DEVELOPER, bool(e.control.value)))
        dev_card = SectionCard(title="Developer options", icon="DEVELOPER_MODE", children=[
            self.dev_switch,
            ft.Text("Shows developer actions such as Library › ⋯ › Copy Path.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=ft.Colors.ON_SURFACE_VARIANT),
            *[ft.ListTile(title=ft.Text(name, color=ft.Colors.ON_SURFACE_VARIANT), disabled=True, dense=True,
                          subtitle=ft.Row([ReasonChip(reason="Not available on mobile", detail=dl.PRIORITY_REASON)]),
                          key=f"diag-{key}")
              for name, key in (("Process priority", "priority"), ("CPU affinity", "affinity"))],
        ], key="diag-developer-card")
        return [logging_card, logs_card, dev_card]

    def _spawn(self, coro: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return dispatcher.spawn(coro)
        import asyncio

        try:
            return asyncio.ensure_future(coro)
        except RuntimeError:
            coro.close()
            return None

    async def _io(self, fn: Callable[..., Any], *args: Any) -> Any:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False):
            return await dispatcher.run_in_thread(lambda: fn(*args), name="gl-diagnostics-io")
        import asyncio

        return await asyncio.to_thread(fn, *args)

    def set_http_logging(self, enabled: bool) -> None:
        self._set_pref(dl.PREF_HTTP_LOG, bool(enabled))
        folder = dl.apply_http_logging(bool(enabled), self.logs_dir)
        self._say(f"HTTP logging on · {folder}" if enabled else "HTTP logging off")

    def set_save_payload(self, enabled: bool) -> None:
        self._set_pref(dl.PREF_SAVE_PAYLOAD, bool(enabled))
        dl.apply_save_payload(bool(enabled))

    def set_memory_stats(self, enabled: bool) -> None:
        self._set_pref(dl.PREF_MEMORY_STATS, bool(enabled))
        if not dl.apply_memory_stats(bool(enabled)) and enabled:
            self._say("Memory stats are not available in this build")

    async def check_environment(self) -> Any:
        """Debug "Check environment" (desktop ``_run_debug_check``): the shared
        ``initialize_environment_variables()`` + ``debug_environment_variables(show_all=True)`` on a
        HeadlessOwner of the current config, under the Env preview scope; values are redacted."""
        from glossarion_mobile.ui.screens.env_preview import run_env_check

        if self.config_snapshot is None:
            self._say("The settings store is not available in this session")
            return None
        self.env_check_status.value = "Checking…"
        self.env_check_status.visible = True
        self._push(self.env_check_status)
        try:
            config = self.config_snapshot() or {}
            result = await self._io(run_env_check, dict(config))
        except Exception as exc:
            self.env_check_status.value = f"❌ Debug check failed: {exc}"
            self._push(self.env_check_status)
            return None
        self.env_check_status.value = result.verdict
        self.env_check_status.color = ft.Colors.PRIMARY if result.passed else ft.Colors.ERROR
        shown = [line for line in result.lines if "[ENV_DEBUG]" in line and not line.startswith("✅")] or \
            result.lines[-12:]
        self.env_check_lines.controls = [log_text(line, page=self.page) for line in shown[:80]]
        if self.copy_handler is not None and result.lines:
            self.env_check_lines.controls.append(ft.TextButton(
                content=f"Copy all {len(result.lines)} lines (redacted)", icon=ft.Icons.COPY_ALL,
                key="diag-env-check-copy", on_click=lambda e, text="\n".join(result.lines): self.copy_handler(text)))
        self.env_check_lines.visible = True
        self._push(self.env_check_status, self.env_check_lines)
        return result

    def open_env_preview(self) -> None:
        if self.navigate is not None:
            try:
                self.navigate("settings.env_preview")
            except Exception:
                pass

    def open_debug_settings(self) -> None:
        if self.navigate is not None:
            try:
                self.navigate("settings.section", {"section": "other.debug"})
            except Exception:
                pass

    def open_folder(self, root: str, path: str) -> None:
        if self.navigate is None:
            return
        if path and not os.path.isdir(path):
            self._say("Nothing saved there yet")
            return
        self.navigate("tools.files", {"root": root})

    def confirm_clear(self, folder_id: str, path: str, label: str) -> Optional[ConfirmDialog]:
        def go() -> None:
            self._spawn(self.clear_folder(folder_id, path, label))

        if self.page is None:
            go()
            return None
        dialog = ConfirmDialog(title=f"Clear {label}?", body=f"Delete every file in {label}? This cannot be undone.",
                               confirm_label="Clear", destructive=True, on_confirm=go)
        dialog.show(self.page)
        return dialog

    async def clear_folder(self, folder_id: str, path: str, label: str) -> int:
        try:
            removed = await self._io(dl.clear_debug_folder, path, [self.data_dir, self.logs_dir])
        except Exception as exc:
            self._say(f"Could not clear {label}: {exc}")
            return 0
        self._say(f"Cleared {label} ({removed} item{'s' if removed != 1 else ''})")
        await self.measure()
        return removed

    async def measure(self) -> dict:
        out: dict = {}
        for folder_id, _label, path in dl.debug_folders(self.data_dir, self.logs_dir):
            size, files = await self._io(dl.folder_usage, path)
            out[folder_id] = (size, files)
            text = self.usage_texts.get(folder_id)
            if text is not None:
                text.value = f"{_human(size)} · {files:,} file{'s' if files != 1 else ''}"
                self._push(text)
        return out

    async def load_log_files(self) -> list:
        items = await self._io(dl.log_files, self.logs_dir)
        if self.log_column is None:
            return items
        rows: list = []
        for item in items:
            rows.append(ft.ListTile(
                title=ft.Text(item.name), dense=True, key=f"diag-logfile-{item.name}",
                subtitle=ft.Text(f"{_human(item.size)} · {time.strftime('%Y-%m-%d %H:%M', time.localtime(item.mtime))}",
                                 theme_style=ft.TextThemeStyle.BODY_SMALL),
                on_click=lambda e, p=item.path: self.view_log(p),
                trailing=ft.IconButton(icon=ft.Icons.IOS_SHARE, tooltip=f"Share {item.name}", size_constraints=HIT_TARGET,
                                       on_click=lambda e, p=item.path: self._spawn(self.share_paths([p])))))
        self.log_column.controls = rows or [ft.Text("No log files yet.", theme_style=ft.TextThemeStyle.BODY_SMALL)]
        self._push(self.log_column)
        return items

    def view_log(self, path: str) -> Optional[ft.BottomSheet]:
        """A monospace viewer (log size) over the last 200 KB of a log file."""
        try:
            with open(path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 200 * 1024))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError as exc:
            self._say(f"Could not open {os.path.basename(path)}: {exc}")
            return None
        sheet = scroll_sheet(os.path.basename(path), [
            log_text(text or "(empty)", page=self.page),
        ], key="diag-log-viewer")
        if self.page is not None:
            self.page.show_dialog(sheet)
        return sheet

    async def share_paths(self, paths: list) -> bool:
        share = getattr(self.files, "share", None) if self.files is not None else None
        if share is None:
            self._say("Sharing is not available in this session")
            return False
        try:
            return bool(await share(paths))
        except Exception as exc:
            self._say(f"Could not share: {exc}")
            return False

    async def share_bundle(self) -> Optional[str]:
        """Share logs bundle: the log files (secrets redacted) + a redacted environment summary."""
        config = {}
        if self.config_snapshot is not None:
            try:
                config = dict(self.config_snapshot() or {})
            except Exception:
                config = {}
        out_dir = str(getattr(self.paths, "temp", "") or getattr(self.paths, "cache", "") or self.logs_dir)
        try:
            path = await self._io(lambda: dl.build_logs_bundle(self.logs_dir, out_dir, config=config,
                                                               extra_lines=runtime_lines(self.page)))
        except Exception as exc:
            self._say(f"Could not build the logs bundle: {exc}")
            return None
        await self.share_paths([path])
        return path

    def did_show(self) -> None:
        self._spawn(self.measure())
        self._spawn(self.load_log_files())
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
        self.e2e_button.disabled = bool(running)
        self.run_progress.visible = bool(running)
        if running:
            suite = getattr(self.runner, "current_suite", None) or "smoke"
            self.result_text.value = f"Running suite '{suite}'…"

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
        self._push(self.run_button, self.e2e_button, self.run_progress, self.result_text)

    def _on_refresh(self, e: Any = None) -> None:
        self.runtime_text.value = "\n".join(runtime_lines(self.page))
        self._push(self.runtime_text)

    async def _on_run(self, e: Any = None) -> Optional[dict]:
        return await self.runner.run("smoke", source="diagnostics")

    async def _on_run_e2e(self, e: Any = None) -> Optional[dict]:
        return await self.runner.run("e2e", source="diagnostics")

    def _on_device_checks(self, e: Any = None) -> None:
        if self.open_device_checks is not None:
            self.open_device_checks()


def _human(size: Any) -> str:
    try:
        value = float(size or 0)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"
