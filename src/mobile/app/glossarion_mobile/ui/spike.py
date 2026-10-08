"""U0 device-spike screen.

A single Material 3 page listing the device checks the U0 milestone must prove
on real phones (plan section 8, U0): self-test, SecureStorage, a foreground
service that keeps a Python thread alive for 25 minutes with the notification
Stop button reaching Python, notifications, Open-with/Share without touching
``page.route``, the OAuth loopback + return deep link, 16 MiB thread stacks and
the iOS background-task APIs. Each card shows its result text; long-running
results are also printed as ``GLOSSARION_SPIKE`` lines (logcat ``flet.python``).

Since U1 the screen is reached from Settings > Logs & diagnostics > "Open device
checks": the app builds ``SpikeApp(page, embedded=True, ...)`` with its own
NativeBridge, UiDispatcher, Router and in-app browser opener, and pushes
``build_view()`` as a non-routable View. The app forwards ``/oauth/return?p=spike``
and lifecycle events to it. ``main(page)`` still runs the screen standalone
(it then owns the page, its routes and the ``GLOSSARION_READY`` marker).

Threading rules (Flet 1.0 runs sync handlers on the asyncio loop):
* every handler here is ``async`` and never blocks; blocking work runs on a
  dedicated thread (16 MiB stack) through ``_in_thread`` (UiDispatcher.run_in_thread);
* worker threads touch controls only through ``post()`` (``call_soon_threadsafe``)
  and native calls only through ``submit()`` (``run_coroutine_threadsafe``).
"""

from __future__ import annotations

import asyncio
import html
import http.server
import json
import logging
import secrets
import sys
import threading
import time
import webbrowser
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, urlsplit

import flet as ft

from glossarion_mobile import JOB_TASK_ID_PREFIX, OAUTH_RETURN_URL, SELFTEST_ROUTE
from glossarion_mobile import runtime_bootstrap as rb
from glossarion_mobile.services.browser import InAppUrlOpener
from glossarion_mobile.services.dispatcher import UiDispatcher
from glossarion_mobile.services.native import NativeBridge
from glossarion_mobile.ui.router import RouteMatch, Router, launch_links, parse_route

log = logging.getLogger("glossarion.spike")

SEED_COLOR = "#E18F98"
FGS_DURATION_S = 25 * 60
CONTINUED_PROCESSING_S = 60
OAUTH_TIMEOUT_S = 300
MARKER_SPIKE = "GLOSSARION_SPIKE"
# Route string of the embedded View: identifies it for page.on_view_pop; never routed to.
DEVICE_CHECKS_VIEW_ROUTE = "/settings/logs/device-checks"

_STATUS_STYLE = {
    "idle": (ft.Icons.RADIO_BUTTON_UNCHECKED, ft.Colors.OUTLINE),
    "running": (ft.Icons.HOURGLASS_TOP, ft.Colors.AMBER),
    "pass": (ft.Icons.CHECK_CIRCLE, ft.Colors.GREEN),
    "fail": (ft.Icons.ERROR, ft.Colors.RED),
    "info": (ft.Icons.INFO, ft.Colors.BLUE),
    "n/a": (ft.Icons.REMOVE_CIRCLE_OUTLINE, ft.Colors.OUTLINE),
}


def _fmt_secs(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_theme() -> ft.Theme:
    return ft.Theme(
        color_scheme_seed=SEED_COLOR,
        use_material3=True,
        visual_density=ft.VisualDensity.COMPACT,
    )


class CheckCard:
    """One spike check: status icon, title, description, buttons, result text."""

    def __init__(
        self,
        key: str,
        title: str,
        description: str,
        actions: list[tuple[str, Callable[..., Any]]],
    ) -> None:
        self.key = key
        self.status = "idle"
        icon, color = _STATUS_STYLE["idle"]
        self.icon = ft.Icon(icon, color=color, size=20)
        self.result = ft.Text("Not run yet.", selectable=True, size=12)
        buttons = [ft.FilledTonalButton(content=label, on_click=handler) for label, handler in actions]
        self.control = ft.Card(
            key=f"check-{key}",
            content=ft.Container(
                padding=12,
                content=ft.Column(
                    tight=True,
                    spacing=6,
                    controls=[
                        ft.Row([self.icon, ft.Text(title, weight=ft.FontWeight.BOLD, size=15, expand=True)]),
                        ft.Text(description, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
                        ft.Row(buttons, wrap=True, spacing=8, run_spacing=6) if buttons else ft.Container(),
                        self.result,
                    ],
                ),
            ),
        )

    def set(self, status: str, text: Optional[str] = None) -> None:
        """Update on the Flet loop thread (use ``SpikeApp.post`` from workers)."""
        self.status = status
        icon, color = _STATUS_STYLE.get(status, _STATUS_STYLE["info"])
        self.icon.icon = icon
        self.icon.color = color
        if text is not None:
            self.result.value = text
        try:
            self.control.update()
        except Exception:  # not mounted yet; goes out with the next page update
            pass


class LoopbackServer:
    """127.0.0.1 HTTP server standing in for an OAuth provider's redirect target.

    ``GET /start?nonce=N`` answers with a page that immediately navigates to
    ``glossarion://app/oauth/return?p=spike&nonce=N`` (plus a button, because
    Custom Tabs may require a user gesture for custom-scheme navigation).
    """

    def __init__(self, nonce: str, on_hit: Callable[[str], Any]) -> None:
        return_url = f"{OAUTH_RETURN_URL}?p=spike&nonce={nonce}"
        page = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Glossarion sign-in spike</title>"
            "<style>body{font-family:sans-serif;background:#121826;color:#f5e9ec;padding:24px}"
            "a{display:inline-block;margin-top:16px;padding:12px 20px;border-radius:20px;"
            "background:#E18F98;color:#121826;text-decoration:none;font-weight:bold}</style></head><body>"
            "<h2>Loopback reached</h2><p>Returning to Glossarion&hellip;</p>"
            f"<a id='back' href='{html.escape(return_url)}'>Return to Glossarion</a>"
            "<script>setTimeout(function(){location.href=document.getElementById('back').href;},700);</script>"
            "</body></html>"
        ).encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib name
                parts = urlsplit(self.path)
                if parts.path == "/start" and parse_qs(parts.query).get("nonce", [""])[0] == nonce:
                    on_hit(self.path)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                else:
                    self.send_error(404)

            def log_message(self, fmt: str, *args: Any) -> None:
                log.info("loopback: " + fmt, *args)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, name="gl-spike-loopback", daemon=True)
        self.thread.start()

    def close(self) -> None:
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass


def stack_probe() -> dict[str, Any]:
    """Deep C and Python recursion on the calling (worker) thread."""
    out: dict[str, Any] = {"thread_stack_size": rb.current_thread_stack_size()}
    reached = 0
    for depth in (1000, 2000, 4000, 8000, 12000, 16000, 24000):
        nested: list[Any] = []
        for _ in range(depth):
            nested = [nested]
        try:
            repr(nested)
            json.dumps(nested)
            reached = depth
        except RecursionError as exc:  # the interpreter's C-recursion guard, not a crash
            out["c_guard"] = f"depth {depth}: {exc}"
            break
    out["c_depth_ok"] = reached
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 60000))
    try:

        def recurse(n: int) -> int:
            return 0 if n == 0 else 1 + recurse(n - 1)

        out["py_depth_ok"] = recurse(50000)
    except RecursionError as exc:
        out["py_guard"] = str(exc)
    finally:
        sys.setrecursionlimit(old_limit)
    return out


class SpikeApp:
    def __init__(
        self,
        page: ft.Page,
        *,
        native: Optional[NativeBridge] = None,
        dispatcher: Optional[UiDispatcher] = None,
        router: Optional[Router] = None,
        opener: Optional[InAppUrlOpener] = None,
        embedded: bool = False,
    ) -> None:
        self.page = page
        self.embedded = embedded
        self.dispatcher = dispatcher if dispatcher is not None else UiDispatcher(page).bind()
        self.loop = self.dispatcher.loop or asyncio.get_running_loop()
        self.router = router if router is not None else Router()
        self.state = rb.get_state()
        self.paths = rb.get_paths()
        self.is_mobile = getattr(page.platform, "value", None) in ("android", "ios") and not page.web

        # Services: keep strong references (Flet unregisters unreferenced services).
        if opener is not None:
            self.opener = opener
            self.url_launcher = opener.url_launcher
        else:
            self.url_launcher = ft.UrlLauncher()
            self.opener = InAppUrlOpener(self.dispatcher, self.url_launcher, is_mobile=self.is_mobile)
        # GlossarionNative on Android/iOS, stub elsewhere (shared with the app when embedded)
        self.native = native if native is not None else NativeBridge(page)
        self._secure_storage: Any = None
        self._permissions: Any = None

        self.native.add_listener("share", self._on_share)
        self.native.add_listener("foreground", self._on_foreground)
        self.native.add_listener("notification", self._on_notification)
        self.native.add_listener("background_task", self._on_background_task)

        self.initial_route: Optional[str] = None
        self.initial_shared: list = []
        self._routed_link_ids: set[str] = set()
        self.lifecycle: list[str] = []
        self._selftest_running = False
        self._fgs_thread: Optional[threading.Thread] = None
        self._fgs_stop = threading.Event()
        self._fgs_stop_reason: Optional[str] = None
        self._oauth: Optional[dict[str, Any]] = None
        self._bg_task_id: int = -1
        self._cp_thread: Optional[threading.Thread] = None
        self._cp_stop = threading.Event()
        self.cards: dict[str, CheckCard] = {}
        self._build_cards()

    # ---- plumbing (UiDispatcher) ----------------------------------------------

    def post(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` on the Flet loop (callable from any thread)."""
        self.dispatcher.post(fn, *args)

    def submit(self, coro_fn: Callable[..., Any], *args: Any):
        """Schedule ``coro_fn(*args)`` on the Flet loop from any thread."""
        return self.dispatcher.submit(coro_fn, *args)

    def spawn(self, coro) -> asyncio.Future:
        return self.dispatcher.spawn(coro)

    @property
    def _tasks(self) -> set:
        return self.dispatcher._tasks

    async def _in_thread(self, fn: Callable[..., Any], *args: Any, name: str = "gl-spike-worker") -> Any:
        """Run blocking ``fn`` on a fresh thread (16 MiB stack) and await its result."""
        return await self.dispatcher.run_in_thread(fn, *args, name=name)

    def _card(self, key: str) -> CheckCard:
        return self.cards[key]

    # ---- build ----------------------------------------------------------------

    async def start(self) -> None:
        page = self.page
        page.title = "Glossarion"
        page.theme_mode = ft.ThemeMode.SYSTEM
        page.theme = build_theme()
        page.dark_theme = build_theme()
        page.padding = 0
        if not self.is_mobile and not page.web and page.platform is not None:
            try:  # phone-sized window for desktop dev
                page.window.width = 412
                page.window.height = 860
            except Exception:
                pass
        page.on_route_change = self.on_route_change
        page.on_app_lifecycle_state_change = self.on_lifecycle

        page.appbar = ft.AppBar(title=self._title(), center_title=False)
        page.add(self._cards_list())
        self._refresh_info()
        page.update()

        rb.set_url_opener(self.opener.open_threadsafe)
        self.initial_route = page.route
        rb.emit_marker(rb.MARKER_READY)

        rb.start_warm_import(on_done=lambda result: self.post(self._on_backend_ready, result))
        self.spawn(self._after_ready())
        await self._dispatch_route(page.route, source="initial")

    def _title(self) -> ft.Control:
        version = (self.state.version if self.state else {}) or {}
        subtitle = f"Device spike · v{version.get('version') or '?'} · {self.paths.platform if self.paths else '?'}"
        return ft.Column(
            tight=True,
            spacing=0,
            controls=[
                ft.Text("Device checks" if self.embedded else "Glossarion", weight=ft.FontWeight.BOLD, size=18),
                ft.Text(subtitle, size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            ],
        )

    def _cards_list(self) -> ft.Control:
        return ft.SafeArea(
            expand=True,
            content=ft.ListView(
                expand=True,
                padding=ft.Padding.symmetric(horizontal=12, vertical=8),
                spacing=8,
                controls=[card.control for card in self.cards.values()],
            ),
        )

    def build_view(self) -> ft.View:
        """The embedded View (pushed by the app; the implied back arrow pops it)."""
        self._refresh_info()
        self._refresh_routes()
        return ft.View(
            route=DEVICE_CHECKS_VIEW_ROUTE,
            appbar=ft.AppBar(title=self._title(), center_title=False),
            padding=0,
            controls=[self._cards_list()],
        )

    async def after_open(self) -> None:
        """Embedded: native probes once the View is on screen."""
        await self._after_ready()

    def _build_cards(self) -> None:
        cards = [
            CheckCard(
                "info",
                "Runtime",
                "Bootstrap result, storage paths, backend warm-import, native platform info.",
                [("Refresh", self.refresh_info)],
            ),
            CheckCard(
                "selftest",
                "Self-test (smoke)",
                "Imports, offline tiktoken, EPUB/lxml, Fernet, openai/pydantic/jiter, PyMuPDF, cv2, "
                f"onnxruntime, env contract. Deep link: glossarion://app{SELFTEST_ROUTE}?suite=smoke",
                [("Run self-test", self.run_selftest_clicked)],
            ),
            CheckCard(
                "secure",
                "SecureStorage round-trip",
                "set / get / remove a random value in Keystore (Android) / Keychain (iOS).",
                [("Run", self.secure_storage_test)],
            ),
            CheckCard(
                "fgs",
                "Foreground service (25 min)",
                "Starts the dataSync service and a Python thread that ticks every second into the "
                "notification for 25 minutes. Turn the screen off; the notification Stop button must "
                "reach Python. 'max gap' > 5 s means the thread was frozen.",
                [
                    ("Start", self.fgs_start),
                    ("Stop", self.fgs_stop),
                    ("Battery: ignore optimizations", self.request_battery_exemption),
                ],
            ),
            CheckCard(
                "notify",
                "Notification",
                "Requests POST_NOTIFICATIONS, creates the channels and shows a notification; tap it.",
                [("Show notification", self.notify_test)],
            ),
            CheckCard(
                "share",
                "Open-with / Share",
                "Open an EPUB/PDF/TXT with Glossarion or share to it. Items must arrive as share "
                "events and page.route must NOT change.",
                [("Re-read initial items", self.reread_shared)],
            ),
            CheckCard(
                "oauth",
                "OAuth loopback + return link",
                "127.0.0.1 server opened through webbrowser.open -> in-app browser; the page "
                "redirects to glossarion://app/oauth/return and the app records the return.",
                [("Start", self.oauth_test)],
            ),
            CheckCard(
                "stack",
                "16 MiB thread stacks",
                "Deep C recursion (nested repr/json) and 50k-deep Python recursion on a worker thread.",
                [("Run", self.stack_test)],
            ),
            CheckCard(
                "background",
                "Background task / continued processing (iOS)",
                "beginBackgroundTask + remaining time; iOS 26 BGContinuedProcessingTask counting 60 s.",
                [
                    ("Begin bg task", self.bg_begin),
                    ("End bg task", self.bg_end),
                    ("Continued processing", self.cp_start),
                ],
            ),
            CheckCard("routes", "Routes seen", "Every page.route change and whether it was whitelisted.", []),
            CheckCard("lifecycle", "Lifecycle", "App lifecycle states (logs are flushed on INACTIVE/HIDE/PAUSE).", []),
        ]
        self.cards = {card.key: card for card in cards}

    # ---- runtime info ---------------------------------------------------------

    def _info_text(self, platform_info: Optional[dict] = None) -> str:
        state = self.state
        if state is None:
            return "bootstrap() did not run"
        paths = state.paths
        backend = state.backend_result
        if backend is None:
            backend_text = "warming up…"
        elif backend.get("ok"):
            backend_text = f"ready: {backend.get('modules')} modules in {backend.get('secs')} s"
        else:
            backend_text = f"FAILED: {_json(backend.get('failed'), 300)}"
        lines = [
            f"version {state.version.get('version')} build {state.version.get('build')} "
            f"bundle {state.version.get('bundle')}",
            f"platform {paths.platform} (page.platform={getattr(self.page.platform, 'value', None)}), "
            f"python {sys.version.split()[0]}",
            f"backend {paths.backend_source}: {paths.backend_dir}",
            f"backend import: {backend_text}",
            f"data {paths.data}",
            f"docs {paths.docs}",
            f"boot {state.secs}s · previous crash {state.previous_crash} · tiktoken {_json(state.tiktoken, 200)}",
        ]
        if state.errors:
            lines.append(f"boot errors: {_json(state.errors, 300)}")
        if platform_info is not None:
            lines.append(f"native ({'stub' if self.native.is_stub else 'extension'}): {_json(platform_info, 400)}")
        return "\n".join(lines)

    def _refresh_info(self, platform_info: Optional[dict] = None) -> None:
        state = self.state
        ok = state is not None and not state.errors and state.paths.backend_dir is not None
        backend = state.backend_result if state else None
        if backend is not None and not backend.get("ok"):
            ok = False
        self._card("info").set("pass" if ok else ("fail" if state is None else "info"), self._info_text(platform_info))

    async def refresh_info(self, e: Any = None) -> None:
        info = await self.native.platform_info()
        self._refresh_info(info)

    def _on_backend_ready(self, result: dict) -> None:
        self._refresh_info()

    async def _after_ready(self) -> None:
        """Native probes that must not delay GLOSSARION_READY."""
        try:
            info = await self.native.platform_info()
            self._refresh_info(info)
        except Exception as exc:
            log.warning("platform info failed: %s", exc)
        await self.reread_shared()
        try:
            launch = await self.native.launch_notification()
            if launch:
                self._card("notify").set("pass", f"app launched from a notification: {_json(launch, 400)}")
        except Exception as exc:
            log.warning("launch notification failed: %s", exc)

    # ---- routes -----------------------------------------------------------------

    async def on_route_change(self, e: ft.RouteChangeEvent) -> None:
        await self._dispatch_route(e.route, source="event")

    async def _dispatch_route(self, raw: Optional[str], *, source: str) -> Optional[RouteMatch]:
        match = self.router.handle(raw)
        self._refresh_routes()
        if match is None:
            log.info("ignored route %r (%s)", raw, source)
            return None
        if match.path == SELFTEST_ROUTE:
            self.spawn(self.run_selftest(match.get("suite") or "smoke", source=f"route/{source}"))
            self.spawn(self._reset_route())
        elif match.path == "/oauth/return":
            self._on_oauth_return(match)
            self.spawn(self._reset_route())
        return match

    async def _reset_route(self) -> None:
        # Flet drops a route event equal to the last one; go back to "/" so the
        # same deep link (e.g. the self-test) can be fired again.
        if self.page.route != "/":
            try:
                await asyncio.wait_for(self.page.push_route("/"), 10)
            except Exception as exc:
                log.info("push_route('/') failed: %s", exc)

    def _refresh_routes(self) -> None:
        history = self.router.history[-10:]
        if not history:
            return
        rejected = [r for r in history if not r.accepted]
        lines = [
            f"{time.strftime('%H:%M:%S', time.localtime(r.at))} {'OK ' if r.accepted else 'IGN'} {r.raw!r} ({r.reason})"
            for r in history
        ]
        self._card("routes").set("info" if not rejected else "fail", "\n".join(lines))

    # ---- lifecycle ----------------------------------------------------------------

    async def on_lifecycle(self, e: Any) -> None:
        state = getattr(e, "state", None)
        name = getattr(state, "value", str(state))
        if name in ("inactive", "hide", "pause", "detach") and not self.embedded:
            rb.flush_logs()  # embedded: the app already flushed
        self.lifecycle.append(f"{time.strftime('%H:%M:%S')} {name}")
        del self.lifecycle[:-12]
        log.info("lifecycle: %s", name)
        self._card("lifecycle").set("info", "\n".join(self.lifecycle))

    # ---- self-test -------------------------------------------------------------------

    async def run_selftest_clicked(self, e: Any = None) -> None:
        await self.run_selftest("smoke", source="button")

    async def run_selftest(self, suite: str = "smoke", *, source: str = "button") -> Optional[dict]:
        from glossarion_mobile.diagnostics import selftest

        card = self._card("selftest")
        if self._selftest_running:
            card.set("running", f"Already running; ignored request from {source}.")
            return None
        self._selftest_running = True
        card.set("running", f"Running suite '{suite}' ({source})…")
        try:
            result = await self._in_thread(selftest.run_selftest, suite, name="gl-selftest")
        except Exception as exc:
            card.set("fail", f"self-test crashed: {type(exc).__name__}: {exc}")
            return None
        finally:
            self._selftest_running = False
        lines = [
            f"{result['passed']} passed · {result['failed']} failed · {result['skipped']} skipped · "
            f"{result['secs']} s · strict={result['strict']}"
        ]
        if result.get("error"):
            lines.append(result["error"])
        for check in result.get("checks", []):
            note = check.get("error") or check.get("reason") or ""
            lines.append(f"{check['status'].upper():4} {check['name']} ({check['secs']}s) {note[:160]}")
        card.set("pass" if result.get("ok") else "fail", "\n".join(lines))
        return result

    # ---- SecureStorage --------------------------------------------------------------

    async def secure_storage_test(self, e: Any = None) -> None:
        card = self._card("secure")
        card.set("running", "Writing…")
        try:
            if self._secure_storage is None:
                from flet_secure_storage import SecureStorage

                self._secure_storage = SecureStorage()
            storage = self._secure_storage
            key = "glossarion.spike.roundtrip"
            value = secrets.token_urlsafe(24)
            t0 = time.monotonic()
            await asyncio.wait_for(storage.set(key, value), 20)
            got = await asyncio.wait_for(storage.get(key), 20)
            await asyncio.wait_for(storage.remove(key), 20)
            gone = await asyncio.wait_for(storage.get(key), 20)
            try:
                available = await asyncio.wait_for(storage.get_availability(), 10)
            except Exception as exc:
                available = f"n/a ({type(exc).__name__})"
            ms = round((time.monotonic() - t0) * 1000)
            ok = got == value and gone is None
            card.set(
                "pass" if ok else "fail",
                f"set/get {'matched' if got == value else 'MISMATCH'}, removed={gone is None}, "
                f"availability={available}, {ms} ms",
            )
        except Exception as exc:
            card.set("fail", f"{type(exc).__name__}: {exc}")

    # ---- permissions ----------------------------------------------------------------

    async def _request_permission(self, name: str) -> str:
        if not self.is_mobile:
            return "n/a (desktop)"
        try:
            from flet_permission_handler import Permission, PermissionHandler

            if self._permissions is None:
                self._permissions = PermissionHandler()
            status = await asyncio.wait_for(self._permissions.request(getattr(Permission, name)), 120)
            return str(getattr(status, "value", status))
        except Exception as exc:
            return f"error {type(exc).__name__}: {exc}"

    async def request_battery_exemption(self, e: Any = None) -> None:
        status = await self._request_permission("IGNORE_BATTERY_OPTIMIZATIONS")
        card = self._card("fgs")
        card.set(card.status if card.status != "idle" else "info", f"ignore battery optimizations: {status}\n{card.result.value}")

    # ---- foreground service ----------------------------------------------------------

    async def fgs_start(self, e: Any = None) -> None:
        card = self._card("fgs")
        if self._fgs_thread is not None and self._fgs_thread.is_alive():
            card.set("running", "Already running. " + (card.result.value or ""))
            return
        permission = await self._request_permission("NOTIFICATION")
        started = await self.native.start_job_service("Glossarion spike", "Starting the 25-minute ticker…")
        self._fgs_stop.clear()
        self._fgs_stop_reason = None
        self._fgs_thread = threading.Thread(
            target=self._fgs_worker, args=(bool(started), permission), name="gl-spike-fgs", daemon=True
        )
        self._fgs_thread.start()
        card.set("running", f"service started={started} (notification permission: {permission}); ticking…")
        rb.emit_marker(MARKER_SPIKE, f"fgs start service={started}")

    async def fgs_stop(self, e: Any = None) -> None:
        self._request_fgs_stop("Stop button in the app")

    def _request_fgs_stop(self, reason: str) -> None:
        if self._fgs_stop_reason is None:
            self._fgs_stop_reason = reason
        self._fgs_stop.set()

    def _fgs_worker(self, service_started: bool, permission: str) -> None:
        total = FGS_DURATION_S
        t0 = time.time()
        last = t0
        max_gap = 0.0
        long_gaps = 0
        tick = 0
        while tick < total:
            if self._fgs_stop.wait(1.0):
                break
            tick += 1
            now = time.time()  # wall clock: keeps counting through CPU suspend
            gap = now - last
            last = now
            max_gap = max(max_gap, gap)
            if gap > 5.0:
                long_gaps += 1
            text = f"Tick {tick}/{total} · {_fmt_secs(now - t0)} · max gap {max_gap:.1f}s"
            self.submit(self.native.update_job_service, text)
            self.post(self._card("fgs").set, "running", f"{text}\nservice={service_started} · long gaps={long_gaps}")
            if tick % 60 == 0:
                rb.emit_marker(MARKER_SPIKE, f"fgs tick={tick} elapsed={now - t0:.0f} max_gap={max_gap:.1f} long_gaps={long_gaps}")
        reason = self._fgs_stop_reason or ("completed" if tick >= total else "stopped")
        summary = {
            "reason": reason,
            "ticks": tick,
            "elapsed": round(time.time() - t0, 1),
            "max_gap": round(max_gap, 1),
            "long_gaps": long_gaps,
            "service": service_started,
            "permission": permission,
        }
        rb.emit_marker(MARKER_SPIKE, {"fgs": summary})
        self.submit(self._fgs_finished, summary)

    async def _fgs_finished(self, summary: dict) -> None:
        try:
            await self.native.stop_job_service()
        except Exception as exc:
            summary["stop_error"] = str(exc)
        reason = summary["reason"]
        alive = summary["long_gaps"] == 0
        if reason == "completed":
            status = "pass" if alive else "fail"
        elif reason.startswith("notification"):
            status = "pass"
        else:
            status = "info"
        self._card("fgs").set(status, f"Finished: {_json(summary, 600)}")

    def _on_foreground(self, event: dict) -> None:
        """Notification buttons / service lifecycle from the FGS task handler."""
        log.info("foreground event: %s", event)
        kind = str(event.get("type") or "")
        button = str(event.get("button_id") or "")
        if kind == "button" and button == "stop":
            self._request_fgs_stop("notification Stop button")
        elif kind == "timeout" or (kind == "destroyed" and event.get("is_timeout")):
            self._request_fgs_stop("service timeout (Android dataSync budget)")
        elif kind == "destroyed" and self._fgs_thread is not None and self._fgs_thread.is_alive():
            self._request_fgs_stop("service destroyed by the system")
        rb.emit_marker(MARKER_SPIKE, {"foreground_event": event})
        card = self._card("fgs")
        card.set(card.status, f"{card.result.value}\nevent: {_json(event, 300)}")

    # ---- notifications -------------------------------------------------------------

    async def notify_test(self, e: Any = None) -> None:
        card = self._card("notify")
        permission = await self._request_permission("NOTIFICATION")
        try:
            initialised = await self.native.init_notifications()
            await self.native.show_notification(
                4242,
                "Glossarion spike",
                f"Test notification at {time.strftime('%H:%M:%S')}. Tap me.",
                channel_id="jobs.action",
                payload="/?spike=notification",
            )
        except Exception as exc:
            card.set("fail", f"{type(exc).__name__}: {exc} (permission {permission})")
            return
        status = "n/a" if self.native.is_stub else "info"
        card.set(status, f"permission={permission} init={initialised}; shown. Tap it to test on_notification.")

    def _on_notification(self, event: dict) -> None:
        rb.emit_marker(MARKER_SPIKE, {"notification_event": event})
        self._card("notify").set("pass", f"notification event: {_json(event, 500)}")

    # ---- open-with / share ------------------------------------------------------------

    async def reread_shared(self, e: Any = None) -> None:
        card = self._card("share")
        try:
            items = await self.native.initial_shared()
        except Exception as exc:
            card.set("fail", f"get_initial_shared failed: {type(exc).__name__}: {exc}")
            return
        self.initial_shared = items
        if not self.embedded:  # the app routes launch links itself
            await self._route_launch_links(items)
        route_ok = (self.initial_route or "/").split("?", 1)[0] in ("/", SELFTEST_ROUTE, "/oauth/return")
        if not items:
            card.set("n/a" if self.native.is_stub else "idle", f"No initial items. Initial route {self.initial_route!r}.")
            return
        card.set(
            "pass" if route_ok else "fail",
            f"{len(items)} initial item(s): {_json(items, 600)}\ninitial page.route={self.initial_route!r} "
            f"({'unchanged' if route_ok else 'CHANGED by Open-with'})",
        )

    @staticmethod
    def _launch_links(items: Any) -> list[tuple[str, str]]:
        """(id, glossarion:// URI) of iOS cold-start deep links delivered as shared items."""
        return launch_links(items)

    async def _route_launch_links(self, items: Any) -> bool:
        routed = False
        for item_id, link in self._launch_links(items):
            if item_id in self._routed_link_ids:
                continue
            self._routed_link_ids.add(item_id)
            # De-duplicate against page.route: Flutter may have delivered it too.
            initial = parse_route(self.initial_route) if self.initial_route else None
            link_match = parse_route(link)
            if initial is not None and link_match is not None and initial.path == link_match.path != "/":
                continue
            await self._dispatch_route(link, source="launch-link")
            routed = True
        return routed

    def _on_share(self, event: dict) -> None:
        since = time.time()
        route_before = self.page.route
        rb.emit_marker(MARKER_SPIKE, {"share_event": event})
        if self._launch_links(event.get("items")):
            if not self.embedded:  # the app routes launch links itself
                self.spawn(self._route_launch_links(event.get("items")))
            return
        self._card("share").set("running", f"share event: {_json(event, 600)}\nchecking page.route…")
        self.spawn(self._verify_route_untouched(since, route_before, event))

    async def _verify_route_untouched(self, since: float, route_before: str, event: dict) -> None:
        await asyncio.sleep(2.5)
        seen = self.router.seen_since(since)
        changed = self.page.route != route_before
        ok = not changed and not seen
        detail = f"page.route {route_before!r} -> {self.page.route!r}; route events since share: {[r.raw for r in seen]}"
        self._card("share").set("pass" if ok else "fail", f"share event: {_json(event, 600)}\n{detail}")

    # ---- OAuth loopback -------------------------------------------------------------

    def _open_url_threadsafe(self, url: str) -> None:
        """webbrowser.open() target (any thread) -> UrlLauncher on the loop."""
        self.opener.open_threadsafe(url)

    async def _launch_url(self, url: str) -> None:
        await self.opener.launch(url)

    def _close_oauth(self) -> None:
        if self._oauth is not None:
            self._oauth["server"].close()
            self._oauth = None

    async def oauth_test(self, e: Any = None) -> None:
        card = self._card("oauth")
        self._close_oauth()
        nonce = secrets.token_hex(6)
        server = LoopbackServer(nonce, on_hit=lambda path: self.post(self._oauth_hit, nonce, path))
        url = f"http://127.0.0.1:{server.port}/start?nonce={nonce}"
        self._oauth = {"nonce": nonce, "server": server, "t0": time.time(), "hit": None}
        card.set("running", f"Loopback on 127.0.0.1:{server.port}; webbrowser.open -> in-app browser…")
        # Same path the unchanged desktop OAuth flows take: webbrowser.open on a worker thread.
        opened = await self._in_thread(webbrowser.open, url, name="gl-spike-oauth")
        if not opened:
            card.set("fail", "webbrowser.open returned False (controller not installed?)")
            return
        self.spawn(self._oauth_timeout(nonce))

    def _oauth_hit(self, nonce: str, path: str) -> None:
        if self._oauth is not None and self._oauth["nonce"] == nonce:
            self._oauth["hit"] = time.time()
            elapsed = self._oauth["hit"] - self._oauth["t0"]
            self._card("oauth").set("running", f"Loopback hit after {elapsed:.1f} s ({path}); waiting for the return link…")

    def _on_oauth_return(self, match: RouteMatch) -> None:
        card = self._card("oauth")
        pending = self._oauth
        nonce = match.get("nonce")
        if pending is None or nonce != pending["nonce"]:
            card.set("info", f"Return link without a matching pending test: {match.query}")
            return
        now = time.time()
        hit = pending["hit"]
        text = (
            f"Return received {now - pending['t0']:.1f} s after open "
            f"(loopback hit: {'yes' if hit else 'NO'}; provider p={match.get('p')!r})."
        )
        card.set("pass" if hit else "fail", text)
        rb.emit_marker(MARKER_SPIKE, f"oauth return ok={bool(hit)}")
        self._close_oauth()
        self.spawn(self._close_in_app_browser())

    async def _close_in_app_browser(self) -> None:
        # SFSafariViewController on iOS; Android Custom Tabs cannot be closed (no-op)
        await self.opener.close_in_app_view()

    async def _oauth_timeout(self, nonce: str) -> None:
        await asyncio.sleep(OAUTH_TIMEOUT_S)
        if self._oauth is not None and self._oauth["nonce"] == nonce:
            hit = self._oauth["hit"]
            self._close_oauth()
            self._card("oauth").set("fail", f"No return link within {OAUTH_TIMEOUT_S} s (loopback hit: {bool(hit)}).")

    # ---- thread stack ---------------------------------------------------------------

    async def stack_test(self, e: Any = None) -> None:
        card = self._card("stack")
        card.set("running", "Recursing on a worker thread…")
        try:
            result = await self._in_thread(stack_probe, name="gl-spike-stack")
        except Exception as exc:
            card.set("fail", f"{type(exc).__name__}: {exc}")
            return
        ok = result.get("thread_stack_size", 0) >= rb.THREAD_STACK_SIZE and result.get("c_depth_ok", 0) >= 1000
        card.set("pass" if ok else "fail", _json(result, 800))

    # ---- iOS background task / continued processing ------------------------------------

    async def bg_begin(self, e: Any = None) -> None:
        card = self._card("background")
        try:
            self._bg_task_id = await self.native.begin_background_task("glossarion-spike")
            if self._bg_task_id < 0:
                card.set("n/a", "begin_background_task returned -1 (not iOS).")
                return
            remaining = await self.native.background_time_remaining()
        except Exception as exc:
            card.set("fail", f"{type(exc).__name__}: {exc}")
            return
        status = "n/a" if self.native.is_stub else "info"
        card.set(status, f"background task id={self._bg_task_id!r}, time remaining={remaining!r}")

    async def bg_end(self, e: Any = None) -> None:
        card = self._card("background")
        if self._bg_task_id < 0:
            card.set(card.status, "No background task to end.")
            return
        try:
            await self.native.end_background_task(self._bg_task_id)
        except Exception as exc:
            card.set("fail", f"{type(exc).__name__}: {exc}")
            return
        card.set("info", f"ended background task {self._bg_task_id!r}")
        self._bg_task_id = -1

    async def cp_start(self, e: Any = None) -> None:
        card = self._card("background")
        if self._cp_thread is not None and self._cp_thread.is_alive():
            card.set("running", "Continued processing already running.")
            return
        identifier = f"{JOB_TASK_ID_PREFIX}spike-{int(time.time())}"
        try:
            started = await self.native.start_continued_processing(identifier, "Glossarion spike", "Counting to 60…")
        except Exception as exc:
            card.set("fail", f"{type(exc).__name__}: {exc}")
            return
        if not started:
            card.set("n/a", f"start_continued_processing({identifier!r}) returned False (needs iOS 26).")
            return
        self._cp_stop.clear()
        self._cp_thread = threading.Thread(target=self._cp_worker, args=(identifier,), name="gl-spike-cp", daemon=True)
        self._cp_thread.start()
        card.set("running", f"BGContinuedProcessingTask {identifier} submitted.")

    def _cp_worker(self, identifier: str) -> None:
        done = 0
        for done in range(1, CONTINUED_PROCESSING_S + 1):
            if self._cp_stop.wait(1.0):
                break
            self.submit(self.native.update_continued_processing, done, CONTINUED_PROCESSING_S, f"{done}/{CONTINUED_PROCESSING_S}")
            self.post(self._card("background").set, "running", f"{identifier}: {done}/{CONTINUED_PROCESSING_S}")
        success = done >= CONTINUED_PROCESSING_S and not self._cp_stop.is_set()
        self.submit(self.native.finish_continued_processing, success)
        self.post(self._card("background").set, "pass" if success else "info", f"{identifier}: finished success={success} at {done}")

    def _on_background_task(self, event: dict) -> None:
        kind = str(event.get("type") or "")
        if kind in ("continued_expired", "continued_failed"):
            self._cp_stop.set()
        if kind == "expiring" and event.get("task_id") == self._bg_task_id:
            self._bg_task_id = -1  # the native side already ended it
        rb.emit_marker(MARKER_SPIKE, {"background_task_event": event})
        card = self._card("background")
        card.set("info", f"{card.result.value}\nevent: {_json(event, 300)}")


async def main(page: ft.Page) -> None:
    app = SpikeApp(page)
    page.data = app  # keep the app (and its services) strongly referenced
    await app.start()
