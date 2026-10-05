"""Env preview (``/settings/logs/env``): the environment the next translation run would get.

``build_env_preview`` runs on a worker thread while holding
``ENV_PREVIEW_LOCK`` (U3 replaces it with ``job_runner.JOB_LOCK``, so a
preview never overlaps a job). It builds a ``HeadlessOwner`` from a
``MobileConfigStore.snapshot()`` (including edits not saved yet) and calls
``run_env.build_translation_env(owner, input_path, api_key)``, the same
builders the desktop uses. The owner's init writes process-global
environment variables, so ``os.environ`` (and the ``large_env`` overflow
store, ``sys.argv`` and the working directory) are snapshotted before and
restored afterwards, key by key (the environment is never cleared, so other
threads keep seeing every unchanged variable). The run builder applies the
snapshot's key pools to ``UnifiedClient``; the preview gets fresh pool objects
and the previous pool state is put back afterwards.

Values are redacted before they reach the UI: names ending in KEY, TOKEN,
SECRET, PASSWORD, COOKIE, AUTHORIZATION or BEARER or containing API_KEY /
ACCESS_TOKEN / REFRESH_TOKEN (the desktop ``debug_env_vars._mask_value`` rule,
narrowed so GLOSSARY_DUPLICATE_KEY_MODE or MAX_OUTPUT_TOKENS stay readable),
values that look like API keys or tokens, JSON carrying ``api_key``/``token``
fields, and anything containing a secret from the config snapshot. Trivial
values ("", "0", "1", "true", numbers) stay visible so flags remain
diagnosable. Redacted rows show ``<REDACTED> (N chars)`` like the desktop
check. The screen lists the result as a searchable key/value list with a
redacted "Copy" export.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
import os
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

__all__ = [
    "ENV_PREVIEW_LOCK",
    "EnvPreviewResult",
    "EnvPreviewScreen",
    "EnvRow",
    "INPUT_KINDS",
    "build_env_preview",
    "config_secrets",
    "filter_rows",
    "isolated_key_pools",
    "preview_input_path",
    "redact_env",
    "restore_mapping",
    "rows_as_text",
    "scoped_process_env",
]

log = logging.getLogger("glossarion.env_preview")

# One HeadlessOwner at a time (its init mutates os.environ). U3: job_runner.JOB_LOCK.
ENV_PREVIEW_LOCK = threading.Lock()

INPUT_KINDS = (("epub", "EPUB"), ("txt", "TXT"), ("pdf", "PDF"))

# A name is secret when its last component is one of these (``OPENAI_API_KEY``, ``AUTH_TOKEN``),
# or it contains API_KEY / ACCESS_TOKEN / REFRESH_TOKEN anywhere (``MULTI_API_KEYS``). Names such
# as GLOSSARY_DUPLICATE_KEY_MODE or MAX_OUTPUT_TOKENS stay visible; their values are still checked.
_SECRET_NAME_PARTS = frozenset({"KEY", "KEYS", "TOKEN", "SECRET", "SECRETS", "PASSWORD", "PASSWD", "COOKIE",
                                "COOKIES", "AUTHORIZATION", "BEARER"})
_SECRET_NAME_FRAGMENTS = ("API_KEY", "APIKEY", "ACCESS_TOKEN", "REFRESH_TOKEN", "ID_TOKEN", "PRIVATE_KEY")
_TRIVIAL_VALUES = frozenset({"", "0", "1", "true", "false", "none", "null", "[]", "{}", "yes", "no", "on", "off"})
_SECRET_VALUE_PATTERNS = (
    re.compile(r"^ENC:"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"\bsk_[A-Za-z0-9]{16,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\b(?:xai|gsk|r8|hf|ghp|glpat)[-_][A-Za-z0-9_\-]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r'(?i)"(?:api_?key|access_token|refresh_token|id_token|token|secret|password|cookie)"\s*:\s*"[^"]{4,}"'),
)
_DISPLAY_LIMIT = 400
_MIN_SECRET_LEN = 6


@dataclass(frozen=True)
class EnvRow:
    key: str
    value: str  # display value (already redacted / shortened)
    redacted: bool
    length: int  # length of the real value


@dataclass
class EnvPreviewResult:
    ok: bool
    rows: list[EnvRow] = field(default_factory=list)
    error: str = ""
    secs: float = 0.0
    input_path: str = ""
    logs: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.rows)


# ---- redaction ------------------------------------------------------------------------------


def config_secrets(config: Any) -> set[str]:
    """Secret strings in a config snapshot (``api_key`` and the key-pool ``api_key`` fields)."""
    found: set[str] = set()

    def add(value: Any) -> None:
        if isinstance(value, str) and len(value.strip()) >= _MIN_SECRET_LEN:
            found.add(value.strip())

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                name = str(key).lower()
                if isinstance(value, str) and any(t.lower() in name for t in ("api_key", "token", "secret", "password", "cookie")):
                    add(value)
                else:
                    walk(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item, depth + 1)

    walk(config)
    return found


def _is_trivial(value: str) -> bool:
    text = value.strip().lower()
    if text in _TRIVIAL_VALUES:
        return True
    try:
        float(text)
        return len(text) <= 24
    except ValueError:
        return False


def _secret_name(name: str) -> bool:
    upper = name.upper()
    if any(fragment in upper for fragment in _SECRET_NAME_FRAGMENTS):
        return True
    parts = [part for part in re.split(r"[^A-Z0-9]+", upper) if part]
    return bool(parts) and parts[-1] in _SECRET_NAME_PARTS


def _needs_redaction(name: str, value: str, secrets: Iterable[str]) -> bool:
    if _is_trivial(value):
        return False
    if _secret_name(name):
        return True
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        return True
    return any(secret and secret in value for secret in secrets)


def redact_env(env: Any, secrets: Iterable[str] = ()) -> list[EnvRow]:
    """Sorted, redacted rows for ``env`` (``"<REDACTED> (N chars)"`` like the desktop env check)."""
    secrets = [s for s in secrets if isinstance(s, str) and len(s) >= _MIN_SECRET_LEN]
    rows: list[EnvRow] = []
    for name in sorted((env or {}).keys(), key=lambda k: str(k).upper()):
        raw = env[name]
        value = "" if raw is None else str(raw)
        if _needs_redaction(str(name), value, secrets):
            rows.append(EnvRow(str(name), f"<REDACTED> ({len(value)} chars)", True, len(value)))
            continue
        shown = value if len(value) <= _DISPLAY_LIMIT else value[:_DISPLAY_LIMIT] + f"… ({len(value):,} chars)"
        rows.append(EnvRow(str(name), shown, False, len(value)))
    return rows


def filter_rows(rows: Iterable[EnvRow], query: str) -> list[EnvRow]:
    needle = (query or "").strip().casefold()
    if not needle:
        return list(rows)
    return [r for r in rows if needle in r.key.casefold() or (not r.redacted and needle in r.value.casefold())]


def rows_as_text(rows: Iterable[EnvRow]) -> str:
    return "\n".join(f"{r.key}={r.value}" for r in rows)


# ---- building -----------------------------------------------------------------------------------


def restore_mapping(target: Any, saved: dict) -> None:
    """Put ``target`` back to ``saved`` key by key: drop added keys, reset changed ones.

    Never clears the mapping, so keys that did not change (``GLOSSARION_*``,
    ``CONFIG_FILE``, ...) stay readable by other threads the whole time; the
    ``mobile_runtime`` gates read ``os.environ`` live."""
    for key in [k for k in list(target.keys()) if k not in saved]:
        target.pop(key, None)
    for key, value in saved.items():
        if target.get(key) != value:
            target[key] = value


#: UnifiedClient class attributes that ``key_pools.apply_key_pools_to_runtime`` writes through the
#: ``set_/clear_in_memory_*`` and ``setup_*_key_pool`` class methods (beyond the ``_in_memory_*``
#: lists, ``*_key_pool`` objects and ``*_pool_logged`` / ``_last_*_pool_setup_status`` flags).
_POOL_EXTRA_ATTRS = ("_force_rotation", "_rotation_frequency", "_rate_limit_cache")


def _is_pool_state_attr(name: str) -> bool:
    if name.endswith("_lock"):
        return False
    return (name.startswith("_in_memory_") or name.endswith(("_key_pool", "_pool_logged", "_pool_setup_status"))
            or name in _POOL_EXTRA_ATTRS)


@contextlib.contextmanager
def isolated_key_pools() -> Iterator[None]:
    """Give the body fresh ``UnifiedClient`` key pools and put the previous pool state back afterwards.

    ``run_env.build_translation_env`` applies the snapshot's key pools
    (``key_pools.apply_key_pools_to_runtime``): class attributes are replaced and the shared
    ``APIKeyPool`` objects are reloaded in place. The pool objects are detached first
    (``setup_*_key_pool`` builds a new pool when the attribute is None), so the previous pools
    are never mutated, and every pool attribute is restored on exit. Only acts when
    ``unified_api_client`` is already imported (``build_env_preview`` imports it first)."""
    module = sys.modules.get("unified_api_client")
    cls = getattr(module, "UnifiedClient", None) if module is not None else None
    saved = {n: v for n, v in vars(cls).items() if _is_pool_state_attr(n)} if cls is not None else {}
    if cls is not None:
        for name, value in saved.items():
            if name.endswith("_key_pool") and value is not None:
                setattr(cls, name, None)
    try:
        yield
    finally:
        if cls is not None:
            missing = object()
            for name in [n for n in list(vars(cls)) if _is_pool_state_attr(n) and n not in saved]:
                try:
                    delattr(cls, name)
                except AttributeError:
                    pass
            for name, value in saved.items():
                if vars(cls).get(name, missing) is not value:
                    setattr(cls, name, value)


@contextlib.contextmanager
def scoped_process_env() -> Iterator[None]:
    """Restore ``os.environ``, ``large_env``'s overflow store, ``sys.argv``, the cwd and the
    ``UnifiedClient`` key pools afterwards (environment restored key by key, never cleared)."""
    saved_env = dict(os.environ)
    saved_argv = list(sys.argv)
    try:
        saved_cwd: Optional[str] = os.getcwd()
    except OSError:
        saved_cwd = None
    large = sys.modules.get("large_env")
    store = getattr(large, "_store", None)
    saved_store = dict(store) if isinstance(store, dict) else None
    try:
        with isolated_key_pools():
            yield
    finally:
        restore_mapping(os.environ, saved_env)
        sys.argv[:] = saved_argv
        if saved_cwd is not None:
            try:
                if os.getcwd() != saved_cwd:
                    os.chdir(saved_cwd)
            except OSError:
                pass
        large = sys.modules.get("large_env")
        store = getattr(large, "_store", None)
        if isinstance(store, dict):
            restore_mapping(store, saved_store or {})


class _PreviewHost:
    """Minimal JobHost for a preview owner: collects log lines, never stops, never answers."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: Any = "", *args: Any, **kwargs: Any) -> None:
        self.lines.append(str(message))
        del self.lines[:-200]

    append_log = log

    def emit(self, kind: str, **data: Any) -> None:
        if kind == "log" and "text" in data:
            self.log(data["text"])

    def ask(self, *args: Any, **kwargs: Any) -> None:
        return None

    def is_stop_requested(self) -> bool:
        return False

    def is_graceful_stop(self) -> bool:
        return False


def _default_owner_factory(config: dict, *, host: Any, api_key: str) -> Any:
    from headless_owner import HeadlessOwner  # shared, GUI-free (U2)

    return HeadlessOwner(config, host=host, api_key=api_key)


def _default_env_builder(owner: Any, input_path: str, api_key: str) -> dict:
    import run_env  # shared, GUI-free (U2)

    return run_env.build_translation_env(owner, input_path, api_key)


def preview_input_path(kind: str = "epub", data_dir: Optional[str] = None) -> str:
    """Placeholder input path under ``<data>/Inbox`` (the file does not have to exist)."""
    base = data_dir or os.environ.get("GLOSSARION_DATA_DIR") or os.getcwd()
    ext = kind if kind in dict(INPUT_KINDS) else "epub"
    return os.path.join(base, "Inbox", f"env-preview.{ext}")


def build_env_preview(
    config: dict,
    *,
    input_path: str,
    api_key: Optional[str] = None,
    lock: Any = ENV_PREVIEW_LOCK,
    lock_timeout: float = 5.0,
    owner_factory: Optional[Callable[..., Any]] = None,
    env_builder: Optional[Callable[[Any, str, str], dict]] = None,
) -> EnvPreviewResult:
    """Blocking: build a HeadlessOwner from ``config`` and return the redacted translation env."""
    started = time.monotonic()
    if not lock.acquire(timeout=lock_timeout):
        return EnvPreviewResult(False, error="Busy: a job or another preview is using the engine. Try again when it finishes.",
                                input_path=input_path)
    host = _PreviewHost()
    try:
        snapshot = copy.deepcopy(config)
        key = api_key if api_key is not None else str(snapshot.get("api_key") or "")
        if env_builder is None:
            # The real builder imports it anyway (key_pools); importing it first lets
            # scoped_process_env snapshot and restore the UnifiedClient key pools.
            try:
                import unified_api_client  # noqa: F401
            except Exception:
                pass
        try:
            with scoped_process_env():
                owner = (owner_factory or _default_owner_factory)(snapshot, host=host, api_key=key)
                env = (env_builder or _default_env_builder)(owner, input_path, key)
        except ImportError as exc:
            return EnvPreviewResult(
                False, error=f"The shared HeadlessOwner / run_env modules are not in this build ({exc}).",
                secs=round(time.monotonic() - started, 2), input_path=input_path, logs=host.lines,
            )
        except Exception as exc:
            tail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            log.exception("env preview failed")
            return EnvPreviewResult(False, error=f"Building the run environment failed: {tail}",
                                    secs=round(time.monotonic() - started, 2), input_path=input_path, logs=host.lines)
        if not isinstance(env, dict):
            env = dict(env or {})
        secrets = config_secrets(snapshot) | ({key} if key else set())
        rows = redact_env(env, secrets)
        return EnvPreviewResult(True, rows=rows, secs=round(time.monotonic() - started, 2), input_path=input_path,
                                logs=host.lines)
    finally:
        lock.release()


# ---- screen -------------------------------------------------------------------------------------

try:  # the pure part above must stay importable without Flet (host tests, services)
    import flet as ft

    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.components.section_card import SectionCard
    from glossarion_mobile.ui.screens.base import Screen
    from glossarion_mobile.ui.theme import HIT_TARGET, mono_family
except ImportError:  # pragma: no cover - Flet missing
    ft = None  # type: ignore[assignment]
    Screen = object  # type: ignore[assignment,misc]


_ROW_LIMIT = 2000


class EnvPreviewScreen(Screen):  # type: ignore[misc,valid-type]
    title = "Env preview"

    def __init__(self, match: Any, *, store: Any, dispatcher: Any = None, page: Any = None,
                 copy_handler: Optional[Callable[[str], Any]] = None, data_dir: Optional[str] = None,
                 builder: Callable[..., EnvPreviewResult] = build_env_preview) -> None:
        super().__init__(match)
        self.store = store
        self.dispatcher = dispatcher
        self.page = page
        self.copy_handler = copy_handler
        self.data_dir = data_dir
        self.builder = builder
        self.kind = "epub"
        self.query = ""
        self.result: Optional[EnvPreviewResult] = None
        self.running = False
        self._unsubs: list[Callable[[], None]] = []

    # ---- body ---------------------------------------------------------------------------------

    def build_body(self) -> "ft.Control":
        self.kind_selector = ft.SegmentedButton(
            segments=[ft.Segment(value=k, label=ft.Text(label)) for k, label in INPUT_KINDS],
            selected=[self.kind],
            show_selected_icon=False,
            on_change=self._on_kind,
        )
        self.build_button = ft.FilledTonalButton(content="Build preview", icon=ft.Icons.PLAY_ARROW, on_click=self._on_build)
        self.progress = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.status = ft.Text("Not built yet. Uses a snapshot of your current settings, including unsaved edits.",
                              theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.copy_button = ft.IconButton(icon=ft.Icons.CONTENT_COPY, tooltip="Copy (redacted)", on_click=self._on_copy,
                                         disabled=True, size_constraints=HIT_TARGET)
        self.search_field = ft.TextField(
            color=ft.Colors.ON_SURFACE,
            hint_text="Filter variables", prefix_icon=ft.Icons.SEARCH, dense=True, filled=True,
            border=ft.NoInputBorder(), border_radius=28,
            content_padding=ft.Padding.symmetric(horizontal=12, vertical=10),
            on_change=lambda e: self.set_query(e.control.value or ""),
        )
        self.rows_view = ft.ListView(expand=True, spacing=0, build_controls_on_demand=True,
                                     padding=ft.Padding.symmetric(horizontal=4, vertical=4))
        header = SectionCard(
            title="Translation run environment",
            icon="DATA_OBJECT",
            subtitle="HeadlessOwner + run_env.build_translation_env on a worker thread; secrets redacted",
            children=[
                ft.Row([ft.Text("Input", theme_style=ft.TextThemeStyle.LABEL_MEDIUM, color=ft.Colors.ON_SURFACE),
                        self.kind_selector], spacing=8, wrap=True),
                ft.Row([self.build_button, self.progress, ft.Container(expand=True), self.copy_button], spacing=8,
                       vertical_alignment=ft.CrossAxisAlignment.CENTER),
                self.status,
            ],
            key="env-preview-header",
        )
        self._render_rows(push=False)
        return ft.Column(
            [
                ft.Container(padding=ft.Padding.only(left=12, right=12, top=8), content=header),
                ft.Container(padding=ft.Padding.symmetric(horizontal=12), content=self.search_field),
                self.rows_view,
            ],
            spacing=tokens.SPACING["sm"],
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def did_show(self) -> None:
        if not self._unsubs and hasattr(self.store, "observe_job"):
            self._unsubs.append(self.store.observe_job(lambda running: self._post(self._sync_enabled)))
        self._sync_enabled(push=False)

    def dispose(self) -> None:
        for unsub in self._unsubs:
            unsub()
        self._unsubs = []

    def _post(self, fn: Callable[..., Any], *args: Any) -> None:
        dispatcher = self.dispatcher
        if dispatcher is not None and getattr(dispatcher, "bound", False) and not dispatcher.on_loop_thread():
            dispatcher.post(fn, *args)
        else:
            fn(*args)

    @staticmethod
    def _push(*controls: Any) -> None:
        for control in controls:
            try:
                control.update()
            except Exception:
                pass

    def _sync_enabled(self, push: bool = True) -> None:
        job = bool(getattr(self.store, "job_running", False))
        self.build_button.disabled = self.running or job
        self.build_button.tooltip = "Wait for the running job to finish" if job else None
        if push:
            self._push(self.build_button)

    # ---- rows -------------------------------------------------------------------------------------

    def visible_rows(self) -> list[EnvRow]:
        if self.result is None or not self.result.ok:
            return []
        return filter_rows(self.result.rows, self.query)

    def _row_control(self, row: EnvRow) -> "ft.Control":
        mono = mono_family(self.page)
        return ft.ListTile(
            title=ft.Text(row.key, font_family=mono, size=12, weight=ft.FontWeight.W_600, selectable=True),
            subtitle=ft.Text(row.value, font_family=mono, size=12, selectable=True, max_lines=4,
                             overflow=ft.TextOverflow.ELLIPSIS,
                             color=ft.Colors.ON_SURFACE_VARIANT if row.redacted else None, italic=row.redacted),
            dense=True,
            key=f"env-{row.key}",
        )

    def _render_rows(self, push: bool = True) -> None:
        rows = self.visible_rows()
        controls = [self._row_control(row) for row in rows[:_ROW_LIMIT]]
        if self.result is not None and self.result.ok and not rows:
            controls = [ft.Text(f"No variables match “{self.query}”.", theme_style=ft.TextThemeStyle.BODY_SMALL)]
        self.rows_view.controls = controls
        if push:
            self._push(self.rows_view)

    def set_query(self, query: str) -> None:
        self.query = (query or "").strip()
        self._render_rows()

    def _on_kind(self, e: Any = None) -> None:
        selected = list(getattr(getattr(e, "control", None), "selected", None) or self.kind_selector.selected or [])
        if selected:
            self.kind = str(selected[0])

    # ---- build ---------------------------------------------------------------------------------------

    async def run_preview(self) -> Optional[EnvPreviewResult]:
        if self.running or getattr(self.store, "job_running", False):
            return None
        self.running = True
        self.progress.visible = True
        self.status.value = "Building…"
        self._sync_enabled()
        self._push(self.progress, self.status)
        snapshot = self.store.snapshot()
        input_path = preview_input_path(self.kind, self.data_dir)
        work = lambda: self.builder(snapshot, input_path=input_path)  # noqa: E731
        try:
            if self.dispatcher is not None and getattr(self.dispatcher, "bound", False):
                result = await self.dispatcher.run_in_thread(work, name="gl-env-preview")
            else:
                result = await asyncio.to_thread(work)
        except Exception as exc:  # builder bugs; build_env_preview itself never raises
            result = EnvPreviewResult(False, error=f"{type(exc).__name__}: {exc}", input_path=input_path)
        self.result = result
        self.running = False
        self.progress.visible = False
        if result.ok:
            redacted = sum(1 for r in result.rows if r.redacted)
            self.status.value = (f"{result.count} variables for {os.path.basename(result.input_path)} "
                                 f"in {result.secs} s · {redacted} redacted")
            self.status.color = None
        else:
            self.status.value = result.error
            self.status.color = ft.Colors.ERROR
        self.copy_button.disabled = not result.ok
        self._sync_enabled()
        self._render_rows(push=False)
        self._push(self.progress, self.status, self.copy_button, self.rows_view)
        return result

    async def _on_build(self, e: Any = None) -> Optional[EnvPreviewResult]:
        return await self.run_preview()

    async def _on_copy(self, e: Any = None) -> Optional[str]:
        rows = self.visible_rows()
        if not rows:
            return None
        text = rows_as_text(rows)
        if self.copy_handler is not None:
            result = self.copy_handler(text)
            if hasattr(result, "__await__"):
                await result
        return text
