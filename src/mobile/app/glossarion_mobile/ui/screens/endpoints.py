"""Endpoints (``/settings/endpoints``; UI_SPEC §4.15 "Models & keys › Endpoints").

The desktop "Custom API Endpoints" section of Other Settings, as schema tiles (the shared
settings tiles, so locks / defaults / reset / help behave like every other setting): custom
OpenAI-compatible endpoint (+ Azure API version, Gemma routing), Gemini custom endpoint
(OpenAI-compatible URL or a bare gRPC host, which follows the dependency rule), Anthropic custom
endpoint, custom image-edit endpoint, Groq/Local and Fireworks base URLs, the TTS voice, Vertex
AI credentials + location (dependency rule) and the Replicate key (SecretTile). URL fields carry
quick-paste chips; the Local AI page (LAN Ollama / LM Studio) is linked; the desktop-only rows
("AuthZA / GLM access mode", "Load Ollama") stay visible, disabled with a ReasonChip.

**Test connection** first probes every configured endpoint the way the desktop Other Settings ›
Test Connections does (``key_pool_service.collect_test_endpoints`` / ``run_endpoint_tests``, moved
out of ``other_settings.test_api_connections`` in U9: one ✅ / ❌ row per endpoint, the same simplified
errors), then the selected model through the shared run path: the environment the next translation run would
get (``HeadlessOwner`` + ``run_env.build_translation_env``, applied with ``large_env`` inside
``job_runner.scoped_process_state`` under ``JOB_LOCK``, as the Env preview does) and
``key_pool_service.run_key_test`` for the configured model and key, so the request goes through
exactly the endpoint settings a run would use. ``run_in_run_env`` is reused by the Multi-Key
Manager's key tests, one call (one engine-lock hold) per key and with every key pool switched
off in the config (``KeysController.single_key_config``): with the Translation pool on, the run
environment would make the probe client rotate through the pool instead of sending the key
under test. Test connection keeps the full run environment (pools included), like a run.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import logging
import os
from typing import Any, Callable, Mapping, Optional

__all__ = [
    "ENDPOINT_SECTIONS",
    "EndpointsScreen",
    "QUICK_PASTE",
    "endpoint_owner",
    "endpoint_summary",
    "run_endpoint_probes",
    "run_in_run_env",
    "test_connection",
]

log = logging.getLogger("glossarion.endpoints")

#: (title, keys) in the order of the desktop section.
ENDPOINT_SECTIONS = (
    ("Custom OpenAI endpoint", ("use_custom_openai_endpoint", "openai_base_url", "azure_api_version",
                                "override_gemma_for_custom_endpoint")),
    ("Gemini custom endpoint", ("use_gemini_openai_endpoint", "gemini_openai_endpoint")),
    ("Anthropic custom endpoint", ("force_native_anthropic", "anthropic_base_url")),
    ("Custom image edit endpoint", ("use_custom_image_edit_endpoint", "custom_image_edit_endpoint")),
    ("Provider base URLs", ("groq_base_url", "fireworks_base_url")),
    ("Text-to-speech", ("tts_voice", "openai_tts_endpoint")),
    ("Vertex AI", ("google_cloud_credentials", "vertex_ai_location")),
    ("Replicate", ("replicate_api_key",)),
    ("Unavailable on mobile", ("authza_use_general_api",)),
)

#: URL fields -> quick-paste chips (label, value). Hosts are examples to edit.
QUICK_PASTE = {
    "openai_base_url": (("OpenAI", "https://api.openai.com/v1"),
                        ("Ollama on LAN", "http://192.168.1.10:11434/v1"),
                        ("LM Studio on LAN", "http://192.168.1.10:1234/v1"),
                        ("Azure", "https://YOUR-RESOURCE.openai.azure.com"),
                        ("TTS on LAN", "http://192.168.1.10:8000/audio/speech"),
                        ("TTS v1 on LAN", "http://192.168.1.10:8000/v1/audio/speech"),
                        ("Clear", "")),
    # the desktop TTS quick-pastes (local FastAPI TTS server; the phone reaches it over the LAN)
    "openai_tts_endpoint": (("TTS on LAN", "http://192.168.1.10:8000/audio/speech"),
                            ("TTS v1 on LAN", "http://192.168.1.10:8000/v1/audio/speech"),
                            ("Clear", "")),
    "gemini_openai_endpoint": (("Google (gRPC host)", "generativelanguage.googleapis.com"),
                               ("Google (OpenAI-compatible)", "https://generativelanguage.googleapis.com/v1beta/openai/")),
    "anthropic_base_url": (("Anthropic", "https://api.anthropic.com"),),
    "groq_base_url": (("Groq", "https://api.groq.com/openai/v1"),
                      ("Local server on LAN", "http://192.168.1.10:8080/v1")),
    "fireworks_base_url": (("Fireworks", "https://api.fireworks.ai/inference/v1"),),
    "custom_image_edit_endpoint": (("OpenAI", "https://api.openai.com/v1"),),
}

#: Fields whose "Clear" chip writes '' (other_settings ``_clear_openai_url`` blanks the URL and the TTS endpoint).
TTS_CLEARS = ("openai_base_url", "openai_tts_endpoint")

DESKTOP_ONLY_ROWS = (
    ("AuthZA / GLM access mode", "Z.AI login and the GLM access modes use desktop-only routes (excluded on mobile)."),
    ("🦙 Load Ollama (ollamapull/)", "Managed Ollama installs and pulls models with a desktop Ollama binary. "
                                    "Use Local AI to reach an Ollama or LM Studio server on your network."),
)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def dependency_reasons() -> dict:
    """Settings whose feature needs a package this build may not have (plan dependency rule).

    U9 pins the Tier-B packages (src/mobile/pyproject.toml: grpcio + google-ai-generativelanguage,
    google-cloud-texttospeech, ...), so a release build has no reasons here; they remain for builds
    and dev environments without them. Vertex AI never needs google-cloud-aiplatform on mobile (it
    cannot be installed next to protobuf 7): unified_api_client runs Vertex Gemini through
    google-genai and Vertex Claude through anthropic's AnthropicVertex, REST + google-auth.
    """
    reasons: dict = {}
    if not (_has("grpc") and _has("google.ai.generativelanguage_v1beta")):
        reasons["gemini_openai_endpoint"] = ("Needs grpcio · not in this build",
                                             "A bare host name selects the Gemini gRPC transport, which needs grpcio and "
                                             "google-ai-generativelanguage. An https://…/openai/ URL (OpenAI-compatible "
                                             "REST) works without them.")
    vertex_sdk = _has("vertexai") or _has("google.cloud.aiplatform")
    if not (vertex_sdk or (_has("google.genai") and _has("google.auth"))):
        reasons["vertex_ai_location"] = ("Needs Vertex support · not in this build",
                                         "Vertex AI routes need google-genai and google-auth (REST) or "
                                         "google-cloud-aiplatform, which this build does not include. The settings "
                                         "are kept.")
    if not _has("google.cloud.texttospeech"):
        reasons["tts_voice"] = ("Google Cloud voices: not in this build",
                                "Google Cloud Text-to-Speech voices need google-cloud-texttospeech. OpenAI-compatible "
                                "and Gemini voices work without it.")
    return reasons


# ---- shared test path ---------------------------------------------------------------------------------


def run_in_run_env(config: Mapping[str, Any], fn: Callable[[], Any], *, lock_timeout: float = 2.0,
                   input_kind: str = "epub", data_dir: Optional[str] = None,
                   owner_factory: Optional[Callable[..., Any]] = None,
                   env_builder: Optional[Callable[..., Any]] = None) -> Any:
    """Blocking: ``fn()`` with the environment the next translation run would get.

    Builds the env like the Env preview (``HeadlessOwner`` + ``run_env.build_translation_env``
    under ``JOB_LOCK``, process state restored afterwards). When a job holds the lock, its own
    environment is already in place and ``fn`` runs as is.
    """
    from glossarion_mobile.ui.screens import env_preview as ep

    lock = ep.ENV_PREVIEW_LOCK
    if not lock.acquire(timeout=lock_timeout):
        return fn()
    try:
        snapshot = copy.deepcopy(dict(config))
        key = str(snapshot.get("api_key") or "")
        try:
            import unified_api_client  # noqa: F401 - lets the scope snapshot / restore the key pools
        except Exception:
            pass
        with ep.scoped_process_env():
            host = ep._PreviewHost()
            owner = (owner_factory or ep._default_owner_factory)(snapshot, host=host, api_key=key)
            env = (env_builder or ep._default_env_builder)(owner, ep.preview_input_path(input_kind, data_dir), key)
            try:
                import large_env

                large_env.update_env(dict(env or {}))
            except ImportError:
                os.environ.update({str(k): str(v) for k, v in dict(env or {}).items() if v is not None})
            return fn()
    finally:
        lock.release()


#: Desktop vars the shared endpoint collector reads (``key_pool_service.collect_test_endpoints``).
ENDPOINT_VARS = ("use_custom_openai_endpoint_var", "openai_base_url_var", "azure_api_version_var", "model_var",
                 "groq_base_url_var", "fireworks_base_url_var", "use_gemini_openai_endpoint_var",
                 "gemini_openai_endpoint_var")
NO_ENDPOINTS_TEXT = "No custom endpoints configured."  # the desktop Info box


def endpoint_owner(config: Mapping[str, Any]) -> Any:
    """An attribute-only owner with the desktop start-up values of the endpoint vars for ``config``
    (``settings_rules._config_var``: the config value, else the desktop init default)."""
    from types import SimpleNamespace

    import settings_rules

    values = {}
    for name in ENDPOINT_VARS:
        try:
            values[name] = settings_rules._config_var(dict(config), name)
        except Exception:
            continue
    return SimpleNamespace(**values)


def collect_endpoints(config: Mapping[str, Any]) -> list:
    """The desktop Test Connections endpoint list ``(name, url, model[, kind])`` for ``config``."""
    from key_pool_service import collect_test_endpoints

    return list(collect_test_endpoints(endpoint_owner(config)))


def run_endpoint_probes(config: Mapping[str, Any], cancel_event: Any = None) -> list:
    """Blocking: the desktop per-endpoint probe lines (``key_pool_service.run_endpoint_tests``; the key is
    the configured ``api_key``, else the desktop "sk-dummy-key" for local servers)."""
    import threading

    from key_pool_service import run_endpoint_tests

    endpoints = collect_endpoints(config)
    if not endpoints:
        return []
    try:
        import openai
    except ImportError:
        return ["❌ OpenAI library not installed"]
    api_key = str(config.get("api_key") or "") or "sk-dummy-key"
    return list(run_endpoint_tests(endpoints, api_key, openai, cancel_event or threading.Event()))


def endpoint_summary(config: Mapping[str, Any]) -> list:
    """``(name, url)`` of the endpoints Test connection probes (the shared desktop collector), plus the
    Anthropic / image-edit overrides a run would use (shown, not probed: the desktop does not either)."""
    rows = []
    get = config.get
    try:
        rows = [(str(item[0]), str(item[1])) for item in collect_endpoints(config)]
    except Exception:
        log.debug("endpoint collector unavailable", exc_info=True)
    if get("force_native_anthropic") and get("anthropic_base_url"):
        rows.append(("Anthropic (Custom)", str(get("anthropic_base_url"))))
    if get("use_custom_image_edit_endpoint") and get("custom_image_edit_endpoint"):
        rows.append(("Image edit", str(get("custom_image_edit_endpoint"))))
    return rows


def test_connection(config: Mapping[str, Any], *, backend: Any = None, timeout: float = 30.0,
                    runner: Optional[Callable[..., Any]] = None) -> dict:
    """Blocking: the configured model + key through ``key_pool_service.run_key_test`` in the run env."""
    from glossarion_mobile.ui.screens.keys import KeyBackend

    backend = backend or KeyBackend()
    model = str(config.get("model") or "").strip()
    if not model:
        return {"ok": None, "status": "untestable", "message": "Choose a model first"}
    entry = backend.new_entry(str(config.get("api_key") or ""), model) if backend.available else {
        "api_key": str(config.get("api_key") or ""), "model": model}
    runner = runner or run_in_run_env
    result = runner(config, lambda: backend.run_test(entry, "main", timeout=timeout))
    result = dict(result or {})
    result.setdefault("model", model)
    return result


# ---- screen ------------------------------------------------------------------------------------------

try:  # the pure part above must stay importable without Flet (host tests, services)
    import flet as ft

    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.components.reason_chip import ReasonChip, unavailable_tile
    from glossarion_mobile.ui.components.section_card import SectionCard
    from glossarion_mobile.ui.screens.base import Screen
    from glossarion_mobile.ui.settings.banners import JobBanner
    from glossarion_mobile.ui.settings.tiles import EffectiveConfig, make_tile
    from glossarion_mobile.ui.theme import semantic
except ImportError:  # pragma: no cover - Flet missing
    ft = None  # type: ignore[assignment]
    Screen = object  # type: ignore[assignment,misc]


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


class EndpointsScreen(Screen):  # type: ignore[misc,valid-type]
    title = "Endpoints"

    def __init__(self, match: Any, ctx: Any, *, run_io: Optional[Callable[..., Any]] = None,
                 open_local_ai: Optional[Callable[[], Any]] = None, tester: Optional[Callable[[dict], dict]] = None,
                 dependency: Optional[Mapping[str, tuple]] = None, copy_text: Optional[Callable[[str], Any]] = None,
                 read_clipboard: Optional[Callable[[], Any]] = None,
                 prober: Optional[Callable[[dict], list]] = None) -> None:
        super().__init__(match)
        self.ctx = ctx
        self.copy_text = copy_text
        self.read_clipboard = read_clipboard
        self.secret_fields: dict = {}
        self.run_io = run_io
        self.open_local_ai = open_local_ai
        self.tester = tester or test_connection
        self.prober = prober or run_endpoint_probes
        self.endpoint_lines: list = []
        self.dependency = dict(dependency) if dependency is not None else dependency_reasons()
        self.tiles: dict = {}
        self.config_view = EffectiveConfig(ctx.store)
        self.banner = JobBanner(ctx)
        self.last_result: Optional[dict] = None
        self._unsubs: list = []
        # U9: Settings › Custom API Endpoints and settings search hits open this page at ``#key``
        self.focus_target: Optional[str] = getattr(match, "fragment", None) if match is not None else None
        self._focus_task: Any = None

    def _tile(self, key: str) -> Any:
        try:
            spec = self.ctx.schema.spec(key)
        except Exception:
            spec = None
        if spec is None:
            return None
        tile = make_tile(spec, self.ctx, config=self.config_view)
        self.tiles[key] = tile
        return tile

    def _quick_chips(self, key: str, tile: Any) -> Optional["ft.Control"]:
        options = QUICK_PASTE.get(key)
        if not options:
            return None
        chips = [ft.Chip(label=ft.Text(label, theme_style=ft.TextThemeStyle.LABEL_SMALL),
                         leading=ft.Icon(ft.Icons.CONTENT_PASTE_GO, size=14), tooltip=value,
                         on_click=lambda e, v=value, t=tile: self.quick_paste(t, v), key=f"paste-{key}-{label}")
                 for label, value in options]
        return ft.Container(content=ft.Row(chips, wrap=True, spacing=6, run_spacing=4),
                            padding=ft.Padding.only(left=12, right=12, bottom=6))

    def _secret_field(self, key: str, tile: Any) -> "ft.Control":
        """Secrets: the masked KeyField (eye · paste · copy) under the tile, writing the same key."""
        from glossarion_mobile.ui.screens.key_editor import KeyField

        def save(value: str) -> None:
            if value:
                tile.apply(value)
            elif self.ctx.store.has(key):
                self.ctx.store.unset(key)
                tile.refresh()

        current = self.ctx.store.get(key, "") or ""
        field = KeyField(value=str(current), label=tile.label, on_change=save, read_clipboard=self.read_clipboard,
                         copy_text=self.copy_text, test_reason="Use Test connection")
        self.secret_fields[key] = field
        return ft.Container(content=field, padding=ft.Padding.only(left=12, right=12, bottom=8))

    def quick_paste(self, tile: Any, value: str) -> bool:
        """A quick-paste chip. Override API Endpoint follows the desktop pastes / Clear
        (other_settings): a ``…/audio/speech`` URL is also the TTS endpoint (the tile's implied
        write), every other paste and Clear blank ``openai_tts_endpoint`` so a stale TTS endpoint
        never keeps priority in run_env (OPENAI_TTS_ENDPOINT)."""
        if not tile.editable:
            return False
        if value == "" and tile.key in TTS_CLEARS:
            ok = self._clear(tile)
        else:
            ok = bool(tile.apply(value))
        if ok and tile.key == "openai_base_url" and not str(value).rstrip("/").endswith("/audio/speech"):
            if self.ctx.store.get("openai_tts_endpoint", "") not in ("", None):
                self.ctx.store.set("openai_tts_endpoint", "")
            other = self.tiles.get("openai_tts_endpoint")
            if other is not None:
                other.refresh()
        tile.refresh()
        return ok

    def _clear(self, tile: Any) -> bool:
        """Clear: the desktop stores '' (not the default) for the URL field."""
        if self.ctx.store.get(tile.key, "") in ("", None) and self.ctx.store.has(tile.key):
            return True
        self.ctx.store.set(tile.key, "")
        return True

    def build_body(self) -> "ft.Control":
        controls: list = [self.banner.control]
        if not self.ctx.schema.available:
            controls.append(ft.Text("Settings schema unavailable; config.json is kept untouched.",
                                    color=ft.Colors.ON_SURFACE_VARIANT))
            return ft.ListView(controls=controls, expand=True, padding=ft.Padding.all(12))
        for title, keys in ENDPOINT_SECTIONS:
            children: list = []
            for key in keys:
                tile = self._tile(key)
                if tile is None:
                    continue
                children.append(tile.control)
                if tile.kind == "secret" and tile.editable:
                    children.append(self._secret_field(key, tile))
                dep = self.dependency.get(key)
                if dep is not None:
                    children.append(ft.Container(content=ReasonChip(reason=dep[0], detail=dep[1]),
                                                 padding=ft.Padding.only(left=12), key=f"dep-{key}"))
                chips = self._quick_chips(key, tile)
                if chips is not None:
                    children.append(chips)
            if title == "Unavailable on mobile":
                for label, reason in DESKTOP_ONLY_ROWS:  # enabled rows: the chip opens the reason
                    children.append(unavailable_tile(label, reason="Not available on mobile", detail=reason,
                                                     key=f"unavailable-{label}"))
            if children:
                focused = bool(self.focus_target) and self.focus_target in keys
                controls.append(SectionCard(title=title, children=children, collapsible=title != "Custom OpenAI endpoint",
                                            expanded=focused or title in ("Custom OpenAI endpoint", "Provider base URLs"),
                                            key=f"endpoints-{title}"))
        controls.insert(1, SectionCard(title="Local AI", icon="LAN_OUTLINED", children=[
            ft.ListTile(title=ft.Text("Ollama / LM Studio on your network"),
                        subtitle=ft.Text("Base URLs, model list and Ollama options"),
                        leading=ft.Icon(ft.Icons.COMPUTER), trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT),
                        on_click=lambda e: self._open_local_ai(), key="endpoints-local-ai"),
        ]))
        self.result_text = ft.Text("", selectable=True, visible=False)
        self.endpoint_results = ft.Column([], spacing=4, tight=True, visible=False, key="endpoints-results")
        self.summary_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)
        self.ring = ft.ProgressRing(width=18, height=18, stroke_width=2, visible=False)
        self.test_button = ft.FilledButton(content="Test connection", icon=ft.Icons.NETWORK_CHECK,
                                           on_click=lambda e: self.ctx.spawn(self.run_test()))
        self._refresh_summary()
        controls.insert(1, SectionCard(title="Test connection", icon="NETWORK_CHECK", children=[
            ft.Text("Probes every configured endpoint (desktop Test Connections), then sends a short request with "
                    "the selected model and key through the endpoint settings a run would use.",
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
            self.summary_text,
            ft.Row([self.test_button, self.ring], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.endpoint_results,
            self.result_text,
        ], key="endpoints-test"))
        # Every card is built (about 12, like SectionPage): scroll_to(scroll_key=…) only reaches built
        # items, and a search hit / open_setting lands on a tile of any section (``focus_key``).
        self.list_view = ft.ListView(controls=controls, expand=True, spacing=tokens.SPACING["sm"],
                                     padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"],
                                                                  vertical=tokens.SPACING["sm"]),
                                     build_controls_on_demand=False, auto_scroll=False)
        return self.list_view

    def _refresh_summary(self) -> None:
        config = self.ctx.store.snapshot()
        rows = endpoint_summary(config)
        model = str(config.get("model") or self.ctx.store.effective("model") or "")
        lines = [f"Model: {model or '(none)'}"] + [f"{name}: {url}" for name, url in rows]
        if not rows:
            lines.append("No custom endpoint is enabled: the provider's own API is used.")
        self.summary_text.value = "\n".join(lines)

    def _open_local_ai(self) -> None:
        if self.open_local_ai is not None:
            self.open_local_ai()

    async def run_test(self) -> dict:
        self.ring.visible = True
        self.test_button.disabled = True
        self.result_text.visible = False
        self._refresh_summary()
        _push(self.ring, self.test_button, self.result_text, self.summary_text)
        config = self.ctx.store.snapshot()
        if "model" not in config:
            config["model"] = self.ctx.store.effective("model")
        run = self.run_io if self.run_io is not None else self.ctx.run_io
        try:
            lines = await run(self.prober, config)
        except Exception as exc:
            lines = [f"❌ Endpoint test failed: {exc}"]
        self.endpoint_lines = list(lines or [])
        self.endpoint_results.controls = [
            ft.Text(line, selectable=True, theme_style=ft.TextThemeStyle.BODY_SMALL,
                    color=semantic("success") if line.startswith("✅") else ft.Colors.ERROR)
            for line in self.endpoint_lines
        ] or [ft.Text(NO_ENDPOINTS_TEXT, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)]
        self.endpoint_results.visible = True
        _push(self.endpoint_results)
        try:
            result = await run(self.tester, config)
        except Exception as exc:
            result = {"ok": False, "status": "error", "message": str(exc)}
        self.last_result = dict(result or {})
        status = self.last_result.get("status")
        model = self.last_result.get("model") or config.get("model")
        if status == "passed":
            text, color = f"✅ Connected successfully! (Model: {model})", semantic("success")
        elif status == "untestable":
            text, color = f"ℹ️ {self.last_result.get('message') or 'This route cannot be tested here'}", ft.Colors.ON_SURFACE_VARIANT
        else:
            text, color = f"❌ {self.last_result.get('message') or 'Connection failed'}", ft.Colors.ERROR
        self.result_text.value = text
        self.result_text.color = color
        self.result_text.visible = True
        self.ring.visible = False
        self.test_button.disabled = False
        _push(self.ring, self.test_button, self.result_text)
        return self.last_result

    def did_show(self) -> None:
        if self.focus_target in self.tiles and self._focus_task is None:
            self._focus_task = self.ctx.spawn(self.focus_key(self.focus_target))
        if self._unsubs:
            return
        # Rules (locks, "not used") may depend on any key: refresh every tile, like SectionPage.
        self._unsubs.append(self.ctx.store.observe_all(lambda key, value: self.ctx.on_ui(self._on_change, key)))
        self.banner.attach()
        self._unsubs.append(self.banner.detach)

    async def focus_key(self, key: str, *, highlight_seconds: float = 1.5) -> bool:
        """Scroll to ``key``'s tile and highlight it (a search hit / ``open_setting`` landing here)."""
        tile = self.tiles.get(key)
        list_view = getattr(self, "list_view", None)
        if tile is None or list_view is None:
            return False
        await asyncio.sleep(0.05)  # the client builds the page first
        try:
            await list_view.scroll_to(scroll_key=ft.ScrollKey(key), duration=300)
        except Exception as exc:  # not mounted (tests) / client gone
            log.debug("scroll_to(%s) failed: %s", key, exc)
        tile.set_highlight(True)
        try:
            await asyncio.sleep(highlight_seconds)
        finally:
            tile.set_highlight(False)
        return True

    def _on_change(self, key: str) -> None:
        for tile in self.tiles.values():
            tile.refresh(push=False)
        self._refresh_summary()
        _push(getattr(self, "list_view", None))

    def dispose(self) -> None:
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs = []
