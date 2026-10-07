"""Local AI (Ollama / LM Studio on your network; UI_SPEC §4.15 Endpoints › Local LLM host).

On the desktop the ``ollama/`` and ``lmstudio/`` routes call a server on the same computer
(``UnifiedClient._LOCAL_OPENAI_ROUTES``: ``localhost:11434`` / ``localhost:1234``), and the managed
``ollamapull/`` route installs Ollama itself. A phone has neither, so this page points a model
prefix at the computer on your network that runs the server, using the desktop's own
**custom prefix routes** (``custom_prefix_routes`` → ``CUSTOM_OPENAI_PREFIX_ROUTES``):

* ``ollama-lan/`` → ``http://<host>:11434/v1`` and ``lmstudio-lan/`` → ``http://<host>:1234/v1``
  (endpoint type ``/chat/completions``). The client treats a private-network base URL as a local
  endpoint (no API key, ``UnifiedClient._is_local_openai_base_url``); the model list comes from
  the server's ``/v1/models`` through ``model_options``' custom-route catalog, so the models show
  up in the ModelSheet as ``ollama-lan/<name>`` with ✓ polled markers. A desktop that imports the
  config routes them the same way.
* **Ollama options** (``ollama_settings``, the shared ``ollama_settings`` helpers): per-model
  request options of the managed ``ollamapull/`` route. That route is desktop-only, so the editor
  is labelled as such; values round-trip with a desktop config.
* ``ollamapull/`` (install / pull / update Ollama) is shown disabled with its reason.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from typing import Any, Callable, Mapping, Optional, Sequence

__all__ = [
    "LAN_SERVERS",
    "LocalAiScreen",
    "OllamaOptionsForm",
    "base_url_of",
    "lan_route",
    "remove_lan_route",
    "routing_for",
    "set_lan_route",
    "validate_base_url",
]

log = logging.getLogger("glossarion.local_ai")

#: id, title, route prefix, default port, help
LAN_SERVERS = (
    ("ollama", "Ollama", "ollama-lan/", 11434,
     "Start Ollama on your computer with OLLAMA_HOST=0.0.0.0 so other devices can reach it."),
    ("lmstudio", "LM Studio", "lmstudio-lan/", 1234,
     "In LM Studio's Developer tab, start the server and enable “Serve on Local Network”."),
)
OLLAMAPULL_REASON = ("Managed Ollama (ollamapull/) installs Ollama and pulls models with the desktop binary, "
                     "which a phone cannot run. Point Ollama on your network here instead.")
OPTIONS_NOTE = ("These per-model options are sent by the managed ollamapull/ route on the desktop. Requests "
                "through a LAN route use the server's model defaults. Values stay in sync with your desktop config.")
_HOST_RE = re.compile(r"^[A-Za-z0-9.\-\[\]:]+$")


def lan_route(routes: Sequence[Mapping[str, Any]], prefix: str) -> Optional[dict]:
    for route in routes or ():
        if isinstance(route, Mapping) and str(route.get("prefix", "")).lower() == prefix.lower():
            return dict(route)
    return None


def base_url_of(routing: str) -> str:
    """``http://host:11434/v1`` -> ``http://host:11434`` (what the user typed)."""
    text = str(routing or "").strip().rstrip("/")
    return text[:-3] if text.lower().endswith("/v1") else text


def validate_base_url(text: str, default_port: int) -> tuple:
    """(base URL, None) or (None, message). ``192.168.1.10`` becomes ``http://192.168.1.10:<port>``."""
    value = str(text or "").strip().rstrip("/")
    if not value:
        return None, "Enter the address of the computer running the server"
    if "://" not in value:
        value = "http://" + value
    scheme, _, rest = value.partition("://")
    if scheme.lower() not in ("http", "https"):
        return None, "The address must start with http:// or https://"
    host_port = rest.split("/", 1)[0]
    if not host_port or not _HOST_RE.match(host_port):
        return None, "That does not look like a host name or IP address"
    path = rest[len(host_port):]
    if path.lower() in ("/v1", ""):
        path = ""
    has_port = bool(re.search(r":\d+$", host_port)) and not host_port.endswith("]")
    if not has_port:
        host_port = f"{host_port}:{default_port}"
    return f"{scheme.lower()}://{host_port}{path}", None


def routing_for(base_url: str) -> str:
    text = str(base_url or "").strip().rstrip("/")
    return text if text.lower().endswith("/v1") else f"{text}/v1"


def set_lan_route(routes: Sequence[Mapping[str, Any]], prefix: str, base_url: str) -> list:
    """Replace (in place) or append the prefix route; other routes keep their order and fields."""
    new = {"prefix": prefix, "routing": routing_for(base_url), "endpoint_type": "/chat/completions"}
    out: list = []
    replaced = False
    for route in routes or ():
        if isinstance(route, Mapping) and str(route.get("prefix", "")).lower() == prefix.lower():
            merged = dict(route)
            merged.update(new)
            out.append(merged)
            replaced = True
        else:
            out.append(copy.deepcopy(dict(route)) if isinstance(route, Mapping) else route)
    if not replaced:
        out.append(new)
    return out


def remove_lan_route(routes: Sequence[Mapping[str, Any]], prefix: str) -> list:
    return [copy.deepcopy(dict(r)) for r in routes or ()
            if isinstance(r, Mapping) and str(r.get("prefix", "")).lower() != prefix.lower()]


# ---- Ollama options (ollama_settings) ------------------------------------------------------------


def collect_model_options(values: Mapping[str, str], *, extra_options: str = "", extra_request: str = "",
                          think: str = "", keep_alive: str = "", response_format: str = "",
                          base: Optional[Mapping[str, Any]] = None) -> dict:
    """One model's settings from the form, with the desktop dialog's checks
    (``OllamaSettingsDialog._collect_model_settings``) on the shared ``ollama_settings`` parsers."""
    from ollama_settings import COMMON_OPTION_KEYS, OPTION_GROUPS, RESERVED_REQUEST_KEYS, _json_object, parse_option

    kinds = {name: kind for _group, rows in OPTION_GROUPS for name, _label, kind in rows}
    extra = _json_object(extra_options, "Additional options")
    duplicates = COMMON_OPTION_KEYS.intersection(extra)
    if duplicates:
        raise ValueError("Additional options duplicate a named field: " + ", ".join(sorted(duplicates)))
    options = dict(extra)
    for key, text in values.items():
        text = str(text or "").strip()
        if text and key in kinds:
            try:
                options[key] = parse_option(text, kinds[key])
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{key}: {exc}") from exc
    if "num_ctx" in options and options["num_ctx"] <= 0:
        raise ValueError("num_ctx must be greater than zero")
    if "draft_num_predict" in options and options["draft_num_predict"] < 0:
        raise ValueError("draft_num_predict must be zero or greater")
    request = _json_object(extra_request, "Additional request fields")
    reserved = RESERVED_REQUEST_KEYS.intersection(request)
    if reserved:
        raise ValueError("Additional request fields cannot override: " + ", ".join(sorted(reserved)))
    result = copy.deepcopy(dict(base or {}))
    result["options"] = options
    result["request"] = request
    think = str(think or "").strip()
    if think and think.casefold() not in ("default", "model default"):
        result["think"] = think.casefold() == "true" if think.casefold() in ("true", "false") else think
    else:
        result.pop("think", None)
    keep_alive = str(keep_alive or "").strip()
    if keep_alive and keep_alive.casefold() not in ("default", "model default"):
        result["keep_alive"] = keep_alive
    else:
        result.pop("keep_alive", None)
    response_format = str(response_format or "").strip()
    if response_format and response_format.casefold() not in ("default", "model default"):
        if response_format.startswith("{"):
            result["format"] = _json_object(response_format, "Response format")
        elif response_format == "json":
            result["format"] = "json"
        else:
            raise ValueError("Response format must be json or a JSON schema object")
    else:
        result.pop("format", None)
    return result


# ---- screen ------------------------------------------------------------------------------------------

try:  # the pure part above must stay importable without Flet (host tests, services)
    import flet as ft

    from glossarion_mobile.ui import tokens
    from glossarion_mobile.ui.components.reason_chip import NOT_ON_MOBILE, ReasonChip
    from glossarion_mobile.ui.components.section_card import SectionCard
    from glossarion_mobile.ui.screens.base import Screen
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


class OllamaOptionsForm:
    """Per-model ``ollama_settings.models[<name>]`` editor (desktop "🦙 Ollama settings")."""

    def __init__(self, store: Any, *, model: str = "") -> None:
        from ollama_settings import COMMON_OPTION_KEYS, OPTION_GROUPS, normalize_ollama_settings

        self.store = store
        self.normalize = normalize_ollama_settings
        self.groups = OPTION_GROUPS
        self.common = COMMON_OPTION_KEYS
        settings = normalize_ollama_settings(store.get("ollama_settings", None))
        names = sorted(settings.get("models", {}))
        self.model_field = ft.Dropdown(value=model or (names[0] if names else None), label="Ollama model",
                                       editable=True, enable_filter=True, dense=True,
                                       options=[ft.DropdownOption(key=n, text=n) for n in names],
                                       on_select=lambda e: self.load(self.model_field.value or ""))
        self.fields: dict = {}
        self.error = ft.Text("", color=ft.Colors.ERROR, visible=False, selectable=True)
        # Success feedback (the server rows show "Saved: …" the same way); nothing changed visibly before.
        self.status = ft.Text("", color=semantic("success"), visible=False)
        groups: list = []
        for group, rows in OPTION_GROUPS:
            controls = []
            for name, label, kind in rows:
                field = ft.TextField(label=label, hint_text=f"{name} · {kind}", dense=True,
                                     border_radius=tokens.RADII["field"])
                self.fields[name] = field
                controls.append(field)
            groups.append(ft.ExpansionTile(title=ft.Text(group), controls=controls, expanded=False,
                                           controls_padding=ft.Padding.only(left=8, right=8, bottom=8)))
        self.extra_options = ft.TextField(label="Additional options (JSON)", multiline=True, min_lines=2, dense=True)
        self.extra_request = ft.TextField(label="Additional request fields (JSON)", multiline=True, min_lines=2,
                                          dense=True)
        self.think = ft.TextField(label="Think", hint_text="model default · true · false · low · medium · high", dense=True)
        self.keep_alive = ft.TextField(label="Keep alive", hint_text="model default · 5m · -1", dense=True)
        self.response_format = ft.TextField(label="Response format", hint_text="model default · json · {schema}",
                                            dense=True)
        self.save_button = ft.FilledTonalButton(content="Save options", icon=ft.Icons.SAVE_OUTLINED,
                                                on_click=lambda e: self.save())
        self.control = ft.Column([self.model_field, *groups, self.extra_options, self.extra_request, self.think,
                                  self.keep_alive, self.response_format, self.error, self.status, self.save_button],
                                 spacing=8, tight=True)
        self.load(self.model_field.value or "")

    def load(self, model: str) -> None:
        settings = self.normalize(self.store.get("ollama_settings", None))
        current = settings.get("models", {}).get(model, {}) if model else {}
        options = dict(current.get("options") or {})
        for name, field in self.fields.items():
            value = options.get(name)
            field.value = "" if value is None else (json.dumps(value) if isinstance(value, (list, dict, bool)) else str(value))
        extra = {k: v for k, v in options.items() if k not in self.common}
        self.extra_options.value = json.dumps(extra, ensure_ascii=False) if extra else ""
        request = current.get("request") or {}
        self.extra_request.value = json.dumps(request, ensure_ascii=False) if request else ""
        think = current.get("think")
        self.think.value = "" if think is None else (str(think).lower() if isinstance(think, bool) else str(think))
        self.keep_alive.value = str(current.get("keep_alive") or "")
        fmt = current.get("format")
        self.response_format.value = json.dumps(fmt) if isinstance(fmt, dict) else str(fmt or "")
        self.error.visible = False
        self.status.visible = False
        _push(self.control)

    def save(self) -> Optional[str]:
        model = (self.model_field.value or "").strip()
        if not model:
            return self._fail("Enter the Ollama model name (for example qwen3:8b)")
        settings = self.normalize(self.store.get("ollama_settings", None))
        try:
            result = collect_model_options(
                {name: field.value or "" for name, field in self.fields.items()},
                extra_options=self.extra_options.value or "", extra_request=self.extra_request.value or "",
                think=self.think.value or "", keep_alive=self.keep_alive.value or "",
                response_format=self.response_format.value or "", base=settings["models"].get(model, {}),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._fail(str(exc))
        settings["models"][model] = result
        self.store.set("ollama_settings", settings)
        self.error.visible = False
        self.status.value = f"Saved options for {model}"
        self.status.visible = True
        _push(self.error, self.status)
        return None

    def _fail(self, message: str) -> str:
        self.error.value = message
        self.error.visible = True
        self.status.visible = False
        _push(self.error, self.status)
        return message


class LocalAiScreen(Screen):  # type: ignore[misc,valid-type]
    title = "Local AI"

    def __init__(self, match: Any, *, store: Any, catalog: Any = None, notify: Optional[Callable[..., Any]] = None,
                 spawn: Optional[Callable[[Any], Any]] = None, navigate: Optional[Callable[..., Any]] = None) -> None:
        super().__init__(match)
        self.store = store
        self.catalog = catalog
        self.notify = notify
        self.spawn_fn = spawn
        self.navigate = navigate
        self.url_fields: dict = {}
        self.status: dict = {}
        self.options_form: Optional[OllamaOptionsForm] = None

    def say(self, message: str) -> None:
        if self.notify is not None:
            try:
                self.notify(message)
            except Exception:
                pass

    def spawn(self, coro: Any) -> Any:
        if self.spawn_fn is not None:
            return self.spawn_fn(coro)
        return asyncio.ensure_future(coro)

    def routes(self) -> list:
        value = self.store.get("custom_prefix_routes", [])
        return list(value) if isinstance(value, list) else []

    def build_body(self) -> "ft.Control":
        controls: list = [ft.Container(padding=ft.Padding.symmetric(horizontal=4, vertical=4), content=ft.Text(
            "Run models on a computer on your network. Each server gets its own model prefix; pick its models in the "
            "model sheet after “Load models”.", theme_style=ft.TextThemeStyle.BODY_SMALL,
            color=ft.Colors.ON_SURFACE_VARIANT))]
        for server_id, title, prefix, port, help_text in LAN_SERVERS:
            route = lan_route(self.routes(), prefix)
            field = ft.TextField(value=base_url_of(route["routing"]) if route else "", label="Server address",
                                 hint_text=f"http://192.168.1.10:{port}", dense=True, keyboard_type=ft.KeyboardType.URL,
                                 border_radius=tokens.RADII["field"])
            self.url_fields[server_id] = field
            status = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False, selectable=True)
            self.status[server_id] = status
            controls.append(SectionCard(title=f"{title} on your network", icon="COMPUTER", key=f"lan-{server_id}",
                                        subtitle=f"Models: {prefix}<name>", children=[
                ft.Text(help_text, theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                field,
                ft.Row([
                    ft.FilledTonalButton(content="Save", icon=ft.Icons.SAVE_OUTLINED,
                                         on_click=lambda e, s=server_id: self.save_server(s)),
                    ft.TextButton(content="Load models", icon=ft.Icons.PUBLIC,
                                  on_click=lambda e, s=server_id: self.spawn(self.load_models(s))),
                    ft.TextButton(content="Remove", icon=ft.Icons.DELETE_OUTLINE,
                                  on_click=lambda e, s=server_id: self.remove_server(s)),
                ], wrap=True, spacing=6),
                status,
            ]))
        try:
            self.options_form = OllamaOptionsForm(self.store)
            options_children: list = [
                ft.Row([ft.Text(OPTIONS_NOTE, theme_style=ft.TextThemeStyle.BODY_SMALL, expand=True),
                        ReasonChip(reason="ollamapull/ is desktop-only", detail=OLLAMAPULL_REASON)]),
                self.options_form.control,
            ]
        except Exception as exc:  # backend helpers missing
            log.info("ollama_settings unavailable: %s", exc)
            options_children = [ft.Text(f"Ollama options need the shared ollama_settings module ({exc}).")]
        controls.append(SectionCard(title="🦙 Ollama options", collapsible=True, expanded=False,
                                    children=options_children, key="lan-ollama-options"))
        controls.append(SectionCard(title="Unavailable on mobile", children=[
            ft.ListTile(title=ft.Text("🦙 Load Ollama (ollamapull/)"), subtitle=ft.Text("Install, pull and update models"),
                        disabled=True, dense=True, trailing=ReasonChip(reason=NOT_ON_MOBILE, detail=OLLAMAPULL_REASON),
                        key="lan-ollamapull"),
        ]))
        self.list_view = ft.ListView(controls=controls, expand=True, spacing=tokens.SPACING["sm"],
                                     padding=ft.Padding.symmetric(horizontal=tokens.SPACING["md"],
                                                                  vertical=tokens.SPACING["sm"]))
        return self.list_view

    def _server(self, server_id: str) -> tuple:
        for row in LAN_SERVERS:
            if row[0] == server_id:
                return row
        raise KeyError(server_id)

    def _set_status(self, server_id: str, text: str, ok: Optional[bool] = None) -> None:
        status = self.status.get(server_id)
        if status is None:
            return
        status.value = text
        status.color = semantic("success") if ok else (ft.Colors.ERROR if ok is False else ft.Colors.ON_SURFACE_VARIANT)
        status.visible = bool(text)
        _push(status)

    def save_server(self, server_id: str) -> Optional[str]:
        _sid, title, prefix, port, _help = self._server(server_id)
        field = self.url_fields[server_id]
        base, error = validate_base_url(field.value or "", port)
        if error:
            field.error = error
            _push(field)
            return error
        field.error = None
        field.value = base
        _push(field)
        routes = set_lan_route(self.routes(), prefix, base)
        if self.catalog is not None:
            ok, problem = self.catalog.save_prefix_routes(routes)
            if not ok:
                message = problem[1] if isinstance(problem, tuple) else str(problem)
                self._set_status(server_id, message, False)
                return message
        else:
            self.store.set("custom_prefix_routes", routes)
        self._set_status(server_id, f"Saved: models use {prefix}<name> → {routing_for(base)}", True)
        return None

    def remove_server(self, server_id: str) -> bool:
        _sid, title, prefix, _port, _help = self._server(server_id)
        routes = self.routes()
        if lan_route(routes, prefix) is None:
            return False
        new = remove_lan_route(routes, prefix)
        if self.catalog is not None:
            self.catalog.save_prefix_routes(new)
        else:
            self.store.set("custom_prefix_routes", new)
        self.url_fields[server_id].value = ""
        _push(self.url_fields[server_id])
        self._set_status(server_id, f"Removed the {prefix} route", None)
        return True

    async def load_models(self, server_id: str) -> Any:
        _sid, title, prefix, _port, _help = self._server(server_id)
        if lan_route(self.routes(), prefix) is None:
            error = self.save_server(server_id)
            if error:
                return None
        if self.catalog is None:
            self._set_status(server_id, "The model catalog service is not available", False)
            return None
        self._set_status(server_id, f"Contacting {title}…", None)
        outcome = await self.catalog.refresh(f"custom:{prefix}", explicit=True)
        status = str(outcome.statuses.get(f"custom:{prefix}", outcome.message))
        self._set_status(server_id, f"{title}: {status}", bool(outcome.ok))
        return outcome
