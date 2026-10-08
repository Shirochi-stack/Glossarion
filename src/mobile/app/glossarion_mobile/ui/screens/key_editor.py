"""KeyField and KeyEditor (UI_SPEC §4.12, §5.7, §5.9).

``KeyField`` is the shared secret input: masked ``sk-…a1B2`` text with reveal (eye), Paste,
Copy and Test → status chip (empty · masked · revealed · testing · passed · failed). It is
used by the ModelSheet route row, Poe setup, the Endpoints page and the KeyEditor.

``KeyEditor`` is the full-screen per-key form of the Multi-Key Manager, field for field the
desktop key (``APIKeyEntry.to_dict``): API key, model (``ModelPicker``), cooldown, per-key
output token limit, temperature and API delay, enabled, the individual endpoint (URL + Azure API
version), Google credentials + region, 🧩 request parameters (``request_parameters`` parsing:
plain text is a string, JSON for numbers / booleans / objects; reserved names are refused) and
the request contexts the key serves (``key_contexts.POOL_CONTEXTS`` for its pool; switching one
off stores it in ``disabled_contexts``). The entry is validated by ``key_pool_service`` before
it is stored (``KeysController.update_key``). It builds on ``settings.editors.FullScreenEditor``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

import flet as ft

from glossarion_mobile.ui import tokens
from glossarion_mobile.ui.components._handlers import await_handler, call_handler
from glossarion_mobile.ui.settings.editors import FullScreenEditor
from glossarion_mobile.ui.settings.model import mask_secret
from glossarion_mobile.ui.theme import HIT_TARGET, semantic

__all__ = ["KeyEditor", "KeyField", "TEST_LABELS", "normalize_test_result"]

log = logging.getLogger("glossarion.keys")

TEST_LABELS = {
    "passed": "Passed",
    "failed": "Failed",
    "error": "Error",
    "rate_limited": "Rate limited",
    "timeout": "Timed out",
    "untestable": "Not testable",
    "busy": "Busy",
}
ENCRYPTED_HINT = "Encrypted with a key this device does not have: enter the key again"


def normalize_test_result(raw: Any) -> dict:
    """``{"ok", "status", "message"}`` from whatever the key test returned (dict, tuple, object, bool)."""
    if isinstance(raw, Mapping):
        data = dict(raw)
    elif isinstance(raw, tuple) and raw:
        data = {"ok": raw[0], "message": raw[1] if len(raw) > 1 else ""}
    elif isinstance(raw, bool):
        data = {"ok": raw}
    elif raw is None:
        data = {"ok": None, "status": "untestable"}
    else:
        data = {name: getattr(raw, name) for name in ("ok", "success", "passed", "status", "result", "message",
                                                      "error", "testable", "reason") if hasattr(raw, name)}
    ok = data.get("ok", data.get("success", data.get("passed")))
    status = str(data.get("status") or data.get("result") or "").strip().lower()
    testable = data.get("testable", True)
    message = str(data.get("message") or data.get("error") or data.get("reason") or "")
    if testable is False or status in ("untestable", "not_testable", "not testable", "skipped"):
        return {"ok": None, "status": "untestable", "message": message or "This key cannot be tested here"}
    if status in ("passed", "success", "ok"):
        ok = True
    elif status in ("failed", "error", "rate_limited", "rate limited", "timeout"):
        ok = False if ok is None else ok
    if not status:
        status = "passed" if ok else "failed"
    if status == "rate limited":
        status = "rate_limited"
    if status not in TEST_LABELS:
        status = "passed" if ok else "failed"
    return {"ok": bool(ok) if ok is not None else None, "status": status, "message": message}


def _push(*controls: Any) -> None:
    for control in controls:
        if control is None:
            continue
        try:
            control.update()
        except Exception:
            pass


class KeyField(ft.Column):
    """Masked secret field: eye · Paste · Copy · Test → status chip."""

    def __init__(
        self,
        *,
        value: str = "",
        label: str = "API key",
        on_change: Optional[Callable[[str], Any]] = None,
        on_test: Optional[Callable[[str], Any]] = None,  # async value -> result
        read_clipboard: Optional[Callable[[], Any]] = None,  # async -> str | None
        copy_text: Optional[Callable[[str], Any]] = None,
        hint: Optional[str] = None,
        test_reason: Optional[str] = None,  # Test disabled with this reason
        key: Optional[str] = None,
    ) -> None:
        super().__init__(spacing=4, tight=True, key=key)
        text = "" if value is None else str(value)
        self.encrypted = text.startswith("ENC:")
        self.on_change_value = on_change
        self.on_test = on_test
        self.read_clipboard = read_clipboard
        self.copy_text = copy_text
        self.state = "masked" if text and not self.encrypted else "empty"
        self.result: Optional[dict] = None
        self.field = ft.TextField(
            value="" if self.encrypted else text,
            label=label,
            hint_text=ENCRYPTED_HINT if self.encrypted else hint,
            password=True,
            can_reveal_password=True,
            autocorrect=False,
            enable_suggestions=False,
            dense=True,
            border_radius=tokens.RADII["field"],
            on_change=self._on_change,
            expand=True,
        )
        self.paste_button = ft.IconButton(icon=ft.Icons.CONTENT_PASTE, tooltip="Paste", on_click=self._on_paste,
                                          size_constraints=HIT_TARGET, disabled=read_clipboard is None)
        self.copy_button = ft.IconButton(icon=ft.Icons.CONTENT_COPY, tooltip="Copy", on_click=self._on_copy,
                                         size_constraints=HIT_TARGET, disabled=copy_text is None)
        reason = test_reason if on_test is not None else (test_reason or "Testing needs the key service")
        self.test_button = ft.FilledTonalButton(content="Test", icon=ft.Icons.NETWORK_CHECK, on_click=self._on_test,
                                                disabled=on_test is None or test_reason is not None,
                                                tooltip=reason)
        self.ring = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False)
        self.status_chip = ft.Chip(label=ft.Text(""), visible=False)
        self.message = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT,
                               visible=False, selectable=True)
        self.controls = [
            ft.Row([self.field, self.paste_button, self.copy_button], spacing=0,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            ft.Row([self.test_button, self.ring, self.status_chip], spacing=8, wrap=True,
                   vertical_alignment=ft.CrossAxisAlignment.CENTER),
            self.message,
        ]

    # ---- value ---------------------------------------------------------------------------------------

    @property
    def value(self) -> str:
        return (self.field.value or "").strip()

    @property
    def masked(self) -> str:
        return mask_secret(self.value)

    def set_value(self, value: str, *, notify: bool = True) -> None:
        self.field.value = str(value or "")
        self.encrypted = False
        self.state = "masked" if self.field.value else "empty"
        self._clear_result()
        _push(self.field, self)
        if notify and self.on_change_value is not None:
            call_handler(self.on_change_value, self.value)

    def _on_change(self, e: Any = None) -> None:
        self.encrypted = False
        self.state = "masked" if self.value else "empty"
        self._clear_result()
        if self.on_change_value is not None:
            call_handler(self.on_change_value, self.value)

    def _clear_result(self) -> None:
        self.result = None
        self.status_chip.visible = False
        self.message.visible = False

    # ---- actions --------------------------------------------------------------------------------------

    async def paste(self) -> Optional[str]:
        if self.read_clipboard is None:
            return None
        try:
            text = await await_handler(self.read_clipboard)
        except Exception as exc:
            log.info("clipboard read failed: %s", exc)
            return None
        text = str(text or "").strip()
        if text:
            self.set_value(text)
        return text or None

    async def _on_paste(self, e: Any = None) -> None:
        await self.paste()

    async def _on_copy(self, e: Any = None) -> None:
        if self.copy_text is not None and self.value:
            await await_handler(self.copy_text, self.value)

    async def test(self) -> Optional[dict]:
        if self.on_test is None:
            return None
        self.state = "testing"
        self.ring.visible = True
        self.test_button.disabled = True
        self._clear_result()
        _push(self)
        try:
            raw = await await_handler(self.on_test, self.value)
            result = normalize_test_result(raw)
        except Exception as exc:
            result = {"ok": False, "status": "error", "message": str(exc)}
        self.apply_result(result)
        return result

    def apply_result(self, result: dict) -> None:
        self.result = dict(result)
        status = result.get("status", "failed")
        self.state = "passed" if status == "passed" else ("failed" if result.get("ok") is False else status)
        color = semantic("success") if status == "passed" else (
            ft.Colors.OUTLINE if status == "untestable" else semantic("warning") if status == "rate_limited"
            else ft.Colors.ERROR)
        icon = ft.Icons.CHECK_CIRCLE if status == "passed" else (
            ft.Icons.HELP_OUTLINE if status == "untestable" else ft.Icons.ERROR_OUTLINE)
        self.status_chip.label = ft.Text(TEST_LABELS.get(status, status), color=color)
        self.status_chip.leading = ft.Icon(icon, color=color, size=16)
        self.status_chip.visible = True
        self.message.value = str(result.get("message") or "")
        self.message.visible = bool(self.message.value)
        self.ring.visible = False
        self.test_button.disabled = self.on_test is None
        _push(self)

    async def _on_test(self, e: Any = None) -> None:
        await self.test()


# ---- KeyEditor ------------------------------------------------------------------------------------------

_NUMBER_FIELDS = (
    # key, label, kind, min, max, empty-means
    ("cooldown", "Cooldown (seconds)", "int", 10, 3600, None),
    ("individual_output_token_limit", "Output token limit", "int", 0, 2_000_000, "global limit"),
    ("individual_key_temperature", "Temperature", "float", 0.0, 2.0, "global temperature"),
    ("api_call_delay", "API delay (seconds)", "float", 0.0, 3600.0, "global delay"),
)


def google_credentials_problem(path: str) -> Optional[str]:
    """The desktop credential pickers' check (``settings_rules.google_credentials_error``: a service-account
    JSON with ``type`` and ``project_id``); None when valid or when the rule is not in this build."""
    try:
        from settings_rules import google_credentials_error
    except Exception:
        return None
    return google_credentials_error(path)


def individual_endpoint_problem(enabled: bool, url: str, api_version: str) -> Optional[str]:
    """``key_pool_service.individual_endpoint_error`` (the desktop Individual Endpoint dialog's rules)."""
    try:
        from key_pool_service import individual_endpoint_error
    except Exception:
        return None
    return individual_endpoint_error(enabled, url, api_version)


def endpoint_quick_paste() -> tuple:
    """The KeyEditor's endpoint chips: the desktop dialog's Ollama / LM Studio / TTS / TTS v1 shortcuts as the
    Endpoints page's LAN hosts (``endpoints.QUICK_PASTE["openai_base_url"]``, without Clear)."""
    try:
        from glossarion_mobile.ui.screens.endpoints import QUICK_PASTE
    except Exception:
        return ()
    return tuple((label, value) for label, value in QUICK_PASTE.get("openai_base_url", ()) if value)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value)) if value >= 0 else str(value)
    return str(value)


class KeyEditor(FullScreenEditor):
    """Full-screen form for one key; ``on_save(entry)`` returns None or an error string."""

    def __init__(
        self,
        ctx: Any,
        *,
        entry: Mapping[str, Any],
        pool_id: str = "main",
        pool_title: str = "Translation Keys",
        contexts: Sequence[str] = (),
        context_labels: Optional[Mapping[str, str]] = None,
        on_save: Optional[Callable[[dict], Optional[str]]] = None,
        on_saved: Optional[Callable[[dict], Any]] = None,  # after the editor closed: the confirmation
        on_test: Optional[Callable[[dict], Any]] = None,  # async entry -> result
        sheet_env: Any = None,
        azure_versions: Sequence[str] = (),
        read_clipboard: Optional[Callable[[], Any]] = None,
        copy_text: Optional[Callable[[str], Any]] = None,
        new: bool = False,
        test_reason: Optional[str] = None,
    ) -> None:
        self.entry = dict(entry)
        self.pool_id = pool_id
        self.contexts = list(contexts)
        self.context_labels = dict(context_labels or {})
        self.on_test_entry = on_test
        self.sheet_env = sheet_env
        self.azure_versions = list(azure_versions)
        self.read_clipboard = read_clipboard
        self.copy_text = copy_text
        self.new = new
        self.test_reason = test_reason
        self.number_fields: dict = {}
        self.param_rows: list = []
        self.context_chips: dict = {}
        disabled = {str(c) for c in (self.entry.get("disabled_contexts") or [])}
        self.context_enabled = {c: c not in disabled for c in self.contexts}
        self.extra_disabled = sorted(disabled - set(self.contexts))  # kept untouched
        title = "Add key" if new else "Edit key"
        super().__init__(ctx, title=title, subtitle=pool_title, on_save=on_save, save_label="Add" if new else "Save",
                         on_saved=on_saved)

    # ---- form ---------------------------------------------------------------------------------------

    def build_body(self) -> ft.Control:
        from glossarion_mobile.ui.sheets.model_sheet import ModelPicker

        entry = self.entry
        self.key_field = KeyField(
            value=str(entry.get("api_key") or ""), label="API key",
            on_test=(lambda _value: self.on_test_entry(self.collect_lenient())) if self.on_test_entry else None,
            read_clipboard=self.read_clipboard, copy_text=self.copy_text, test_reason=self.test_reason, key="key-api",
        )
        self.model_picker = ModelPicker(value=str(entry.get("model") or ""), label="Model", env=self.sheet_env,
                                        page=getattr(self.ctx, "page", None), key="key-model")
        self.enabled_switch = ft.Switch(label="Enabled", value=bool(entry.get("enabled", True)))
        numbers: list = []
        for name, label, kind, low, high, empty in _NUMBER_FIELDS:
            value = entry.get(name)
            if name == "cooldown" and value is None:
                value = 60
            if name == "api_call_delay" and not value:
                value = None
            field = ft.TextField(
                value=_fmt(value), label=label, dense=True,
                hint_text=f"Empty = {empty}" if empty else None,
                keyboard_type=ft.KeyboardType.NUMBER, border_radius=tokens.RADII["field"],
                helper=f"{_fmt(low)}–{high:,}" if kind == "int" else f"{_fmt(low)}–{_fmt(high)}",
            )
            self.number_fields[name] = (field, kind, low, high, empty)
            numbers.append(field)
        self.endpoint_switch = ft.Switch(label="Use individual endpoint",
                                         value=bool(entry.get("use_individual_endpoint", False)),
                                         on_change=self._on_endpoint_toggle)
        self.endpoint_field = ft.TextField(value=str(entry.get("azure_endpoint") or ""), label="Endpoint URL",
                                           hint_text="https://…", dense=True, keyboard_type=ft.KeyboardType.URL,
                                           border_radius=tokens.RADII["field"])
        # the desktop dialog's shortcut buttons, as LAN hosts to edit (a phone has no localhost server)
        self.endpoint_chips = ft.Row([ft.Chip(label=ft.Text(label), on_click=lambda e, v=value: self.paste_endpoint(v),
                                              key=f"key-endpoint-paste-{index}")
                                      for index, (label, value) in enumerate(endpoint_quick_paste())],
                                     wrap=True, spacing=6, run_spacing=6, key="key-endpoint-paste")
        version = str(entry.get("azure_api_version") or "2025-01-01-preview")
        if self.azure_versions:
            options = list(dict.fromkeys([version, *self.azure_versions]))
            self.version_field: Any = ft.Dropdown(value=version, label="Azure API version", editable=True,
                                                  options=[ft.DropdownOption(key=v, text=v) for v in options],
                                                  dense=True)
        else:
            self.version_field = ft.TextField(value=version, label="Azure API version", dense=True,
                                              border_radius=tokens.RADII["field"])
        self.endpoint_box = ft.Column([self.endpoint_field, self.endpoint_chips, self.version_field], spacing=8,
                                      tight=True, visible=bool(self.endpoint_switch.value))
        self.creds_field = ft.TextField(value=str(entry.get("google_credentials") or ""),
                                        label="Google credentials (service account JSON path)", dense=True,
                                        border_radius=tokens.RADII["field"])
        self.region_field = ft.TextField(value=str(entry.get("google_region") or ""), label="Google region",
                                         hint_text="us-east5", dense=True, border_radius=tokens.RADII["field"])
        self.params_column = ft.Column([], spacing=6, tight=True)
        params = entry.get("request_parameters") or {}
        if isinstance(params, Mapping):
            for name, value in params.items():
                self._add_param_row(str(name), value, push=False)
        self.add_param_button = ft.TextButton(content="Add parameter", icon=ft.Icons.ADD,
                                              on_click=lambda e: self._add_param_row("", "", push=True))
        context_controls: list = []
        for context in self.contexts:
            chip = ft.Chip(label=ft.Text(self.context_labels.get(context, context)),
                           selected=self.context_enabled.get(context, True),
                           on_click=lambda e, c=context: self.toggle_context(c), key=f"ctx-{context}")
            self.context_chips[context] = chip
            context_controls.append(chip)
        sections: list = [
            self.key_field,
            self.model_picker,
            self.enabled_switch,
            _section("Limits", numbers),
            _section("Individual endpoint", [self.endpoint_switch, self.endpoint_box]),
            _section("Google Cloud", [self.creds_field,
                                      ft.Row([ft.TextButton(content="Import credentials…", icon=ft.Icons.FILE_OPEN_OUTLINED,
                                                            on_click=lambda e: self.open_credentials())]),
                                      self.region_field]),
            _section("🧩 Request parameters", [
                ft.Text("Sent with every request of this key. Plain text is a string; use JSON for numbers, "
                        "booleans and objects.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                        color=ft.Colors.ON_SURFACE_VARIANT),
                self.params_column, self.add_param_button]),
        ]
        if context_controls:
            from glossarion_mobile.ui.screens.keys import context_preset_row

            # Enable all · Disable all · 🖼️ Images only (the desktop context dialog's shortcuts)
            self.context_presets = context_preset_row(self.contexts, self.apply_context_preset, key="key-ctx-presets")
            sections.append(_section("Request contexts", [
                ft.Text("The key serves the selected requests; switch one off to keep this key away from it.",
                        theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT),
                self.context_presets,
                ft.Row(context_controls, wrap=True, spacing=6, run_spacing=6)]))
        return ft.ListView(controls=sections, expand=True, spacing=12, padding=ft.Padding.only(left=4, right=4, bottom=24))

    def open_credentials(self) -> Any:
        """The settings PathEditor (Import… copies the service-account JSON into app data)."""
        from glossarion_mobile.ui.settings.editors import PathEditor

        def save(value: Any) -> Optional[str]:
            path = str(value or "").strip()
            problem = google_credentials_problem(path) if path else None
            if problem:  # the desktop pickers refuse it; the field keeps its value
                return problem
            self.creds_field.value = path
            _push(self.creds_field)
            return None

        extras = getattr(self.ctx, "extras", None) or {}
        self.path_editor = PathEditor(self.ctx, title="Google credentials", value=self.creds_field.value or "",
                                      on_save=save, import_dir=extras.get("import_dir"),
                                      allowed_extensions=["json"]).show()
        return self.path_editor

    def paste_endpoint(self, url: str) -> None:
        self.endpoint_field.value = str(url or "")
        _push(self.endpoint_field)

    def apply_context_preset(self, allowed: Any) -> None:
        """Enable all · Disable all · Images only: each context on exactly when it is in ``allowed``."""
        allowed = set(allowed or ())
        for context in self.contexts:
            self.context_enabled[context] = context in allowed
            chip = self.context_chips.get(context)
            if chip is not None:
                chip.selected = context in allowed
        _push(*self.context_chips.values())

    def _on_endpoint_toggle(self, e: Any = None) -> None:
        self.endpoint_box.visible = bool(self.endpoint_switch.value)
        _push(self.endpoint_box)

    def toggle_context(self, context: str) -> bool:
        value = not self.context_enabled.get(context, True)
        self.context_enabled[context] = value
        chip = self.context_chips.get(context)
        if chip is not None:
            chip.selected = value
            _push(chip)
        return value

    def _add_param_row(self, name: str, value: Any, *, push: bool = True) -> None:
        from glossarion_mobile.ui.screens.keys import request_param_display

        name_field = ft.TextField(value=name, label="Name", dense=True, expand=2, border_radius=tokens.RADII["field"])
        value_field = ft.TextField(value=request_param_display(value) if name or value != "" else "", label="Value",
                                   dense=True, expand=3, border_radius=tokens.RADII["field"])
        row = ft.Row([name_field, value_field], spacing=6, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        remove = ft.IconButton(icon=ft.Icons.REMOVE_CIRCLE_OUTLINE, tooltip="Remove parameter",
                               size_constraints=HIT_TARGET, on_click=lambda e, r=row: self._remove_param_row(r))
        row.controls.append(remove)
        self.param_rows.append((row, name_field, value_field))
        self.params_column.controls.append(row)
        if push:
            _push(self.params_column)

    def _remove_param_row(self, row: Any) -> None:
        self.param_rows = [r for r in self.param_rows if r[0] is not row]
        self.params_column.controls = [r[0] for r in self.param_rows]
        _push(self.params_column)

    # ---- collect --------------------------------------------------------------------------------------

    def _number(self, name: str) -> Any:
        field, kind, low, high, empty = self.number_fields[name]
        text = (field.value or "").strip()
        label = field.label
        if not text:
            if empty is None:
                raise ValueError(f"{label}: enter a number")
            return None
        try:
            number: Any = int(text) if kind == "int" else float(text)
        except ValueError:
            raise ValueError(f"{label}: enter a {'whole ' if kind == 'int' else ''}number") from None
        if number < low or number > high:
            raise ValueError(f"{label}: must be between {_fmt(low)} and {_fmt(high)}")
        return number

    def request_parameters(self) -> dict:
        from glossarion_mobile.ui.screens.keys import parse_request_params

        pairs = [((n.value or "").strip(), v.value or "") for _row, n, v in self.param_rows]
        return parse_request_params(pairs)

    def collect(self) -> dict:
        model = (self.model_picker.value or "").strip()
        if not model:
            raise ValueError("Please enter a model name")
        entry = dict(self.entry)  # keeps unknown fields and the test / usage metadata
        api_key = self.key_field.value
        if self.key_field.encrypted and not api_key:
            api_key = str(self.entry.get("api_key") or "")  # unchanged ENC: value round-trips
        use_endpoint = bool(self.endpoint_switch.value)
        version = (self.version_field.value or "").strip() or "2025-01-01-preview"
        endpoint = (self.endpoint_field.value or "").strip()
        problem = individual_endpoint_problem(use_endpoint, endpoint, version)
        if problem:  # the desktop dialog's Validation Error
            raise ValueError(problem)
        creds = (self.creds_field.value or "").strip()
        if creds and creds != str(self.entry.get("google_credentials") or "").strip():
            problem = google_credentials_problem(creds)
            if problem:  # a typed path: the desktop pickers' check
                raise ValueError(problem)
        cooldown = self._number("cooldown")
        params = self.request_parameters()
        disabled = sorted([c for c, on in self.context_enabled.items() if not on] + self.extra_disabled)
        entry.update({
            "api_key": api_key,
            "model": model,
            "enabled": bool(self.enabled_switch.value),
            "individual_output_token_limit": self._number("individual_output_token_limit"),
            "individual_key_temperature": self._number("individual_key_temperature"),
            "api_call_delay": self._number("api_call_delay") or 0.0,
            "use_individual_endpoint": use_endpoint,
            "azure_endpoint": endpoint or None,
            "azure_api_version": version,
            "google_credentials": creds or None,
            "google_region": (self.region_field.value or "").strip() or None,
        })
        # The Translation pool stores APIKeyEntry.to_dict(); the other pools keep the dict shape
        # of their "Add key" buttons, so optional fields are only written when used.
        if self.pool_id == "main" or "cooldown" in self.entry or cooldown != 60:
            entry["cooldown"] = cooldown
        if params or "request_parameters" in self.entry:
            entry["request_parameters"] = params
        if disabled or "disabled_contexts" in self.entry:
            entry["disabled_contexts"] = disabled
        return entry

    def collect_lenient(self) -> dict:
        """The entry for a Test press (invalid numbers fall back to the stored ones)."""
        try:
            return self.collect()
        except ValueError:
            entry = dict(self.entry)
            entry["api_key"] = self.key_field.value or entry.get("api_key", "")
            entry["model"] = self.model_picker.value or entry.get("model", "")
            return entry


def _section(title: str, controls: Sequence[ft.Control]) -> ft.Control:
    return ft.Container(
        content=ft.Column([ft.Text(title, theme_style=ft.TextThemeStyle.TITLE_SMALL, color=ft.Colors.PRIMARY,
                                   weight=ft.FontWeight.W_600), *controls], spacing=8, tight=True),
        padding=ft.Padding.all(12),
        border_radius=tokens.RADII["card"],
        bgcolor=ft.Colors.SURFACE_CONTAINER_LOW,
    )
