"""Assistant prefill profiles (``/settings/prefill``; UI_SPEC §4.14, desktop "Asst. Prompt").

The desktop "Assistant Prompt (Optional)" dialog on a page: a profile picker
("Default" + named profiles), the profile name, the prompt (``PromptEditorPane``,
no placeholders) with its token count, and **+ New Profile** · **Save Profile** ·
**Delete Profile** · **Clear**. An empty prompt disables the prefill (desktop
"Leave empty to disable this feature (default)").

Semantics come from the shared ``prompt_profiles`` core (the assistant-prefill
equivalents of the dialog's ``new_profile`` / ``save_profile`` / ``delete_profile`` /
``select_profile`` / ``persist_profiles``): "Default" is reserved and cannot be deleted,
renaming keeps the dropdown order, the keys written are exactly the dialog's
``assistant_prompt_profiles`` / ``assistant_prompt_profile_default`` /
``active_assistant_prompt_profile`` / ``assistant_prompt``. ``PrefillCore`` is the only
place that calls the core; ``PrefillService`` writes the changed keys sparsely through
``MobileConfigStore``.
"""

from __future__ import annotations

import copy
import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import flet as ft

from glossarion_mobile.ui.components.dialogs import ConfirmDialog
from glossarion_mobile.ui.screens.base import Screen
from glossarion_mobile.ui.screens.profiles import ProfilesUnavailable, _error_of, _get, bind_call
from glossarion_mobile.ui.screens.prompt_editor import PromptEditorPane

__all__ = ["DEFAULT_LABEL", "PREFILL_CONFIG_KEYS", "PrefillCore", "PrefillList", "PrefillScreen", "PrefillService"]

log = logging.getLogger("glossarion.prefill")

DEFAULT_LABEL = "Default"
#: Keys the desktop dialog's ``persist_profiles`` writes.
PREFILL_CONFIG_KEYS = (
    "assistant_prompt_profiles",
    "assistant_prompt_profile_default",
    "active_assistant_prompt_profile",
    "assistant_prompt",
)


@dataclass
class PrefillList:
    names: list = field(default_factory=list)  # named profiles (without Default)
    texts: dict = field(default_factory=dict)
    default_text: str = ""
    active: str = ""  # "" = Default

    @property
    def options(self) -> list:
        return [DEFAULT_LABEL] + list(self.names)

    def text_for(self, name: str) -> str:
        if not name or name.casefold() == DEFAULT_LABEL.casefold():
            return self.default_text
        return self.texts.get(name, "")


class PrefillCore:
    """Adapter over the assistant-prefill part of the shared ``prompt_profiles`` module.

    Expected core API: ``prefill_state_from_config(config)`` -> state with ``profiles`` /
    ``default_prompt`` / ``active_name``; ``prefill_select(state, name)``,
    ``prefill_save(state, name, text)``, ``prefill_new(state)``, ``prefill_delete(state, name)``
    and ``prefill_apply_to_config(state, config, prompt_text)`` (the dialog's
    ``persist_profiles`` update dict, written into ``config``).
    """

    MODULE = "prompt_profiles"

    def __init__(self, module: Any = None) -> None:
        self._module = module
        self._tried = module is not None
        self.error: Optional[str] = None

    @property
    def module(self) -> Any:
        if not self._tried:
            self._tried = True
            try:
                self._module = importlib.import_module(self.MODULE)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._module = None
        return self._module

    @property
    def available(self) -> bool:
        module = self.module
        return module is not None and callable(getattr(module, "prefill_state_from_config", None) or
                                                getattr(module, "assistant_prefill_state_from_config", None))

    def fn(self, *names: str) -> Callable[..., Any]:
        module = self.module
        if module is None:
            raise ProfilesUnavailable(f"The prompt profiles core is not available in this build ({self.error}).")
        for name in names:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate
        raise ProfilesUnavailable(f"prompt_profiles has no {names[0]}()")

    def _call(self, names: tuple, state: Any, **values: Any) -> Any:
        return bind_call(self.fn(*names), {"state": state, **values})

    def state(self, config: dict) -> Any:
        return self.fn("prefill_state_from_config", "assistant_prefill_state_from_config")(config)

    def listing(self, state: Any) -> PrefillList:
        profiles = dict(_get(state, "profiles", "assistant_prompt_profiles", default={}) or {})
        return PrefillList(
            names=list(profiles),
            texts=profiles,
            default_text=str(_get(state, "default_prompt", "default_text", default="") or ""),
            active=str(_get(state, "active_name", "active", default="") or ""),
        )

    def select(self, state: Any, name: str) -> Any:
        return self._call(("prefill_select", "prefill_select_profile"), state, name=name)

    def save(self, state: Any, name: str, text: str) -> Any:
        return self._call(("prefill_save", "prefill_save_profile"), state, name=name, content=text)

    def new(self, state: Any) -> Any:
        return self._call(("prefill_new", "prefill_new_profile"), state)

    def delete(self, state: Any, name: str) -> Any:
        return self._call(("prefill_delete", "prefill_delete_profile"), state, name=name)

    def persist(self, state: Any, config: dict, prompt_text: str) -> None:
        result = self._call(("prefill_apply_to_config", "prefill_config_updates"), state, config=config,
                            content=prompt_text)
        if isinstance(result, Mapping):
            config.update(result)


class PrefillService:
    def __init__(self, store: Any, *, core: Optional[PrefillCore] = None) -> None:
        self.store = store
        self.core = core or PrefillCore()

    @property
    def available(self) -> bool:
        return self.core.available

    def _config(self) -> dict:
        return self.store.snapshot() if self.store is not None else {}

    def listing(self) -> PrefillList:
        config = self._config()
        try:
            return self.core.listing(self.core.state(config))
        except ProfilesUnavailable:
            stored = config.get("assistant_prompt_profiles")
            profiles = {k: v for k, v in stored.items() if isinstance(k, str) and isinstance(v, str)} \
                if isinstance(stored, Mapping) else {}
            active = config.get("active_assistant_prompt_profile") or ""
            return PrefillList(names=list(profiles), texts=profiles,
                               default_text=str(config.get("assistant_prompt_profile_default",
                                                           config.get("assistant_prompt", "")) or ""),
                               active=active if active in profiles else "")

    def _run(self, op: Callable[[Any], Any], prompt_text: Callable[[Any], str], *,
             string_is_error: bool = False) -> tuple[Any, list]:
        config = self._config()
        before = copy.deepcopy(config)
        state = self.core.state(config)
        result = op(state)
        error = _error_of(result, string_is_error=string_is_error)
        if error:
            raise ValueError(error)
        self.core.persist(state, config, prompt_text(state))
        updates = {key: config[key] for key in PREFILL_CONFIG_KEYS
                   if key in config and (key not in before or before[key] != config[key])}
        changed = self.store.set_many(updates) if updates else []
        return result, changed

    def _active_text(self, state: Any) -> str:
        listing = self.core.listing(state)
        return listing.text_for(listing.active)

    def select(self, name: str) -> list:
        """Switch profile; like the desktop dialog this only takes effect when saved."""
        _r, changed = self._run(lambda state: self.core.select(state, name), self._active_text)
        return changed

    def save(self, name: str, text: str) -> str:
        if not str(name or "").strip():
            raise ValueError("Enter a profile name before saving.")
        self._run(lambda state: self.core.save(state, name.strip(), text), lambda state: str(text or "").strip(),
                  string_is_error=True)
        return name.strip()

    def new(self) -> str:
        result, _changed = self._run(lambda state: self.core.new(state), lambda state: "")
        return result if isinstance(result, str) and result else self.listing().active

    def delete(self, name: str) -> str:
        if not name or name.casefold() == DEFAULT_LABEL.casefold():
            raise ValueError("The Default assistant prompt profile cannot be deleted.")
        self._run(lambda state: self.core.delete(state, name), self._active_text, string_is_error=True)
        return self.listing().active


class PrefillScreen(Screen):
    title = "Assistant prefill"

    def __init__(self, match: Any, ctx: Any, *, service: Optional[PrefillService] = None,
                 model: Callable[[], str] = lambda: "", mono: str = "monospace") -> None:
        super().__init__(match)
        self.ctx = ctx
        self.service = service or PrefillService(getattr(ctx, "store", None))
        self.model = model
        self.mono = mono
        self.listing = PrefillList()
        self.selected = ""
        self.error_text = ft.Text("", color=ft.Colors.ERROR, theme_style=ft.TextThemeStyle.BODY_SMALL, visible=False)
        self.status_text = ft.Text("", theme_style=ft.TextThemeStyle.BODY_SMALL, color=ft.Colors.ON_SURFACE_VARIANT)

    def say(self, message: str) -> None:
        say = getattr(self.ctx, "say", None)
        if callable(say):
            say(message)

    def push(self, *controls: Any) -> None:
        push = getattr(self.ctx, "push", None)
        if callable(push):
            push(*controls)

    def build_body(self) -> ft.Control:
        self.listing = self.service.listing()
        self.selected = self.listing.active or DEFAULT_LABEL
        editable = self.service.available
        self.dropdown = ft.Dropdown(
            label="Assistant Profile", value=self.selected, dense=True, expand=True,
            options=[ft.DropdownOption(key=n, text=n) for n in self.listing.options],
            on_select=lambda e: self.select(e.control.value or DEFAULT_LABEL), disabled=not editable,
        )
        self.name_field = ft.TextField(label="Profile name", value=self.selected, dense=True, read_only=not editable)
        self.editor = PromptEditorPane(value=self.listing.text_for(self.listing.active), placeholders=(), mono=self.mono,
                                       hint="Enter assistant prefill prompt here... (leave empty to disable)",
                                       run_io=getattr(self.ctx, "run_io", None), model=self.model, min_lines=10)
        self.editor.field.read_only = not editable
        self._status()
        rows: list[ft.Control] = [
            ft.Text("This prompt is sent as an 'assistant' message before your content. It can help prime the "
                    "model's response style or format. Leave empty to disable this feature (default).",
                    theme_style=ft.TextThemeStyle.BODY_SMALL),
        ]
        if not editable:
            rows.append(ft.Text("Editing prefill profiles needs the shared prompt_profiles core, which is missing in "
                                "this build. Your values are kept untouched.", theme_style=ft.TextThemeStyle.BODY_SMALL,
                                color=ft.Colors.ERROR, key="prefill-unavailable"))
        rows += [
            self.dropdown,
            self.name_field,
            self.status_text,
            self.editor,
            self.error_text,
            ft.Row([
                ft.FilledButton(content="💾 Save Profile", on_click=lambda e: self.save(), disabled=not editable),
                ft.TextButton(content="+ New Profile", on_click=lambda e: self.new(), disabled=not editable),
                ft.TextButton(content="🗑 Delete Profile", on_click=lambda e: self.confirm_delete(), disabled=not editable,
                              style=ft.ButtonStyle(color=ft.Colors.ERROR)),
                ft.TextButton(content="Clear", on_click=lambda e: self.editor.set_value("", initial=False),
                              disabled=not editable),
            ], wrap=True, spacing=4),
        ]
        return ft.Container(padding=12, expand=True, content=ft.Column(rows, spacing=8, expand=True))

    def _status(self) -> None:
        active = bool((self.editor.value if hasattr(self, "editor") else "").strip())
        self.status_text.value = ("✓ Prefill is active (sent before every request)" if active
                                  else "Prefill is disabled (empty prompt)")

    def _error(self, message: Optional[str]) -> None:
        self.error_text.value = message or ""
        self.error_text.visible = bool(message)
        self.push(self.error_text)

    def _reload(self) -> None:
        self.listing = self.service.listing()
        self.selected = self.listing.active or DEFAULT_LABEL
        self.dropdown.options = [ft.DropdownOption(key=n, text=n) for n in self.listing.options]
        self.dropdown.value = self.selected
        self.name_field.value = self.selected
        self.editor.set_value(self.listing.text_for(self.listing.active))
        self._status()
        self.push(self.dropdown, self.name_field, self.status_text)

    def select(self, name: str) -> None:
        try:
            self.service.select(name)
        except Exception as exc:
            self._error(str(exc))
            return
        self._error(None)
        self._reload()

    def save(self) -> Optional[str]:
        try:
            saved = self.service.save(self.name_field.value or "", self.editor.value)
        except Exception as exc:
            self._error(str(exc))
            return None
        self._error(None)
        self._reload()
        self.say(f"✅ Saved assistant prompt profile: '{saved or DEFAULT_LABEL}'")
        return saved

    def new(self) -> Optional[str]:
        try:
            name = self.service.new()
        except Exception as exc:
            self._error(str(exc))
            return None
        self._reload()
        self.say(f"✅ Created assistant prompt profile: '{name}'")
        return name

    def confirm_delete(self) -> ConfirmDialog:
        name = self.selected
        dialog = ConfirmDialog(title="Delete Profile", body=f"Delete assistant prompt profile '{name}'?",
                               confirm_label="Yes", cancel_label="No", destructive=True,
                               on_confirm=lambda: self.delete(name))
        page = getattr(self.ctx, "page", None)
        if page is not None:
            dialog.show(page)
        return dialog

    def delete(self, name: str) -> Optional[str]:
        try:
            self.service.delete(name)
        except Exception as exc:
            self._error(str(exc))
            return None
        self._reload()
        self.say(f"🗑️ Deleted assistant prompt profile: '{name}'")
        return name
